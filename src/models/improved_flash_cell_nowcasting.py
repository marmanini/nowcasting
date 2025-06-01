# REEMPLAZO COMPLETO para: src/models/improved_flash_cell_nowcasting.py
# Esta versión se entrena automáticamente con datos acumulados

import numpy as np
import pandas as pd
from shapely.geometry import Point
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import geopandas as gpd
import logging
from datetime import datetime, timedelta
import warnings
import os
import pickle
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SmartImprovedFlashCellNowcaster:
    """
    Nowcaster mejorado que se entrena automáticamente con datos históricos.
    """
    
    def __init__(self, forecast_minutes=20, min_history_points=2, 
                 ensemble_models=True, uncertainty_quantification=True,
                 training_data_dir=None):
        """
        Inicializa el nowcaster inteligente.
        
        Args:
            forecast_minutes: Minutos hacia el futuro para predicción
            min_history_points: Puntos mínimos de historial requeridos
            ensemble_models: Si usar ensemble de modelos
            uncertainty_quantification: Si calcular incertidumbre
            training_data_dir: Directorio donde guardar/cargar datos de entrenamiento
        """
        self.forecast_minutes = forecast_minutes
        self.min_history_points = min_history_points
        self.ensemble_models = ensemble_models
        self.uncertainty_quantification = uncertainty_quantification
        
        # Configuración de directorio de entrenamiento
        self.training_data_dir = training_data_dir or './training_data'
        os.makedirs(self.training_data_dir, exist_ok=True)
        
        # Configuración de modelos
        self.models = {
            'linear': LinearRegression(),
            'rf': None  # Se inicializará cuando tengamos datos suficientes
        }
        
        # Estado del entrenamiento
        self.rf_trained = False
        self.training_data = []
        self.min_training_samples = 50  # Mínimo para entrenar RF
        
        # Escalador para normalización
        self.scaler = StandardScaler()
        
        # Estadísticas históricas
        self.historical_errors = []
        self.velocity_stats = {
            'mean_speed': 25.0,
            'std_speed': 15.0,
            'max_speed': 80.0
        }
        
        # Intentar cargar modelo pre-entrenado
        self._load_trained_model()
        
        logger.info(f"Nowcaster inicializado. RF entrenado: {self.rf_trained}")
    
    def _load_trained_model(self):
        """Carga modelo Random Forest pre-entrenado si existe."""
        model_file = os.path.join(self.training_data_dir, 'rf_model.pkl')
        training_file = os.path.join(self.training_data_dir, 'training_data.pkl')
        
        try:
            if os.path.exists(model_file) and os.path.exists(training_file):
                # Cargar modelo
                with open(model_file, 'rb') as f:
                    self.models['rf'] = pickle.load(f)
                
                # Cargar datos de entrenamiento
                with open(training_file, 'rb') as f:
                    self.training_data = pickle.load(f)
                
                self.rf_trained = True
                logger.info(f"✅ Modelo RF cargado con {len(self.training_data)} muestras de entrenamiento")
            
        except Exception as e:
            logger.warning(f"No se pudo cargar modelo pre-entrenado: {e}")
            self.models['rf'] = None
            self.rf_trained = False
    
    def _save_trained_model(self):
        """Guarda el modelo Random Forest entrenado."""
        try:
            model_file = os.path.join(self.training_data_dir, 'rf_model.pkl')
            training_file = os.path.join(self.training_data_dir, 'training_data.pkl')
            
            # Guardar modelo
            with open(model_file, 'wb') as f:
                pickle.dump(self.models['rf'], f)
            
            # Guardar datos de entrenamiento
            with open(training_file, 'wb') as f:
                pickle.dump(self.training_data, f)
            
            logger.info(f"✅ Modelo RF guardado con {len(self.training_data)} muestras")
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {e}")
    
    def add_training_sample(self, features, target_lat, target_lon, target_intensity, target_area):
        """
        Agrega una muestra de entrenamiento y re-entrena si es necesario.
        
        Args:
            features: Diccionario con características extraídas
            target_lat: Latitud real observada
            target_lon: Longitud real observada  
            target_intensity: Intensidad real observada
            target_area: Área real observada
        """
        try:
            # Crear vector de características
            feature_vector = [
                features.get('mean_velocity', 0),
                features.get('velocity_trend', 0),
                features.get('velocity_std', 0),
                features.get('mean_intensity', 0),
                features.get('intensity_trend', 0),
                features.get('intensity_growth_rate', 0),
                features.get('max_intensity', 0),
                features.get('mean_area', 0),
                features.get('area_trend', 0),
                features.get('area_growth_rate', 0),
                features.get('track_age', 0),
                features.get('time_span', 0),
                features.get('direction_lat', 0),
                features.get('direction_lon', 0),
                features.get('direction_consistency', 0),
                self.forecast_minutes / 60.0  # Forecast hours
            ]
            
            # Target: [lat_change, lon_change, intensity_change, area_change]
            current_lat = features.get('current_lat', 0)
            current_lon = features.get('current_lon', 0)
            current_intensity = features.get('current_n_flashes', 1)
            current_area = features.get('current_area', 0.1)
            
            target_vector = [
                target_lat - current_lat,
                target_lon - current_lon,
                target_intensity - current_intensity,
                target_area - current_area
            ]
            
            # Agregar muestra
            self.training_data.append({
                'features': feature_vector,
                'target': target_vector,
                'timestamp': datetime.now()
            })
            
            # Entrenar si tenemos suficientes datos
            if len(self.training_data) >= self.min_training_samples and not self.rf_trained:
                self._train_random_forest()
            
            # Re-entrenar periódicamente si ya está entrenado
            elif self.rf_trained and len(self.training_data) % 25 == 0:  # Cada 25 nuevas muestras
                self._train_random_forest()
                
        except Exception as e:
            logger.error(f"Error agregando muestra de entrenamiento: {e}")
    
    def _train_random_forest(self):
        """Entrena el modelo Random Forest con datos acumulados."""
        try:
            if len(self.training_data) < self.min_training_samples:
                logger.warning(f"Insuficientes datos para entrenar RF: {len(self.training_data)} < {self.min_training_samples}")
                return
            
            logger.info(f"🎓 Entrenando Random Forest con {len(self.training_data)} muestras...")
            
            # Preparar datos de entrenamiento
            X = np.array([sample['features'] for sample in self.training_data])
            y = np.array([sample['target'] for sample in self.training_data])
            
            # Crear y entrenar modelo
            self.models['rf'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            # Entrenar modelo multitarget (predice [lat, lon, intensity, area] simultáneamente)
            self.models['rf'].fit(X, y)
            
            self.rf_trained = True
            
            # Calcular y mostrar métricas de entrenamiento
            train_score = self.models['rf'].score(X, y)
            logger.info(f"✅ Random Forest entrenado. Score: {train_score:.3f}")
            
            # Guardar modelo
            self._save_trained_model()
            
        except Exception as e:
            logger.error(f"Error entrenando Random Forest: {e}")
            self.models['rf'] = None
            self.rf_trained = False
    
    # ... (mantener todas las funciones existentes de extract_features, etc.)
    
    def _extract_features(self, track_history):
        """Extrae características relevantes del historial de track."""
        if len(track_history) < self.min_history_points:
            return None
        
        # ... (código existente de extract_features - mantener igual)
        # [CÓDIGO COMPLETO OMITIDO POR BREVEDAD - USAR EL ORIGINAL]
        
        # Ordenar por tiempo
        history = sorted(track_history, key=lambda x: x.get('timestamp', datetime.min))
        
        features = {}
        
        # 1. Características de posición y movimiento
        positions = [(cell['centroid_lat'], cell['centroid_lon']) for cell in history]
        times = [cell.get('timestamp', datetime.now()) for cell in history]
        
        # Velocidades instantáneas
        velocities = []
        for i in range(1, len(positions)):
            dt = (times[i] - times[i-1]).total_seconds() / 3600.0  # horas
            if dt > 0:
                dlat = positions[i][0] - positions[i-1][0]
                dlon = positions[i][1] - positions[i-1][1]
                vel = np.sqrt(dlat**2 + dlon**2) * 111 / dt
                velocities.append(vel)
        
        if velocities:
            features['mean_velocity'] = np.mean(velocities)
            features['velocity_trend'] = velocities[-1] - velocities[0] if len(velocities) > 1 else 0
            features['velocity_std'] = np.std(velocities) if len(velocities) > 1 else 0
        else:
            features['mean_velocity'] = 0
            features['velocity_trend'] = 0
            features['velocity_std'] = 0
        
        # 2. Características de intensidad
        intensities = [cell.get('n_flashes', 0) for cell in history]
        features['mean_intensity'] = np.mean(intensities)
        features['intensity_trend'] = intensities[-1] - intensities[0]
        features['intensity_growth_rate'] = (intensities[-1] - intensities[0]) / len(intensities)
        features['max_intensity'] = max(intensities)
        
        # 3. Características de tamaño
        areas = [cell.get('area_km2', 0) for cell in history]
        features['mean_area'] = np.mean(areas)
        features['area_trend'] = areas[-1] - areas[0]
        features['area_growth_rate'] = (areas[-1] - areas[0]) / len(areas)
        
        # 4. Características temporales
        features['track_age'] = len(history)
        features['time_span'] = (times[-1] - times[0]).total_seconds() / 60.0
        
        # 5. Características de movimiento direccional
        if len(positions) >= 2:
            total_dlat = positions[-1][0] - positions[0][0]
            total_dlon = positions[-1][1] - positions[0][1]
            features['direction_lat'] = total_dlat
            features['direction_lon'] = total_dlon
            features['direction_consistency'] = self._calculate_direction_consistency(positions)
        else:
            features['direction_lat'] = 0
            features['direction_lon'] = 0
            features['direction_consistency'] = 0
        
        return features
    
    def _calculate_direction_consistency(self, positions):
        """Calcula consistencia de dirección."""
        if len(positions) < 3:
            return 1.0
        
        directions = []
        for i in range(2, len(positions)):
            v1 = np.array([positions[i-1][0] - positions[i-2][0], 
                          positions[i-1][1] - positions[i-2][1]])
            v2 = np.array([positions[i][0] - positions[i-1][0], 
                          positions[i][1] - positions[i-1][1]])
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                directions.append(cos_angle)
        
        return np.mean(directions) if directions else 0
    
    def _ensemble_prediction(self, track_history):
        """Realiza predicción usando ensemble de modelos disponibles."""
        if len(track_history) < self.min_history_points:
            return None
        
        features = self._extract_features(track_history)
        if not features:
            return None
        
        # Agregar información actual
        current_cell = track_history[-1]
        features['current_lat'] = current_cell['centroid_lat']
        features['current_lon'] = current_cell['centroid_lon']
        features['current_n_flashes'] = current_cell.get('n_flashes', 1)
        features['current_area'] = current_cell.get('area_km2', 0.1)
        features['current_energy'] = current_cell.get('total_energy', 0)
        
        forecast_hours = self.forecast_minutes / 60.0
        predictions = []
        
        # 1. Predicción basada en física (siempre disponible)
        physics_pred = self._predict_with_physics_constraints(features, forecast_hours)
        if physics_pred:
            predictions.append(physics_pred)
        
        # 2. Predicción lineal mejorada (siempre disponible)
        if len(track_history) >= 2:
            linear_pred = self._improved_linear_prediction(track_history, forecast_hours)
            if linear_pred:
                predictions.append(linear_pred)
        
        # 3. Predicción con Random Forest (solo si está entrenado)
        if (self.ensemble_models and 
            self.rf_trained and 
            self.models['rf'] is not None and 
            len(track_history) >= 3):
            try:
                rf_pred = self._random_forest_prediction(track_history, features, forecast_hours)
                if rf_pred:
                    predictions.append(rf_pred)
                    logger.debug(f"✅ Usando predicción RF entrenado")
            except Exception as e:
                logger.warning(f"Random Forest prediction failed: {e}")
        
        if not predictions:
            return None
        
        # Combinar predicciones
        final_pred = self._combine_predictions(predictions)
        
        # Agregar información de incertidumbre
        if self.uncertainty_quantification:
            uncertainty = self._calculate_uncertainty(predictions, features)
            final_pred.update(uncertainty)
        
        return final_pred
    
    def _random_forest_prediction(self, track_history, features, forecast_hours):
        """Predicción usando Random Forest ENTRENADO."""
        try:
            # Preparar vector de características
            feature_vector = np.array([[
                features.get('mean_velocity', 0),
                features.get('velocity_trend', 0),
                features.get('velocity_std', 0),
                features.get('mean_intensity', 0),
                features.get('intensity_trend', 0),
                features.get('intensity_growth_rate', 0),
                features.get('max_intensity', 0),
                features.get('mean_area', 0),
                features.get('area_trend', 0),
                features.get('area_growth_rate', 0),
                features.get('track_age', 0),
                features.get('time_span', 0),
                features.get('direction_lat', 0),
                features.get('direction_lon', 0),
                features.get('direction_consistency', 0),
                forecast_hours
            ]])
            
            # Hacer predicción
            prediction = self.models['rf'].predict(feature_vector)[0]
            
            # Interpretar predicción: [lat_change, lon_change, intensity_change, area_change]
            current_cell = track_history[-1]
            
            pred_lat = current_cell['centroid_lat'] + prediction[0]
            pred_lon = current_cell['centroid_lon'] + prediction[1]
            pred_intensity = max(1, current_cell.get('n_flashes', 1) + prediction[2])
            pred_area = max(0.1, current_cell.get('area_km2', 0.1) + prediction[3])
            
            return {
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'pred_n_flashes': int(round(pred_intensity)),
                'pred_area': pred_area,
                'pred_energy': current_cell.get('total_energy', 0),
                'forecast_method': 'random_forest_trained'
            }
            
        except Exception as e:
            logger.warning(f"RF trained prediction failed: {e}")
            return None
    
    # ... (mantener todas las demás funciones existentes)
    
    def _predict_with_physics_constraints(self, features, forecast_hours):
        """Predicción con restricciones físicas."""
        if not features:
            return None
        
        pred_lat = features['direction_lat'] * forecast_hours + features.get('current_lat', 0)
        pred_lon = features['direction_lon'] * forecast_hours + features.get('current_lon', 0)
        
        implied_speed = np.sqrt(features['direction_lat']**2 + features['direction_lon']**2) * 111 / forecast_hours
        if implied_speed > self.velocity_stats['max_speed']:
            scale_factor = self.velocity_stats['max_speed'] / implied_speed
            pred_lat = features.get('current_lat', 0) + features['direction_lat'] * forecast_hours * scale_factor
            pred_lon = features.get('current_lon', 0) + features['direction_lon'] * forecast_hours * scale_factor
        
        current_intensity = features.get('current_n_flashes', 1)
        growth_rate = features.get('intensity_growth_rate', 0)
        
        max_intensity = 1000
        if current_intensity > 0:
            pred_intensity = current_intensity + growth_rate * forecast_hours
            pred_intensity = min(pred_intensity, max_intensity)
            pred_intensity = max(pred_intensity, 1)
        else:
            pred_intensity = 1
        
        current_area = features.get('current_area', 0.1)
        area_growth = features.get('area_growth_rate', 0)
        pred_area = current_area + area_growth * forecast_hours
        pred_area = max(pred_area, 0.1)
        pred_area = min(pred_area, 10000)
        
        return {
            'pred_lat': pred_lat,
            'pred_lon': pred_lon,
            'pred_n_flashes': int(round(pred_intensity)),
            'pred_area': pred_area,
            'pred_energy': features.get('current_energy', 0) * (pred_intensity / current_intensity),
            'forecast_method': 'physics_constrained'
        }
    
    def _improved_linear_prediction(self, track_history, forecast_hours):
        """Predicción lineal mejorada."""
        try:
            n_points = min(len(track_history), 5)
            recent_history = track_history[-n_points:]
            
            times = []
            lats = []
            lons = []
            intensities = []
            areas = []
            
            base_time = recent_history[0].get('timestamp', datetime.now())
            
            for cell in recent_history:
                cell_time = cell.get('timestamp', datetime.now())
                time_hours = (cell_time - base_time).total_seconds() / 3600.0
                times.append(time_hours)
                lats.append(cell['centroid_lat'])
                lons.append(cell['centroid_lon'])
                intensities.append(cell.get('n_flashes', 1))
                areas.append(cell.get('area_km2', 0.1))
            
            times = np.array(times).reshape(-1, 1)
            
            lat_model = LinearRegression().fit(times, lats)
            lon_model = LinearRegression().fit(times, lons)
            
            target_time = times[-1] + forecast_hours
            pred_lat = float(lat_model.predict([[target_time]])[0])
            pred_lon = float(lon_model.predict([[target_time]])[0])
            
            if len(set(intensities)) > 1:
                intensity_model = LinearRegression().fit(times, intensities)
                pred_intensity = max(1, int(round(intensity_model.predict([[target_time]])[0])))
            else:
                pred_intensity = intensities[-1]
            
            if len(set(areas)) > 1:
                area_model = LinearRegression().fit(times, areas)
                pred_area = max(0.1, float(area_model.predict([[target_time]])[0]))
            else:
                pred_area = areas[-1]
            
            velocity_lat = lat_model.coef_[0]
            velocity_lon = lon_model.coef_[0]
            implied_speed = np.sqrt(velocity_lat**2 + velocity_lon**2) * 111
            
            return {
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'pred_n_flashes': pred_intensity,
                'pred_area': pred_area,
                'pred_energy': recent_history[-1].get('total_energy', 0),
                'velocity_lat': velocity_lat,
                'velocity_lon': velocity_lon,
                'implied_speed': implied_speed,
                'forecast_method': 'improved_linear'
            }
            
        except Exception as e:
            logger.warning(f"Improved linear prediction failed: {e}")
            return None
    
    def _combine_predictions(self, predictions):
        """Combina múltiples predicciones."""
        if len(predictions) == 1:
            return predictions[0]
        
        weights = []
        for pred in predictions:
            method = pred.get('forecast_method', 'unknown')
            if method == 'physics_constrained':
                weights.append(0.3)
            elif method == 'improved_linear':
                weights.append(0.3)
            elif method == 'random_forest_trained':
                weights.append(0.4)  # Mayor peso al RF entrenado
            else:
                weights.append(0.1)
        
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        combined = {
            'pred_lat': sum(pred['pred_lat'] * w for pred, w in zip(predictions, weights)),
            'pred_lon': sum(pred['pred_lon'] * w for pred, w in zip(predictions, weights)),
            'pred_n_flashes': int(round(sum(pred['pred_n_flashes'] * w for pred, w in zip(predictions, weights)))),
            'pred_area': sum(pred['pred_area'] * w for pred, w in zip(predictions, weights)),
            'pred_energy': sum(pred['pred_energy'] * w for pred, w in zip(predictions, weights)),
            'forecast_method': 'ensemble'
        }
        
        if 'velocity_lat' in predictions[0]:
            combined['velocity_lat'] = sum(pred.get('velocity_lat', 0) * w for pred, w in zip(predictions, weights))
            combined['velocity_lon'] = sum(pred.get('velocity_lon', 0) * w for pred, w in zip(predictions, weights))
        
        return combined
    
    def _calculate_uncertainty(self, predictions, features):
        """Calcula métricas de incertidumbre."""
        if len(predictions) < 2:
            speed = features.get('mean_velocity', 25)
            uncertainty_km = max(2, speed * 0.1)
            
            return {
                'uncertainty_lat': uncertainty_km / 111,
                'uncertainty_lon': uncertainty_km / 111,
                'uncertainty_intensity': max(1, features.get('current_n_flashes', 10) * 0.2),
                'uncertainty_area': max(0.1, features.get('current_area', 1) * 0.3),
                'confidence_level': 0.8 if self.rf_trained else 0.7  # Mayor confianza con RF
            }
        
        lats = [pred['pred_lat'] for pred in predictions]
        lons = [pred['pred_lon'] for pred in predictions]
        intensities = [pred['pred_n_flashes'] for pred in predictions]
        areas = [pred['pred_area'] for pred in predictions]
        
        uncertainty = {
            'uncertainty_lat': np.std(lats),
            'uncertainty_lon': np.std(lons),
            'uncertainty_intensity': np.std(intensities),
            'uncertainty_area': np.std(areas),
            'prediction_spread': np.sqrt(np.std(lats)**2 + np.std(lons)**2) * 111,
            'confidence_level': max(0.3, 1.0 - np.std(lats) - np.std(lons))
        }
        
        return uncertainty
    
    def predict_cells(self, current_cells, tracked_cells=None):
        """Interfaz principal para predicción."""
        if current_cells.empty:
            logger.warning("No current cells to predict")
            return pd.DataFrame()
        
        if tracked_cells is None:
            tracked_cells = {}
        
        predictions = []
        
        for _, cell in current_cells.iterrows():
            track_id = cell.get('track_id', -1)
            
            if track_id in tracked_cells and len(tracked_cells[track_id]) >= self.min_history_points:
                track_history = tracked_cells[track_id]
                
                prediction = self._ensemble_prediction(track_history)
                
                if prediction:
                    prediction['track_id'] = track_id
                    prediction['last_cell_id'] = cell['cell_id']
                    prediction['last_lat'] = cell['centroid_lat']
                    prediction['last_lon'] = cell['centroid_lon']
                    prediction['last_time'] = cell.get('timestamp', datetime.now())
                    prediction['pred_time'] = prediction['last_time'] + timedelta(minutes=self.forecast_minutes)
                    prediction['last_n_flashes'] = cell['n_flashes']
                    prediction['last_area'] = cell['area_km2']
                    prediction['lead_time_min'] = self.forecast_minutes
                    prediction['rf_trained'] = self.rf_trained  # Indicar si RF está activo
                    
                    predictions.append(prediction)
        
        if not predictions:
            logger.warning("No predictions generated")
            return pd.DataFrame()
        
        logger.info(f"Generadas {len(predictions)} predicciones. RF activo: {self.rf_trained}")
        return pd.DataFrame(predictions)
    
    def create_prediction_geometries(self, predictions_df):
        """Crea geometrías para predicciones."""
        if predictions_df.empty:
            return gpd.GeoDataFrame()
        
        geometries = [Point(row['pred_lon'], row['pred_lat']) for _, row in predictions_df.iterrows()]
        gdf = gpd.GeoDataFrame(predictions_df, geometry=geometries, crs="EPSG:4326")
        
        if 'uncertainty_lat' in predictions_df.columns:
            uncertainty_geoms = []
            for _, row in predictions_df.iterrows():
                center = Point(row['pred_lon'], row['pred_lat'])
                uncertainty_geoms.append(center.buffer(max(row.get('uncertainty_lat', 0.01), 0.01)))
            
            gdf['uncertainty_geometry'] = uncertainty_geoms
        
        return gdf

# Para mantener compatibilidad, crear alias
ImprovedFlashCellNowcaster = SmartImprovedFlashCellNowcaster