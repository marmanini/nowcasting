import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedNowcastPredictor:
    """
    Clase para implementar algoritmos avanzados de predicción para nowcasting de tormentas.
    """
    
    def __init__(self, model_type='random_forest'):
        """
        Inicializa el predictor con el tipo de modelo especificado.
        
        Args:
            model_type (str): Tipo de modelo a utilizar ('random_forest', 'gradient_boosting', 'lstm')
        """
        self.model_type = model_type
        self.models = {
            'position_lat': None,
            'position_lon': None,
            'intensity': None,
            'area': None
        }
        self.scalers = {}
        self.feature_importances = {}
        self.trained = False
        
    def _prepare_features(self, track_df):
        """
        Prepara características para entrenamiento/predicción a partir de un DataFrame de track.
        
        Args:
            track_df: DataFrame con datos históricos de un track, ordenados por tiempo
            
        Returns:
            DataFrame: DataFrame con características para el modelo
        """
        if len(track_df) < 2:
            return None
        
        # Ordenar por tiempo para asegurar secuencia temporal correcta
        track_df = track_df.sort_values('timestamp')
        
        features_list = []
        
        # Para cada par de registros consecutivos (t y t+1)
        for i in range(len(track_df) - 1):
            current = track_df.iloc[i]
            next_state = track_df.iloc[i + 1]
            
            # Calcular delta de tiempo en minutos
            time_delta = (next_state['timestamp'] - current['timestamp']).total_seconds() / 60.0
            
            # Características básicas del estado actual
            features = {
                # Coordenadas actuales
                'current_lat': current['centroid_lat'],
                'current_lon': current['centroid_lon'],
                
                # Características de celda actual
                'current_intensity': current['n_flashes'],
                'current_area': current['area_km2'],
                'current_energy': current.get('total_energy', 0),
                
                # Tiempo de vida
                'age_minutes': current.get('age_minutes', 0),
                
                # Delta de tiempo
                'time_delta_minutes': time_delta,
            }
            
            # Historial reciente (velocidad y aceleración)
            if i > 0:
                prev = track_df.iloc[i - 1]
                # Velocidad previa
                if time_delta > 0.001:  # Umbral mínimo para evitar división por valores muy pequeños
                    prev_velocity_lat = (current['centroid_lat'] - prev['centroid_lat']) / time_delta
                    prev_velocity_lon = (current['centroid_lon'] - prev['centroid_lon']) / time_delta
                else:
                    # Si el delta de tiempo es demasiado pequeño, usar valores predeterminados
                    prev_velocity_lat = 0.0
                    prev_velocity_lon = 0.0
                
                features.update({
                    'prev_velocity_lat': prev_velocity_lat,
                    'prev_velocity_lon': prev_velocity_lon,
                    # Cambio en intensidad y área
                    'prev_delta_intensity': (current['n_flashes'] - prev['n_flashes']),
                    'prev_delta_area': (current['area_km2'] - prev['area_km2']),
                })
            else:
                # Valores por defecto para el primer registro
                features.update({
                    'prev_velocity_lat': 0,
                    'prev_velocity_lon': 0,
                    'prev_delta_intensity': 0,
                    'prev_delta_area': 0,
                })
            
            # Variables objetivo (lo que queremos predecir para t+1)
            targets = {
                'next_lat': next_state['centroid_lat'],
                'next_lon': next_state['centroid_lon'],
                'next_intensity': next_state['n_flashes'],
                'next_area': next_state['area_km2'],
            }
            
            # Añadir variables objetivo al conjunto de características
            features.update(targets)
            
            # Añadir a la lista
            features_list.append(features)
        
        if not features_list:
            return None
            
        # Convertir a DataFrame
        features_df = pd.DataFrame(features_list)

        # Verificar y limpiar valores infinitos o muy grandes
        for col in features_df.columns:
            # Reemplazar infinitos con NaN
            features_df[col] = features_df[col].replace([np.inf, -np.inf], np.nan)
            # Imputar valores NaN con la media o mediana de la columna
            if features_df[col].isnull().any():
                median_val = features_df[col].median()
                if np.isnan(median_val):  # Si la mediana es NaN, usar 0
                    features_df[col] = features_df[col].fillna(0)
                else:
                    features_df[col] = features_df[col].fillna(median_val)

        return features_df
    
    def _build_model(self, param_grid=None):
        """
        Construye el modelo según el tipo especificado.
        
        Args:
            param_grid: Diccionario opcional con hiperparámetros
            
        Returns:
            Pipeline: Pipeline de scikit-learn con preprocesamiento y modelo
        """
        if self.model_type == 'random_forest':
            params = param_grid or {
                'n_estimators': 100,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            model = RandomForestRegressor(**params)
        
        elif self.model_type == 'gradient_boosting':
            params = param_grid or {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            model = GradientBoostingRegressor(**params)
            
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
        
        # Crear pipeline con escalado de características
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        return pipeline
    
    def train(self, tracks_data):
        """
        Entrena modelos para predecir posición, intensidad y área.
        
        Args:
            tracks_data: DataFrame o lista de DataFrames con datos históricos de tracks
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        logger.info(f"Iniciando entrenamiento con modelo {self.model_type}")
        
        # Preparar datos de todos los tracks
        all_features = []
        
        if isinstance(tracks_data, list):
            # Si recibimos una lista de DataFrames
            for track_df in tracks_data:
                track_features = self._prepare_features(track_df)
                if track_features is not None:
                    all_features.append(track_features)
        else:
            # Si recibimos un DataFrame con múltiples tracks
            for track_id, track_df in tracks_data.groupby('track_id'):
                track_features = self._prepare_features(track_df)
                if track_features is not None:
                    all_features.append(track_features)
        
        if not all_features:
            logger.error("No hay datos suficientes para entrenar el modelo")
            return False
        
        # Combinar todos los datos
        features_df = pd.concat(all_features, ignore_index=True)
        
        logger.info(f"Datos de entrenamiento preparados: {len(features_df)} muestras")
        
        # Definir variables predictoras y objetivos
        X_columns = [col for col in features_df.columns if not col.startswith('next_')]
        
        # Entrenar un modelo para cada variable objetivo
        target_variables = {
            'position_lat': 'next_lat',
            'position_lon': 'next_lon',
            'intensity': 'next_intensity',
            'area': 'next_area'
        }
        
        for model_name, target_col in target_variables.items():
            logger.info(f"Entrenando modelo para {model_name}")
            
            X = features_df[X_columns]
            y = features_df[target_col]
            
            # Dividir datos para validación
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Construir y entrenar modelo
            pipeline = self._build_model()
            pipeline.fit(X_train, y_train)
            
            # Evaluar
            y_pred = pipeline.predict(X_val)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            
            logger.info(f"Modelo {model_name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
            
            # Guardar modelo y su evaluación
            self.models[model_name] = pipeline
            
            # Guardar feature importances si disponible
            if hasattr(pipeline['model'], 'feature_importances_'):
                self.feature_importances[model_name] = {
                    'features': X_columns,
                    'importances': pipeline['model'].feature_importances_
                }
        
        self.trained = True
        logger.info("Entrenamiento completado")
        
        return True
    
    def predict_next_state(self, current_state, time_delta_minutes=10):
        """
        Predice el próximo estado de una celda de tormenta.
        
        Args:
            current_state: Diccionario o Series con estado actual de la celda
            time_delta_minutes: Tiempo en minutos hasta la predicción
            
        Returns:
            dict: Estado predicho
        """
        if not self.trained:
            logger.error("El modelo no ha sido entrenado")
            return None
        
        # Preparar datos de entrada
        input_data = {
            'current_lat': current_state['centroid_lat'],
            'current_lon': current_state['centroid_lon'],
            'current_intensity': current_state['n_flashes'],
            'current_area': current_state['area_km2'],
            'current_energy': current_state.get('total_energy', 0),
            'age_minutes': current_state.get('age_minutes', 0),
            'time_delta_minutes': time_delta_minutes,
            'prev_velocity_lat': current_state.get('prev_velocity_lat', 0),
            'prev_velocity_lon': current_state.get('prev_velocity_lon', 0),
            'prev_delta_intensity': current_state.get('prev_delta_intensity', 0),
            'prev_delta_area': current_state.get('prev_delta_area', 0)
        }
        
        # Convertir a DataFrame
        X = pd.DataFrame([input_data])
        
        # Realizar predicciones para cada variable
        predictions = {}
        
        for model_name, model in self.models.items():
            if model is not None:
                pred_value = model.predict(X)[0]
                
                # Mapear nombre del modelo a nombre de variable predicha
                if model_name == 'position_lat':
                    predictions['pred_lat'] = pred_value
                elif model_name == 'position_lon':
                    predictions['pred_lon'] = pred_value
                elif model_name == 'intensity':
                    predictions['pred_n_flashes'] = max(1, round(pred_value))  # No puede ser negativo
                elif model_name == 'area':
                    predictions['pred_area'] = max(0.1, pred_value)  # No puede ser negativo
        
        # Añadir información adicional
        predictions['lead_time_min'] = time_delta_minutes
        predictions['last_lat'] = current_state['centroid_lat']
        predictions['last_lon'] = current_state['centroid_lon']
        predictions['last_n_flashes'] = current_state['n_flashes']
        predictions['last_area'] = current_state['area_km2']
        
        # Calcular velocidad implícita
        predictions['velocity_lat'] = (predictions['pred_lat'] - current_state['centroid_lat']) / (time_delta_minutes / 60.0)
        predictions['velocity_lon'] = (predictions['pred_lon'] - current_state['centroid_lon']) / (time_delta_minutes / 60.0)
        
        # Calcular tiempos si es posible
        if 'timestamp' in current_state:
            last_time = current_state['timestamp']
            pred_time = last_time + timedelta(minutes=time_delta_minutes)
            
            predictions['last_time'] = last_time
            predictions['pred_time'] = pred_time
        
        return predictions
    
    def predict_trajectory(self, current_state, lead_times_minutes=[10, 20, 30]):
        """
        Predice la trayectoria de una celda para varios tiempos de anticipación.
        
        Args:
            current_state: Diccionario o Series con estado actual de la celda
            lead_times_minutes: Lista de tiempos de anticipación en minutos
            
        Returns:
            list: Lista de estados predichos para cada tiempo
        """
        predictions = []
        
        # Predicción para cada tiempo de anticipación
        for lead_time in lead_times_minutes:
            pred = self.predict_next_state(current_state, lead_time)
            if pred:
                predictions.append(pred)
        
        return predictions
    
    def predict_ensemble(self, current_state, lead_time_minutes=10, n_ensemble=10):
        """
        Genera un ensemble de predicciones para cuantificar incertidumbre.
        Solo funciona con Random Forest.
        
        Args:
            current_state: Diccionario o Series con estado actual de la celda
            lead_time_minutes: Tiempo de anticipación en minutos
            n_ensemble: Número de predicciones en el ensemble
            
        Returns:
            dict: Predicción con estadísticas de incertidumbre
        """
        if not self.trained or self.model_type != 'random_forest':
            logger.error("Ensemble solo está disponible para Random Forest")
            return None
        
        # Crear predicción base
        base_pred = self.predict_next_state(current_state, lead_time_minutes)
        
        # Para Random Forest, podemos acceder a las predicciones de cada árbol
        ensemble_results = {
            'lat_predictions': [],
            'lon_predictions': [],
            'intensity_predictions': [],
            'area_predictions': []
        }
        
        # Preparar datos de entrada
        input_data = {
            'current_lat': current_state['centroid_lat'],
            'current_lon': current_state['centroid_lon'],
            'current_intensity': current_state['n_flashes'],
            'current_area': current_state['area_km2'],
            'current_energy': current_state.get('total_energy', 0),
            'age_minutes': current_state.get('age_minutes', 0),
            'time_delta_minutes': lead_time_minutes,
            'prev_velocity_lat': current_state.get('prev_velocity_lat', 0),
            'prev_velocity_lon': current_state.get('prev_velocity_lon', 0),
            'prev_delta_intensity': current_state.get('prev_delta_intensity', 0),
            'prev_delta_area': current_state.get('prev_delta_area', 0)
        }
        
        # Convertir a DataFrame
        X = pd.DataFrame([input_data])
        
        # Para cada modelo, obtener predicciones de árboles individuales
        for model_name, pipeline in self.models.items():
            if pipeline is None:
                continue
                
            # Obtener modelo tras el escalador
            rf_model = pipeline['model']
            
            # Aplicar escalador primero
            X_scaled = pipeline['scaler'].transform(X)
            
            # Obtener predicciones individuales de cada árbol
            tree_preds = [tree.predict(X_scaled)[0] for tree in rf_model.estimators_]
            
            # Limitar a n_ensemble muestras
            if len(tree_preds) > n_ensemble:
                indices = np.random.choice(len(tree_preds), n_ensemble, replace=False)
                tree_preds = [tree_preds[i] for i in indices]
            
            # Almacenar según modelo
            if model_name == 'position_lat':
                ensemble_results['lat_predictions'] = tree_preds
            elif model_name == 'position_lon':
                ensemble_results['lon_predictions'] = tree_preds
            elif model_name == 'intensity':
                ensemble_results['intensity_predictions'] = [max(1, round(p)) for p in tree_preds]
            elif model_name == 'area':
                ensemble_results['area_predictions'] = [max(0.1, p) for p in tree_preds]
        
        # Calcular estadísticas de ensemble
        result = base_pred.copy()
        
        # Añadir estadísticas de dispersión para cuantificar incertidumbre
        result.update({
            'lat_std': np.std(ensemble_results['lat_predictions']),
            'lon_std': np.std(ensemble_results['lon_predictions']),
            'intensity_std': np.std(ensemble_results['intensity_predictions']),
            'area_std': np.std(ensemble_results['area_predictions']),
            
            # Intervalos de confianza del 90%
            'lat_90ci': [
                np.percentile(ensemble_results['lat_predictions'], 5),
                np.percentile(ensemble_results['lat_predictions'], 95)
            ],
            'lon_90ci': [
                np.percentile(ensemble_results['lon_predictions'], 5),
                np.percentile(ensemble_results['lon_predictions'], 95)
            ],
            'intensity_90ci': [
                np.percentile(ensemble_results['intensity_predictions'], 5),
                np.percentile(ensemble_results['intensity_predictions'], 95)
            ],
            'area_90ci': [
                np.percentile(ensemble_results['area_predictions'], 5),
                np.percentile(ensemble_results['area_predictions'], 95)
            ]
        })
        
        return result
    
    def plot_feature_importances(self, save_path=None):
        """
        Visualiza la importancia de características para cada modelo.
        
        Args:
            save_path: Ruta para guardar la figura generada
        """
        if not self.feature_importances:
            logger.error("No hay importancias de características disponibles")
            return
        
        # Crear figura con subplots para cada modelo
        n_models = len(self.feature_importances)
        fig, axs = plt.subplots(n_models, 1, figsize=(10, 5 * n_models))
        
        if n_models == 1:
            axs = [axs]
        
        for i, (model_name, data) in enumerate(self.feature_importances.items()):
            features = data['features']
            importances = data['importances']
            
            # Ordenar por importancia
            indices = np.argsort(importances)
            
            # Graficar
            axs[i].barh(range(len(features)), importances[indices], align='center')
            axs[i].set_yticks(range(len(features)))
            axs[i].set_yticklabels([features[j] for j in indices])
            axs[i].set_title(f'Importancia de Características - Modelo {model_name}')
            axs[i].set_xlabel('Importancia Relativa')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()

    def save_models(self, output_dir):
        """
        Guarda los modelos entrenados en archivos.
        
        Args:
            output_dir: Directorio para guardar los modelos
            
        Returns:
            list: Rutas a los archivos de modelo guardados
        """
        if not self.trained:
            logger.error("No hay modelos entrenados para guardar")
            return []
        
        # Asegurar que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar cada modelo
        model_paths = []
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        for model_name, model in self.models.items():
            if model is not None:
                model_path = os.path.join(
                    output_dir, 
                    f"nowcast_{model_name}_{self.model_type}_{timestamp}.joblib"
                )
                joblib.dump(model, model_path)
                model_paths.append(model_path)
                logger.info(f"Modelo {model_name} guardado en {model_path}")
        
        # Guardar feature importances
        if self.feature_importances:
            fi_path = os.path.join(
                output_dir,
                f"nowcast_feature_importances_{self.model_type}_{timestamp}.png"
            )
            self.plot_feature_importances(fi_path)
        
        return model_paths
    
    def load_models(self, model_paths):
        """
        Carga modelos desde archivos.
        
        Args:
            model_paths: Diccionario con {nombre_modelo: ruta_archivo}
            
        Returns:
            bool: True si la carga fue exitosa
        """
        for model_name, path in model_paths.items():
            if model_name not in self.models:
                logger.warning(f"Nombre de modelo no reconocido: {model_name}")
                continue
                
            try:
                self.models[model_name] = joblib.load(path)
                logger.info(f"Modelo {model_name} cargado desde {path}")
            except Exception as e:
                logger.error(f"Error cargando modelo {model_name}: {e}")
                return False
        
        self.trained = any(model is not None for model in self.models.values())
        return self.trained


    def train_and_evaluate_models(tracks_data, output_dir="./models"):
        """
        Entrena y evalúa múltiples modelos para nowcasting.
        
        Args:
            tracks_data: DataFrame o lista de DataFrames con datos históricos de tracks
            output_dir: Directorio para guardar modelos y resultados
            
        Returns:
            dict: Resultados de evaluación para cada modelo
        """
        # Asegurar que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Dividir datos en entrenamiento y prueba
        if isinstance(tracks_data, list):
            # Si es una lista de DataFrames, dividir la lista
            n_train = int(len(tracks_data) * 0.8)
            train_data = tracks_data[:n_train]
            test_data = tracks_data[n_train:]
        else:
            # Si es un DataFrame único, dividir por track_ids
            track_ids = tracks_data['track_id'].unique()
            np.random.shuffle(track_ids)
            n_train = int(len(track_ids) * 0.8)
            
            train_track_ids = track_ids[:n_train]
            test_track_ids = track_ids[n_train:]
            
            train_data = tracks_data[tracks_data['track_id'].isin(train_track_ids)]
            test_data = tracks_data[tracks_data['track_id'].isin(test_track_ids)]
        
        # Modelos a evaluar
        model_types = ['random_forest', 'gradient_boosting']
        
        # Resultados
        results = {}
        
        # Entrenar y evaluar cada tipo de modelo
        for model_type in model_types:
            logger.info(f"Evaluando modelo: {model_type}")
            
            # Crear y entrenar modelo
            predictor = AdvancedNowcastPredictor(model_type=model_type)
            if predictor.train(train_data):
                # Evaluar modelo
                evaluation = evaluate_model_performance(predictor, test_data)
                
                # Guardar modelo
                model_paths = predictor.save_models(output_dir)
                
                # Guardar resultados
                results[model_type] = {
                    'evaluation': evaluation,
                    'model_paths': model_paths
                }
                
                # Guardar gráficos de evaluación
                plot_path = os.path.join(output_dir, f"evaluation_{model_type}.png")
                plot_prediction_performance(evaluation, plot_path)
            else:
                logger.error(f"Fallo en entrenamiento de modelo {model_type}")
        
        # Determinar mejor modelo basado en RMSE posicional
        best_model = None
        best_rmse = float('inf')
        
        for model_type, result in results.items():
            eval_metrics = result['evaluation']['metrics']
            position_rmse = eval_metrics.get('position_rmse', float('inf'))
            
            if position_rmse < best_rmse:
                best_rmse = position_rmse
                best_model = model_type
        
        if best_model:
            logger.info(f"Mejor modelo: {best_model} con RMSE posicional de {best_rmse:.2f} km")
            results['best_model'] = best_model
        
        return results
    
    def evaluate_model_performance(predictor, test_data):
        """
        Evalúa el rendimiento de un modelo con datos de prueba.
        
        Args:
            predictor: AdvancedNowcastPredictor entrenado
            test_data: DataFrame o lista de DataFrames con datos de prueba
            
        Returns:
            dict: Métricas y resultados de evaluación
        """
        # Preparar lista para almacenar predicciones y valores reales
        predictions = []
        actuals = []
        tracks_evaluated = 0
        
        # Procesar cada track de prueba
        if isinstance(test_data, list):
            test_tracks = test_data
        else:
            # Agrupar por track_id
            test_tracks = [group for _, group in test_data.groupby('track_id')]
        
        for track_df in test_tracks:
            if len(track_df) < 3:  # Necesitamos al menos 3 puntos
                continue
                
            # Ordenar por tiempo
            track_df = track_df.sort_values('timestamp')
            
            # Para cada punto excepto los 2 últimos
            for i in range(len(track_df) - 2):
                # Estado actual
                current_state = track_df.iloc[i]
                
                # Estado siguiente (para calcular velocidad previa)
                next_state = track_df.iloc[i + 1]
                
                # Estado "futuro" (para validar predicción)
                future_state = track_df.iloc[i + 2]
                
                # Calcular velocidad previa para agregar al estado actual
                time_delta = (next_state['timestamp'] - current_state['timestamp']).total_seconds() / 60.0
                
                current_state_with_velocity = current_state.copy()
                current_state_with_velocity['prev_velocity_lat'] = (next_state['centroid_lat'] - current_state['centroid_lat']) / time_delta
                current_state_with_velocity['prev_velocity_lon'] = (next_state['centroid_lon'] - current_state['centroid_lon']) / time_delta
                current_state_with_velocity['prev_delta_intensity'] = next_state['n_flashes'] - current_state['n_flashes']
                current_state_with_velocity['prev_delta_area'] = next_state['area_km2'] - current_state['area_km2']
                
                # Calcular tiempo entre próximo y futuro para predicción
                future_time_delta = (future_state['timestamp'] - next_state['timestamp']).total_seconds() / 60.0
                
                # Predecir estado futuro
                pred = predictor.predict_next_state(current_state_with_velocity, time_delta_minutes=future_time_delta)
                
                if pred:
                    # Almacenar predicción y valor real
                    predictions.append({
                        'track_id': current_state['track_id'],
                        'timestamp': future_state['timestamp'],
                        'pred_lat': pred['pred_lat'],
                        'pred_lon': pred['pred_lon'],
                        'pred_intensity': pred['pred_n_flashes'],
                        'pred_area': pred['pred_area'],
                        'lead_time_min': future_time_delta
                    })
                    
                    actuals.append({
                        'track_id': current_state['track_id'],
                        'timestamp': future_state['timestamp'],
                        'actual_lat': future_state['centroid_lat'],
                        'actual_lon': future_state['centroid_lon'],
                        'actual_intensity': future_state['n_flashes'],
                        'actual_area': future_state['area_km2'],
                    })
            
            tracks_evaluated += 1
        
        logger.info(f"Evaluación completada para {tracks_evaluated} tracks, {len(predictions)} predicciones")
        
        # Si no hay predicciones suficientes, devolver resultados vacíos
        if len(predictions) < 5:
            logger.warning("Datos insuficientes para una evaluación confiable")
            return {
                'metrics': {},
                'predictions_df': None,
                'actuals_df': None
            }
        
        # Convertir a DataFrames
        predictions_df = pd.DataFrame(predictions)
        actuals_df = pd.DataFrame(actuals)
        
        # Fusionar para cálculo de errores
        combined = pd.merge(
            predictions_df,
            actuals_df,
            on=['track_id', 'timestamp'],
            how='inner'
        )
        
        # Calcular métricas de error
        # 1. Error posicional (distancia en km)
        def haversine_distance(lat1, lon1, lat2, lon2):
            """Calcula distancia Haversine en km entre dos puntos"""
            R = 6371.0  # Radio de la Tierra en km
            
            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)
            
            dlon = lon2_rad - lon1_rad
            dlat = lat2_rad - lat1_rad
            
            a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            return R * c
        
        combined['position_error_km'] = combined.apply(
            lambda row: haversine_distance(
                row['actual_lat'], row['actual_lon'],
                row['pred_lat'], row['pred_lon']
            ),
            axis=1
        )
        
        # 2. Error de intensidad (porcentaje)
        combined['intensity_error_pct'] = combined.apply(
            lambda row: abs(row['actual_intensity'] - row['pred_intensity']) / max(1, row['actual_intensity']) * 100,
            axis=1
        )
        
        # 3. Error de área (porcentaje)
        combined['area_error_pct'] = combined.apply(
            lambda row: abs(row['actual_area'] - row['pred_area']) / max(0.1, row['actual_area']) * 100,
            axis=1
        )
        
        # Calcular métricas agregadas
        metrics = {
            'n_predictions': len(combined),
            'n_tracks': combined['track_id'].nunique(),
            
            # Métricas posicionales
            'position_rmse': np.sqrt(np.mean(combined['position_error_km']**2)),
            'position_mae': np.mean(combined['position_error_km']),
            'position_median_error': np.median(combined['position_error_km']),
            'position_90pct_error': np.percentile(combined['position_error_km'], 90),
            
            # Métricas de intensidad
            'intensity_rmse': np.sqrt(np.mean((combined['actual_intensity'] - combined['pred_intensity'])**2)),
            'intensity_mae': np.mean(abs(combined['actual_intensity'] - combined['pred_intensity'])),
            'intensity_median_error_pct': np.median(combined['intensity_error_pct']),
            
            # Métricas de área
            'area_rmse': np.sqrt(np.mean((combined['actual_area'] - combined['pred_area'])**2)),
            'area_mae': np.mean(abs(combined['actual_area'] - combined['pred_area'])),
            'area_median_error_pct': np.median(combined['area_error_pct'])
        }
        
        # Calcular métricas por tiempo de anticipación
        metrics['error_by_leadtime'] = {}
        
        # Redondear tiempos de anticipación a intervalos de 5 minutos
        combined['lead_time_rounded'] = combined['lead_time_min'].apply(lambda x: round(x / 5) * 5)
        
        for lead_time, group in combined.groupby('lead_time_rounded'):
            metrics['error_by_leadtime'][lead_time] = {
                'count': len(group),
                'position_rmse': np.sqrt(np.mean(group['position_error_km']**2)),
                'position_mae': np.mean(group['position_error_km']),
                'intensity_mae': np.mean(abs(group['actual_intensity'] - group['pred_intensity'])),
                'area_mae': np.mean(abs(group['actual_area'] - group['pred_area']))
            }
        
        logger.info(f"Métricas de evaluación: RMSE posicional = {metrics['position_rmse']:.2f} km")
        
        return {
            'metrics': metrics,
            'predictions_df': predictions_df,
            'actuals_df': actuals_df,
            'combined_df': combined
        }

    def plot_prediction_performance(evaluation_results, save_path=None):
        """
        Genera gráficos de rendimiento de predicción.
        
        Args:
            evaluation_results: Resultados de evaluación de modelo
            save_path: Ruta para guardar la figura generada
        """
        if not evaluation_results or 'metrics' not in evaluation_results:
            logger.error("No hay resultados de evaluación para graficar")
            return
        
        metrics = evaluation_results['metrics']
        combined_df = evaluation_results.get('combined_df')
        
        if combined_df is None or combined_df.empty:
            logger.error("No hay datos combinados para graficar")
            return
        
        # Crear figura con subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Histograma de errores posicionales
        axs[0, 0].hist(combined_df['position_error_km'], bins=20, color='blue', alpha=0.7)
        axs[0, 0].set_title('Distribución de Errores Posicionales')
        axs[0, 0].set_xlabel('Error de Posición (km)')
        axs[0, 0].set_ylabel('Frecuencia')
        axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # Añadir líneas verticales para métricas importantes
        axs[0, 0].axvline(metrics['position_rmse'], color='red', linestyle='--', 
                        label=f"RMSE: {metrics['position_rmse']:.2f} km")
        axs[0, 0].axvline(metrics['position_median_error'], color='green', linestyle='--',
                        label=f"Mediana: {metrics['position_median_error']:.2f} km")
        axs[0, 0].axvline(metrics['position_90pct_error'], color='orange', linestyle='--',
                        label=f"Percentil 90: {metrics['position_90pct_error']:.2f} km")
        axs[0, 0].legend()
        
        # 2. Error vs tiempo de anticipación
        if 'error_by_leadtime' in metrics and metrics['error_by_leadtime']:
            lead_times = sorted(metrics['error_by_leadtime'].keys())
            position_rmses = [metrics['error_by_leadtime'][lt]['position_rmse'] for lt in lead_times]
            
            axs[0, 1].plot(lead_times, position_rmses, 'o-', color='blue', linewidth=2)
            axs[0, 1].set_title('Error Posicional vs Tiempo de Anticipación')
            axs[0, 1].set_xlabel('Tiempo de Anticipación (min)')
            axs[0, 1].set_ylabel('RMSE Posicional (km)')
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Gráfico de dispersión: Predicho vs Real (Intensidad)
        axs[1, 0].scatter(combined_df['actual_intensity'], combined_df['pred_intensity'], 
                        alpha=0.5, color='green')
        
        # Añadir línea de referencia y=x
        max_val = max(combined_df['actual_intensity'].max(), combined_df['pred_intensity'].max())
        axs[1, 0].plot([0, max_val], [0, max_val], 'r--')
        
        axs[1, 0].set_title('Predicho vs Real: Intensidad')
        axs[1, 0].set_xlabel('Intensidad Real (n_flashes)')
        axs[1, 0].set_ylabel('Intensidad Predicha (n_flashes)')
        axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Gráfico de dispersión: Predicho vs Real (Área)
        axs[1, 1].scatter(combined_df['actual_area'], combined_df['pred_area'], 
                        alpha=0.5, color='purple')
        
        # Añadir línea de referencia y=x
        max_val = max(combined_df['actual_area'].max(), combined_df['pred_area'].max())
        axs[1, 1].plot([0, max_val], [0, max_val], 'r--')
        
        axs[1, 1].set_title('Predicho vs Real: Área')
        axs[1, 1].set_xlabel('Área Real (km²)')
        axs[1, 1].set_ylabel('Área Predicha (km²)')
        axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        # Añadir título general
        plt.suptitle(
            f"Evaluación de Modelo - {metrics['n_predictions']} predicciones, {metrics['n_tracks']} tracks",
            fontsize=16
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en {save_path}")
        
        plt.show()


    def load_and_process_track_data(geojson_files):
        """
        Carga y prepara datos de tracks para entrenamiento/evaluación.
        
        Args:
            geojson_files: Lista de rutas a archivos GeoJSON con datos de celdas
            
        Returns:
            DataFrame: DataFrame con datos de tracks procesados
        """
        import geopandas as gpd
        from datetime import datetime
        
        # Ordenar archivos
        geojson_files.sort()
        
        # Datos de celdas por tiempo
        cells_by_time = []
        timestamps = []
        
        # Cargar archivos
        for geojson_file in geojson_files:
            try:
                # Extraer timestamp del nombre de archivo
                filename = os.path.basename(geojson_file)
                time_str = filename.split('_')[-1].split('.')[0]
                
                # Parse timestamp
                year = int(time_str[:4])
                month = int(time_str[4:6])
                day = int(time_str[6:8])
                hour = int(time_str[8:10])
                minute = int(time_str[10:12])
                
                timestamp = datetime(year, month, day, hour, minute)
                timestamps.append(timestamp)
                
                # Cargar GeoJSON
                gdf = gpd.read_file(geojson_file)
                
                # Añadir timestamp
                gdf['timestamp'] = timestamp
                
                cells_by_time.append(gdf)
            except Exception as e:
                logger.error(f"Error procesando archivo {geojson_file}: {e}")
        
        # Si no hay datos suficientes, salir
        if len(cells_by_time) < 3:
            logger.error("Datos insuficientes para procesamiento")
            return None
        
        # Combinar todos los GeoDataFrames
        all_cells = pd.concat(cells_by_time)
        
        # Procesar tracks
        tracks_data = []
        
        # Si no hay track_id, no podemos procesar tracks
        if 'track_id' not in all_cells.columns:
            logger.error("No hay columna track_id en los datos. Se requiere identificación de tracks.")
            return None
        
        # Procesar cada track
        for track_id, track_df in all_cells.groupby('track_id'):
            # Ignorar tracks con menos de 3 puntos
            if len(track_df) < 3:
                continue
                
            # Ordenar por tiempo
            track_df = track_df.sort_values('timestamp')
            
            # Calcular edad en minutos desde la primera observación
            first_time = track_df['timestamp'].min()
            track_df['age_minutes'] = track_df['timestamp'].apply(
                lambda t: (t - first_time).total_seconds() / 60.0
            )
            
            # Si no hay columnas de centroide, calcularlas
            if 'centroid_lat' not in track_df.columns or 'centroid_lon' not in track_df.columns:
                # Usar centroide de geometría si está disponible
                if hasattr(track_df, 'geometry') and not track_df.geometry.isna().all():
                    track_df['centroid_lat'] = track_df.geometry.centroid.y
                    track_df['centroid_lon'] = track_df.geometry.centroid.x
                else:
                    logger.warning(f"No se pueden determinar coordenadas de centroide para track {track_id}")
                    continue
            
            # Añadir a la lista de tracks
            tracks_data.append(track_df)
        
        logger.info(f"Datos procesados: {len(tracks_data)} tracks")
        
        return tracks_data


    def generate_nowcast_predictions(geojson_files, output_dir="./predictions", model_paths=None):
        """
        Genera predicciones de nowcasting utilizando modelos avanzados.
        
        Args:
            geojson_files: Lista de rutas a archivos GeoJSON con datos de celdas
            output_dir: Directorio para guardar predicciones
            model_paths: Diccionario opcional con rutas a modelos pre-entrenados
            
        Returns:
            list: Lista de rutas a archivos de predicción generados
        """
        # Asegurar que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Cargar datos
        tracks_data = load_and_process_track_data(geojson_files)
        
        if not tracks_data:
            logger.error("No se pudieron cargar datos de tracks")
            return []
        
        # Si no se proporcionan modelos pre-entrenados, entrenar nuevos
        if not model_paths:
            # Usar 80% de datos para entrenamiento
            n_train = int(len(tracks_data) * 0.8)
            train_data = tracks_data[:n_train]
            
            # Entrenar modelo Random Forest (mejor rendimiento general)
            predictor = AdvancedNowcastPredictor(model_type='random_forest')
            success = predictor.train(train_data)
            
            if not success:
                logger.error("Error entrenando modelo")
                return []
        else:
            # Cargar modelos pre-entrenados
            predictor = AdvancedNowcastPredictor(model_type='random_forest')
            success = predictor.load_models(model_paths)
            
            if not success:
                logger.error("Error cargando modelos pre-entrenados")
                return []
        
        # Generar predicciones para el último conjunto de datos
        last_cells = None
        last_time = None
        
        # Obtener datos más recientes
        if isinstance(tracks_data, list):
            # Ordenar tracks por su último timestamp
            sorted_tracks = sorted(
                tracks_data, 
                key=lambda df: df['timestamp'].max() if not df.empty else datetime.min,
                reverse=True
            )
            
            if sorted_tracks:
                last_track_time = sorted_tracks[0]['timestamp'].max()
                
                # Obtener todas las celdas del último tiempo
                last_cells = pd.concat([
                    track_df[track_df['timestamp'] == last_track_time]
                    for track_df in tracks_data
                    if last_track_time in track_df['timestamp'].values
                ])
                
                last_time = last_track_time
        else:
            # Si es un DataFrame único
            last_time = tracks_data['timestamp'].max()
            last_cells = tracks_data[tracks_data['timestamp'] == last_time]
        
        if last_cells is None or last_cells.empty:
            logger.error("No se encontraron datos del último tiempo para generar predicciones")
            return []
        
        logger.info(f"Generando predicciones para {len(last_cells)} celdas en tiempo {last_time}")
        
        # Tiempos de predicción (10, 20, 30 minutos en adelante)
        lead_times = [10, 20, 30]
        
        # Generar predicciones para cada tiempo
        predictions_by_leadtime = {}
        
        for lead_time in lead_times:
            predictions = []
            
            for _, cell in last_cells.iterrows():
                # Calcular historial de velocidad si es posible
                cell_with_velocity = cell.copy()
                
                # Buscar celda anterior para este track
                track_id = cell['track_id']
                track_data = [df for df in tracks_data if track_id in df['track_id'].values]
                
                if track_data:
                    track_df = track_data[0]
                    track_df = track_df.sort_values('timestamp')
                    
                    # Buscar el registro anterior
                    prev_cells = track_df[track_df['timestamp'] < cell['timestamp']]
                    
                    if not prev_cells.empty:
                        prev_cell = prev_cells.iloc[-1]
                        time_delta = (cell['timestamp'] - prev_cell['timestamp']).total_seconds() / 60.0
                        
                        # Calcular velocidad previa
                        cell_with_velocity['prev_velocity_lat'] = (cell['centroid_lat'] - prev_cell['centroid_lat']) / time_delta
                        cell_with_velocity['prev_velocity_lon'] = (cell['centroid_lon'] - prev_cell['centroid_lon']) / time_delta
                        cell_with_velocity['prev_delta_intensity'] = cell['n_flashes'] - prev_cell['n_flashes']
                        cell_with_velocity['prev_delta_area'] = cell['area_km2'] - prev_cell['area_km2']
                
                # Generar predicción para este tiempo de anticipación
                pred = predictor.predict_next_state(cell_with_velocity, time_delta_minutes=lead_time)
                
                if pred:
                    # Añadir identificadores
                    pred['track_id'] = cell['track_id']
                    pred['last_cell_id'] = cell['cell_id']
                    
                    # Añadir a lista
                    predictions.append(pred)
            
            # Guardar predicciones para este tiempo
            predictions_by_leadtime[lead_time] = predictions
        
        # Guardar predicciones en archivos
        output_files = []
        
        for lead_time, preds in predictions_by_leadtime.items():
            if not preds:
                continue
                
            # Crear DataFrame
            preds_df = pd.DataFrame(preds)
            
            # Nombre de archivo
            timestamp_str = last_time.strftime('%Y%m%d_%H%M%S')
            filename = f"predictions_{timestamp_str}_lead{lead_time:02d}.csv"
            output_path = os.path.join(output_dir, filename)
            
            # Guardar CSV
            preds_df.to_csv(output_path, index=False)
            logger.info(f"Predicciones para t+{lead_time}min guardadas en {output_path}")
            
            output_files.append(output_path)
        
        return output_files


if __name__ == "__main__":
    import glob
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Entrenar y usar modelos avanzados de nowcasting.')
    parser.add_argument('--action', type=str, choices=['train', 'predict', 'evaluate'], default='predict',
                        help='Acción a realizar: entrenar modelos, generar predicciones o evaluar rendimiento')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--model-dir', type=str, default='models', 
                        help='Directorio para guardar/cargar modelos')
    parser.add_argument('--output-dir', type=str, default='predictions', 
                        help='Directorio para guardar predicciones')
    parser.add_argument('--model-type', type=str, choices=['random_forest', 'gradient_boosting'], default='random_forest',
                        help='Tipo de modelo a usar (para acción predict)')
    parser.add_argument('--eval-dir', type=str, default='evaluation',
                       help='Directorio para guardar resultados de evaluación')
    
    args = parser.parse_args()
    
    # Buscar archivos GeoJSON
    geojson_files = sorted(glob.glob(f"{args.geojson_dir}/cells_*.geojson"))
    
    if not geojson_files:
        logger.error(f"No se encontraron archivos GeoJSON en {args.geojson_dir}")
        exit(1)
        
    logger.info(f"Se encontraron {len(geojson_files)} archivos GeoJSON")
    
    # Ejecutar la acción especificada
    if args.action == 'train':
        # Cargar y procesar datos
        tracks_data = load_and_process_track_data(geojson_files)
        
        if tracks_data:
            # Entrenar y evaluar modelos
            results = train_and_evaluate_models(tracks_data, output_dir=args.model_dir)
            
            # Mostrar resultados
            if 'best_model' in results:
                best_model = results['best_model']
                eval_metrics = results[best_model]['evaluation']['metrics']
                
                print(f"\n=== Mejor modelo: {best_model.upper()} ===")
                print(f"RMSE posicional: {eval_metrics['position_rmse']:.2f} km")
                print(f"Error mediano: {eval_metrics['position_median_error']:.2f} km")
                print(f"RMSE intensidad: {eval_metrics['intensity_rmse']:.2f}")
                print(f"RMSE área: {eval_metrics['area_rmse']:.2f} km²")
                
                # Listar modelos guardados
                print(f"\nModelos guardados en directorio: {args.model_dir}")
                for path in results[best_model]['model_paths']:
                    print(f"- {os.path.basename(path)}")
            else:
                print("No se pudo determinar el mejor modelo")
        else:
            logger.error("No se pudieron procesar datos para entrenamiento")
    
    elif args.action == 'predict':
        # Buscar modelos
        model_paths = {}
        model_files = glob.glob(f"{args.model_dir}/nowcast_*_{args.model_type}_*.joblib")
        
        if not model_files:
            logger.warning(f"No se encontraron modelos pre-entrenados en {args.model_dir}")
            logger.info("Generando predicciones con un nuevo modelo entrenado sobre estos datos")
            
            # Generar predicciones sin modelos pre-entrenados
            prediction_files = generate_nowcast_predictions(
                geojson_files,
                output_dir=args.output_dir
            )
        else:
            # Mapear archivos de modelo a variables
            for path in model_files:
                filename = os.path.basename(path)
                parts = filename.split('_')
                if len(parts) >= 2:
                    model_name = parts[1]  # 'position_lat', 'position_lon', etc.
                    model_paths[model_name] = path
            
            logger.info(f"Usando {len(model_paths)} modelos pre-entrenados del tipo {args.model_type}")
            
            # Generar predicciones con modelos pre-entrenados
            prediction_files = generate_nowcast_predictions(
                geojson_files,
                output_dir=args.output_dir,
                model_paths=model_paths
            )
        
        # Mostrar resultados
        if prediction_files:
            print(f"\nPredicciones generadas y guardadas en: {args.output_dir}")
            for path in prediction_files:
                print(f"- {os.path.basename(path)}")
        else:
            print("No se pudieron generar predicciones")
    
    elif args.action == 'evaluate':
        # Buscar archivos de predicción
        prediction_files = sorted(glob.glob(f"{args.output_dir}/predictions_*.csv"))
        
        if not prediction_files:
            logger.error(f"No se encontraron archivos de predicción en {args.output_dir}")
            exit(1)
        
        # Evaluar predicciones
        logger.info(f"Evaluando {len(prediction_files)} archivos de predicción")
        
        # Importar evaluador si está disponible
        try:
            from src.evaluation.nowcasting_evaluator import evaluate_nowcasting_performance
            
            # Ejecutar evaluación
            eval_results = evaluate_nowcasting_performance(
                geojson_files,
                prediction_files,
                output_dir=args.eval_dir
            )
            
            # Mostrar resultados
            if eval_results and 'aggregated_metrics' in eval_results:
                print("\n=== Resumen de Evaluación de Nowcasting ===")
                print(f"Total de predicciones: {eval_results['aggregated_metrics']['total_predictions']}")
                print(f"Aciertos: {eval_results['aggregated_metrics']['total_hits']}")
                print(f"Falsas alarmas: {eval_results['aggregated_metrics']['total_false_positives']}")
                print(f"Falsos negativos: {eval_results['aggregated_metrics']['total_false_negatives']}")
                print(f"POD: {eval_results['aggregated_metrics']['probability_of_detection']:.3f}")
                print(f"FAR: {eval_results['aggregated_metrics']['false_alarm_ratio']:.3f}")
                print(f"CSI: {eval_results['aggregated_metrics']['critical_success_index']:.3f}")
                print(f"Error medio de posición: {eval_results['aggregated_metrics']['mean_position_error_km']:.2f} km")
                print(f"Error mediano de posición: {eval_results['aggregated_metrics']['median_position_error_km']:.2f} km")
                print(f"\nResultados guardados en: {args.eval_dir}")
            else:
                print("No se pudieron generar resultados de evaluación")
        except ImportError:
            logger.error("No se pudo importar el módulo de evaluación. Asegúrate de que exista 'src/evaluation/nowcasting_evaluator.py'")
            
            # Evaluación básica sin el módulo de evaluación
            print("\nRealizando evaluación básica (sin módulo de evaluación completo)")
            
            # Cargar datos de tracking
            tracks_data = load_and_process_track_data(geojson_files)
            
            if not tracks_data:
                logger.error("No se pudieron cargar datos para evaluación")
                exit(1)
            
            # Obtener últimas posiciones reales
            last_positions = {}
            for track_df in tracks_data:
                if len(track_df) < 2:
                    continue
                
                track_id = track_df['track_id'].iloc[0]
                last_positions[track_id] = track_df.sort_values('timestamp').iloc[-1]
            
            # Cargar predicciones
            predictions = []
            for pred_file in prediction_files:
                pred_df = pd.read_csv(pred_file)
                
                # Convertir columnas de tiempo a datetime
                if 'pred_time' in pred_df.columns:
                    pred_df['pred_time'] = pd.to_datetime(pred_df['pred_time'])
                if 'last_time' in pred_df.columns:
                    pred_df['last_time'] = pd.to_datetime(pred_df['last_time'])
                
                predictions.append(pred_df)
            
            all_predictions = pd.concat(predictions)
            
            # Mostrar estadísticas básicas
            print(f"\nPredicciones generadas para {len(all_predictions)} celdas")
            print(f"Tracks únicos: {all_predictions['track_id'].nunique()}")
            
            lead_times = all_predictions['lead_time_min'].unique()
            print(f"Tiempos de anticipación: {sorted(lead_times)} minutos")
            
            # Nota: una evaluación completa requeriría el módulo NowcastingEvaluator")