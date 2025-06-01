import numpy as np
import pandas as pd
from shapely.geometry import Point
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.linear_model import LinearRegression
import geopandas as gpd
import logging
from datetime import datetime, timedelta

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FlashCellNowcaster:
    """
    Clase para realizar nowcasting de celdas de tormenta.
    Predice la posición futura de celdas basada en su movimiento histórico.
    """
    
    def __init__(self, forecast_minutes=20, min_history_points=2, max_lags=None):
        """
        Inicializa el nowcaster.
        
        Args:
            forecast_minutes (int): Minutos hacia el futuro para realizar la predicción.
            min_history_points (int): Número mínimo de puntos históricos necesarios.
            max_lags (int): Número máximo de retrasos para el modelo VAR. Si es None, se determinará automáticamente.
        """
        self.forecast_minutes = forecast_minutes
        self.min_history_points = min_history_points
        self.max_lags = max_lags  # Nuevo parámetro para controlar los retrasos en VAR
    
    def predict_cells(self, current_cells, tracked_cells=None):
        """
        Predice el movimiento futuro de celdas de tormenta.
        
        Args:
            current_cells (GeoDataFrame): GeoDataFrame con celdas actuales.
            tracked_cells (dict): Diccionario con historial de celdas por track_id.
            
        Returns:
            DataFrame: DataFrame con predicciones.
        """
        if current_cells.empty:
            logger.warning("No current cells to predict")
            return pd.DataFrame()
        
        if tracked_cells is None:
            tracked_cells = {}
        
        predictions = []
        
        # Procesar cada celda activa
        for _, cell in current_cells.iterrows():
            track_id = cell.get('track_id', -1)
            
            # Verificar si hay suficiente historial para esta celda
            if track_id in tracked_cells and len(tracked_cells[track_id]) >= self.min_history_points:
                track_history = tracked_cells[track_id]
                
                # Realizar predicción
                prediction = self._predict_cell_movement(track_history)
                
                if prediction:
                    # Añadir información de la celda actual
                    prediction['track_id'] = track_id
                    prediction['last_cell_id'] = cell['cell_id']
                    prediction['last_lat'] = cell['centroid_lat']
                    prediction['last_lon'] = cell['centroid_lon']
                    prediction['last_time'] = cell.get('timestamp', datetime.now())
                    prediction['pred_time'] = prediction['last_time'] + timedelta(minutes=self.forecast_minutes)
                    prediction['last_n_flashes'] = cell['n_flashes']
                    prediction['last_area'] = cell['area_km2']
                    prediction['lead_time_min'] = self.forecast_minutes
                    
                    predictions.append(prediction)
        
        if not predictions:
            logger.warning("No predictions generated - insufficient history or no tracked cells")
            return pd.DataFrame()
        
        # Convertir a DataFrame
        return pd.DataFrame(predictions)
    
    def _predict_cell_movement(self, track_history):
        """
        Predice el movimiento de una celda basado en su historial.
        
        Args:
            track_history (list): Lista de diccionarios con historial de la celda.
            
        Returns:
            dict: Predicción con posición futura y características.
        """
        # Ordenar historial por tiempo
        track_history = sorted(track_history, key=lambda x: x.get('timestamp', datetime.min))
        
        # Intentar predicción con VAR primero
        var_pred = self._forecast_var(track_history)
        
        if var_pred:
            return var_pred
        
        # Si VAR falla, usar predicción lineal como fallback
        return self._predict_linear_trend(track_history)
    
    def _forecast_var(self, track_history):
        """
        Realiza predicción usando modelo VAR (Vector Autoregression).
        
        Args:
            track_history (list): Lista de diccionarios con historial de la celda.
            
        Returns:
            dict: Predicción con posición futura y características.
        """
        try:
            # Extraer series temporales
            data = {
                'lat': [cell['centroid_lat'] for cell in track_history],
                'lon': [cell['centroid_lon'] for cell in track_history],
                'n_flashes': [cell.get('n_flashes', 0) for cell in track_history],
                'area': [cell.get('area_km2', 0) for cell in track_history],
                'energy': [cell.get('total_energy', 0) for cell in track_history]
            }
            
            # Crear DataFrame con índice de tiempo
            timestamps = [cell.get('timestamp', datetime.now()) for cell in track_history]
            
            # Asegurar que hay frecuencia explícita
            freq = pd.infer_freq(pd.DatetimeIndex(timestamps))
            if freq is None:
                freq = '10T'  # 10 minutos como valor predeterminado
            
            # Crear DataFrame con índice de tiempo y frecuencia
            endog = pd.DataFrame(data, index=pd.DatetimeIndex(timestamps, freq=freq))
            
            # Verificar si hay columnas constantes
            std_devs = endog.std(axis=0)
            constant_cols = std_devs < 1e-10  # Umbral pequeño para detectar columnas casi constantes
            
            if constant_cols.any():
                # Si todas las columnas son constantes, usar método alternativo
                if constant_cols.all():
                    logger.warning("Todas las columnas son constantes. Usando método alternativo.")
                    return self._predict_linear_trend(track_history)
                
                # Si solo algunas son constantes, añadir un pequeño ruido aleatorio
                np.random.seed(42)  # Para reproducibilidad
                for col in range(len(constant_cols)):
                    if constant_cols[col]:
                        # Añadir un pequeño ruido aleatorio a columnas constantes
                        col_name = endog.columns[col]
                        noise_scale = 0.0001 * (np.abs(endog[col_name]).mean() or 1.0)
                        endog[col_name] = endog[col_name] + np.random.normal(0, noise_scale, len(endog))
            
            # Ajustar modelo VAR
            model = VAR(endog)
            
            # Determinar máximo de lags
            if hasattr(self, 'max_lags') and self.max_lags is not None:
                max_lags = min(self.max_lags, len(endog) // 2)
            else:
                max_lags = min(2, len(endog) // 2)  # Por defecto, máximo 2 o mitad de longitud
            
            if max_lags < 1:
                max_lags = 1  # Mínimo 1 lag
            
            # Ajustar modelo sin tendencia constante
            results = model.fit(maxlags=max_lags, trend='n')
            
            # Realizar predicción
            steps = self.forecast_minutes // 10  # Asumiendo intervalos de 10 minutos
            if steps < 1:
                steps = 1
            
            forecast = results.forecast(endog.values[-results.k_ar:], steps)
            
            # Obtener última predicción
            pred_values = forecast[-1]
            
            # Calcular velocidad implícita
            last_pos = [track_history[-1]['centroid_lat'], track_history[-1]['centroid_lon']]
            pred_pos = [pred_values[0], pred_values[1]]
            
            time_delta_hours = self.forecast_minutes / 60.0
            velocity_lat = (pred_pos[0] - last_pos[0]) / time_delta_hours
            velocity_lon = (pred_pos[1] - last_pos[1]) / time_delta_hours
            
            # Crear diccionario de predicción
            prediction = {
                'pred_lat': float(pred_values[0]),
                'pred_lon': float(pred_values[1]),
                'pred_n_flashes': max(1, int(round(pred_values[2]))),
                'pred_area': max(0.1, float(pred_values[3])),
                'pred_energy': max(0, float(pred_values[4])),
                'velocity_lat': float(velocity_lat),
                'velocity_lon': float(velocity_lon),
                'forecast_method': 'var'
            }
            
            return prediction
            
        except Exception as e:
            logger.warning(f"Error in VAR forecast: {e}")
            # En caso de error, intentar con método alternativo
            return None
    
    def _predict_linear_trend(self, track_history):
        """
        Método alternativo que usa tendencia lineal simple cuando VAR falla.
        
        Args:
            track_history (list): Lista de diccionarios con historial de la celda.
            
        Returns:
            dict: Predicción con posición futura y características.
        """
        try:
            # Calcular tendencia lineal para lat, lon basada en puntos recientes
            n_points = min(len(track_history), 3)  # Usar últimos 3 puntos o menos
            recent_history = track_history[-n_points:]
            
            # Si solo hay un punto, mantener la posición actual
            if len(recent_history) == 1:
                cell = recent_history[0]
                return {
                    'pred_lat': cell['centroid_lat'],
                    'pred_lon': cell['centroid_lon'],
                    'pred_n_flashes': cell.get('n_flashes', 1),
                    'pred_area': cell.get('area_km2', 0.1),
                    'pred_energy': cell.get('total_energy', 0),
                    'velocity_lat': 0.0,
                    'velocity_lon': 0.0,
                    'forecast_method': 'static'
                }
            
            # Extraer tiempos y posiciones
            if all('timestamp' in cell for cell in recent_history):
                # Calcular tiempo en minutos desde la primera celda
                first_time = recent_history[0]['timestamp']
                times = [(cell['timestamp'] - first_time).total_seconds() / 60.0 for cell in recent_history]
            else:
                # Si no hay timestamps, usar índices
                times = list(range(len(recent_history)))
            
            lats = [cell['centroid_lat'] for cell in recent_history]
            lons = [cell['centroid_lon'] for cell in recent_history]
            
            # Datos adicionales para predecir si están disponibles
            flashes = [cell.get('n_flashes', 0) for cell in recent_history]
            areas = [cell.get('area_km2', 0) for cell in recent_history]
            energies = [cell.get('total_energy', 0) for cell in recent_history]
            
            # Calcular tendencia lineal
            times_array = np.array(times).reshape(-1, 1)
            
            # Predecir coordenadas
            lat_model = LinearRegression().fit(times_array, lats)
            lon_model = LinearRegression().fit(times_array, lons)
            
            # Tiempo objetivo (en minutos desde el inicio)
            last_time = times[-1]
            target_time = last_time + self.forecast_minutes
            
            # Predecir posición
            pred_lat = float(lat_model.predict([[target_time]])[0])
            pred_lon = float(lon_model.predict([[target_time]])[0])
            
            # Predecir otros atributos si hay suficientes datos
            if len(flashes) > 1:
                flash_model = LinearRegression().fit(times_array, flashes)
                pred_flashes = max(1, int(round(flash_model.predict([[target_time]])[0])))
            else:
                pred_flashes = flashes[-1]
            
            if len(areas) > 1:
                area_model = LinearRegression().fit(times_array, areas)
                pred_area = max(0.1, float(area_model.predict([[target_time]])[0]))
            else:
                pred_area = areas[-1]
            
            if len(energies) > 1:
                energy_model = LinearRegression().fit(times_array, energies)
                pred_energy = max(0, float(energy_model.predict([[target_time]])[0]))
            else:
                pred_energy = energies[-1]
            
            # Calcular velocidad implícita
            time_delta_hours = self.forecast_minutes / 60.0
            if time_delta_hours > 0.001:
                velocity_lat = (pred_lat - lats[-1]) / time_delta_hours
                velocity_lon = (pred_lon - lons[-1]) / time_delta_hours
            else:
                velocity_lat = 0.0
                velocity_lon = 0.0
            
            # Crear diccionario de predicción
            prediction = {
                'pred_lat': pred_lat,
                'pred_lon': pred_lon,
                'pred_n_flashes': pred_flashes,
                'pred_area': pred_area,
                'pred_energy': pred_energy,
                'velocity_lat': velocity_lat,
                'velocity_lon': velocity_lon,
                'forecast_method': 'linear_trend'
            }
            
            return prediction
            
        except Exception as e:
            logger.warning(f"Error in linear trend forecast: {e}")
            # Si todo falla, devolver la última posición conocida
            last_cell = track_history[-1]
            return {
                'pred_lat': last_cell['centroid_lat'],
                'pred_lon': last_cell['centroid_lon'],
                'pred_n_flashes': last_cell.get('n_flashes', 1),
                'pred_area': last_cell.get('area_km2', 0.1),
                'pred_energy': last_cell.get('total_energy', 0),
                'velocity_lat': 0.0,
                'velocity_lon': 0.0,
                'forecast_method': 'last_known'
            }
    
    def create_prediction_geometries(self, predictions_df):
        """
        Crea geometrías para las predicciones.
        
        Args:
            predictions_df (DataFrame): DataFrame con predicciones.
            
        Returns:
            GeoDataFrame: GeoDataFrame con geometrías de predicciones.
        """
        if predictions_df.empty:
            return gpd.GeoDataFrame()
        
        # Crear geometrías de puntos para las predicciones
        geometries = [Point(row['pred_lon'], row['pred_lat']) for _, row in predictions_df.iterrows()]
        
        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(predictions_df, geometry=geometries, crs="EPSG:4326")
        
        return gdf
    