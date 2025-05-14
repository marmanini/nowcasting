# src/models/flash_cell_nowcasting.py

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.api import VAR
import logging
from datetime import timedelta

# Configuración del logger
logger = logging.getLogger(__name__)

class FlashCellNowcaster:
    """
    Clase para realizar predicciones de la posición futura de celdas de rayos.
    """
    
    def __init__(self, forecast_minutes=10, min_history_points=3):
        """
        Inicializa el nowcaster de celdas de rayos.
        
        Args:
            forecast_minutes (int): Minutos hacia el futuro para la predicción
            min_history_points (int): Número mínimo de puntos históricos requeridos para la predicción
        """
        self.forecast_minutes = forecast_minutes
        self.min_history_points = min_history_points
    
    def _prepare_track_data(self, track_history):
        """
        Prepara los datos de un track para la predicción.
        
        Args:
            track_history (list): Lista de diccionarios con el historial del track
            
        Returns:
            pandas.DataFrame: DataFrame con los datos del track preparados para la predicción
        """
        # Extraer datos relevantes
        data = []
        
        for cell_info in track_history:
            data.append({
                'timestamp': pd.Timestamp(cell_info['end_time']),
                'lon': cell_info['centroid_lon'],
                'lat': cell_info['centroid_lat'],
                'area': cell_info['area_km2'],
                'n_flashes': cell_info['n_flashes'],
                'total_energy': cell_info['total_energy']
            })
        
        # Crear DataFrame
        df = pd.DataFrame(data)
        
        # Ordenar por timestamp
        df = df.sort_values('timestamp')
        
        # Establecer timestamp como índice
        df = df.set_index('timestamp')
        
        return df
    
    def _forecast_var(self, track_data):
        """
        Realiza predicción usando modelo VAR (Vector Autoregression).
        
        Args:
            track_data (pandas.DataFrame): DataFrame con datos del track
            
        Returns:
            tuple: (predicción de longitud, predicción de latitud, predicción de área,
                   predicción de número de flashes, predicción de energía total)
        """
        try:
            # Crear modelo VAR
            model = VAR(track_data)
            
            # Ajustar modelo (encontrar orden óptimo)
            results = model.fit(maxlags=2, ic='aic')
            
            # Realizar predicción
            forecast = results.forecast(track_data.values, steps=1)
            
            # Extraer valores predichos
            pred_lon, pred_lat, pred_area, pred_n_flashes, pred_energy = forecast[0]
            
            return pred_lon, pred_lat, pred_area, pred_n_flashes, pred_energy
            
        except Exception as e:
            logger.warning(f"Error in VAR forecast: {e}")
            
            # Fallback: usar el último valor conocido
            last_values = track_data.iloc[-1]
            return (
                last_values['lon'],
                last_values['lat'],
                last_values['area'],
                last_values['n_flashes'],
                last_values['total_energy']
            )
    
    def _predict_cell_movement(self, track_history):
        """
        Predice el movimiento futuro de una celda basado en su historial.
        
        Args:
            track_history (list): Lista de diccionarios con el historial del track
            
        Returns:
            dict: Predicción de la celda
        """
        # Si no hay suficientes puntos en el historial, retornar None
        if len(track_history) < self.min_history_points:
            return None
        
        # Preparar datos
        track_data = self._prepare_track_data(track_history)
        
        # Realizar predicción
        pred_lon, pred_lat, pred_area, pred_n_flashes, pred_energy = self._forecast_var(track_data[['lon', 'lat', 'area', 'n_flashes', 'total_energy']])
        
        # Obtener última celda conocida
        last_cell = track_history[-1]
        
        # Crear timestamp para la predicción
        last_time = pd.Timestamp(last_cell['end_time'])
        pred_time = last_time + pd.Timedelta(minutes=self.forecast_minutes)
        
        # Crear diccionario con la predicción
        prediction = {
            'track_id': last_cell['track_id'],
            'last_cell_id': last_cell['cell_id'],
            'last_time': last_time,
            'pred_time': pred_time,
            'last_lon': last_cell['centroid_lon'],
            'last_lat': last_cell['centroid_lat'],
            'pred_lon': pred_lon,
            'pred_lat': pred_lat,
            'last_area': last_cell['area_km2'],
            'pred_area': max(0, pred_area),  # Asegurar que el área no sea negativa
            'last_n_flashes': last_cell['n_flashes'],
            'pred_n_flashes': max(0, int(round(pred_n_flashes))),
            'last_energy': last_cell['total_energy'],
            'pred_energy': max(0, pred_energy),
            'velocity_lon': (pred_lon - last_cell['centroid_lon']) / (self.forecast_minutes / 60),  # grados/hora
            'velocity_lat': (pred_lat - last_cell['centroid_lat']) / (self.forecast_minutes / 60)   # grados/hora
        }
        
        return prediction
    
    def predict_cells(self, tracked_cells, track_history_dict):
        """
        Predice el movimiento futuro de todas las celdas en seguimiento.
        
        Args:
            tracked_cells (geopandas.GeoDataFrame): GeoDataFrame con celdas en seguimiento
            track_history_dict (dict): Diccionario con el historial de los tracks
            
        Returns:
            pandas.DataFrame: DataFrame con predicciones para cada celda
        """
        predictions = []
        
        # Procesar cada track con suficiente historial
        for track_id, track_history in track_history_dict.items():
            # Verificar si hay suficientes puntos en el historial
            if len(track_history) >= self.min_history_points:
                prediction = self._predict_cell_movement(track_history)
                if prediction:
                    predictions.append(prediction)
        
        if not predictions:
            logger.warning("No predictions could be made, not enough track history")
            return pd.DataFrame()
        
        # Crear DataFrame con predicciones
        pred_df = pd.DataFrame(predictions)
        
        return pred_df
    
    def create_prediction_geometries(self, predictions_df):
        """
        Crea geometrías para las predicciones para visualización.
        
        Args:
            predictions_df (pandas.DataFrame): DataFrame con predicciones
            
        Returns:
            geopandas.GeoDataFrame: GeoDataFrame con geometrías para visualización
        """
        if predictions_df.empty:
            return gpd.GeoDataFrame()
        
        # Crear puntos para las posiciones predichas
        geometries = []
        
        for _, pred in predictions_df.iterrows():
            # Crear punto para la posición predicha
            point = Point(pred['pred_lon'], pred['pred_lat'])
            geometries.append(point)
        
        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(
            predictions_df,
            geometry=geometries,
            crs="EPSG:4326"
        )
        
        return gdf