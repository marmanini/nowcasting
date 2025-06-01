# src/models/flash_cell_tracking.py

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import logging

# Configuración del logger
logger = logging.getLogger(__name__)

class FlashCellTracker:
    """
    Clase para realizar el seguimiento de celdas de rayos entre ventanas de tiempo.
    """
    
    def __init__(self, max_distance_km=30, time_weight=0.5, overlap_weight=0.5):
        """
        Inicializa el rastreador de celdas.
        
        Args:
            max_distance_km (float): Distancia máxima (km) para considerar que dos celdas son la misma
            time_weight (float): Peso del factor tiempo en el cálculo de similitud
            overlap_weight (float): Peso del solapamiento geométrico en el cálculo de similitud
        """
        self.max_distance_km = max_distance_km
        self.time_weight = time_weight
        self.overlap_weight = overlap_weight
        self.tracked_cells = {}  # Diccionario para almacenar historial de celdas {track_id: [cell_info_t0, cell_info_t1, ...]}
        self.last_track_id = 0
        
    def _calculate_distance(self, lon1, lat1, lon2, lat2):
        """
        Calcula la distancia aproximada en kilómetros entre dos puntos.
        
        Args:
            lon1, lat1: Coordenadas del primer punto
            lon2, lat2: Coordenadas del segundo punto
            
        Returns:
            float: Distancia en kilómetros
        """
        # Convertir a radianes
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        # Radio de la Tierra en km
        R = 6371.0
        
        # Fórmula de Haversine
        dlon = lon2_rad - lon1_rad
        dlat = lat2_rad - lat1_rad
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        distance = R * c
        
        return distance
    

    def _predict_next_position(self, track_history):
        """
        Predice la próxima posición de una tormenta basada en su trayectoria.
        
        Args:
            track_history: Lista de celdas históricas de un track
                
        Returns:
            tuple: (pred_lon, pred_lat) - coordenadas predichas
        """
        if len(track_history) < 2:
            return None, None
        
        # Obtener las últimas dos posiciones
        last_cell = track_history[-1]
        prev_cell = track_history[-2]
        
        # Calcular vector de velocidad
        time_diff = (pd.to_datetime(last_cell['end_time']) - 
                    pd.to_datetime(prev_cell['end_time'])).total_seconds() / 60.0
        
        if time_diff <= 0:
            return None, None
        
        # Calcular velocidad en grados por minuto
        vel_lon = (last_cell['centroid_lon'] - prev_cell['centroid_lon']) / time_diff
        vel_lat = (last_cell['centroid_lat'] - prev_cell['centroid_lat']) / time_diff
        
        # Predecir próxima posición (10 minutos más tarde)
        pred_lon = last_cell['centroid_lon'] + vel_lon * 10
        pred_lat = last_cell['centroid_lat'] + vel_lat * 10
        
        return pred_lon, pred_lat

    def _calculate_similarity(self, cell1, cell2):
        """
        Calcula el índice de similitud entre dos celdas, con mejoras para considerar
        el movimiento típico de tormentas.
        
        Args:
            cell1: Primera celda (celda previa)
            cell2: Segunda celda (celda actual)
            
        Returns:
            float: Índice de similitud (0-1, mayor es más similar)
        """
        # 1. Calcular distancia entre centroides
        distance = self._calculate_distance(
            cell1['centroid_lon'], cell1['centroid_lat'],
            cell2['centroid_lon'], cell2['centroid_lat']
        )
        
        # Si la distancia es extremadamente grande, descartar inmediatamente
        if distance > self.max_distance_km * 1.2:
            return 0.0
        
        # Normalizar distancia (1 = cerca, 0 = lejos)
        # Usar una función exponencial para dar más peso a distancias cortas
        distance_factor = np.exp(-distance / (self.max_distance_km/2))
        
        # 2. Calcular similitud de tamaño (área)
        area_ratio = min(cell1['area_km2'], cell2['area_km2']) / max(cell1['area_km2'], cell2['area_km2'])
        
        # 3. Calcular similitud de intensidad (rayos)
        flash_ratio = min(cell1['n_flashes'], cell2['n_flashes']) / max(cell1['n_flashes'], cell2['n_flashes'])
        
        # 4. Verificar solapamiento solo si están relativamente cerca
        overlap_factor = 0.0
        if distance < self.max_distance_km * 0.7:
            try:
                if cell1['geometry'].intersects(cell2['geometry']):
                    intersection = cell1['geometry'].intersection(cell2['geometry']).area
                    union = cell1['geometry'].union(cell2['geometry']).area
                    overlap_factor = intersection / union
            except:
                pass
        
        # 5. Calcular similitud ponderada
        # La distancia es el factor más importante, luego la intensidad/tamaño
        similarity = (
            0.5 * distance_factor +  # Distancia (50%)
            0.2 * flash_ratio +      # Similitud de intensidad (20%)
            0.1 * area_ratio +       # Similitud de tamaño (10%)
            0.2 * overlap_factor     # Solapamiento (20%)
        )
        
        return similarity
    
    def track_cells(self, current_cells, timestamp):
        """
        Realiza el seguimiento de celdas entre la ventana de tiempo actual y la anterior.
        
        Args:
            current_cells (geopandas.GeoDataFrame): GeoDataFrame con celdas de la ventana actual
            timestamp (datetime): Timestamp de la ventana actual
            
        Returns:
            geopandas.GeoDataFrame: GeoDataFrame con información de seguimiento
        """
        if current_cells.empty:
            logger.warning("No cells to track in current time window")
            return current_cells
        
        # Si es la primera invocación, inicializar todas las celdas como nuevas
        if not self.tracked_cells:
            logger.info("First tracking step, initializing all cells as new")
            
            # Crear copia del GeoDataFrame
            tracked_gdf = current_cells.copy()
            
            # Asignar IDs de seguimiento
            track_ids = []
            for _ in range(len(tracked_gdf)):
                self.last_track_id += 1
                track_ids.append(self.last_track_id)
            
            tracked_gdf['track_id'] = track_ids
            tracked_gdf['first_seen'] = timestamp
            tracked_gdf['age_minutes'] = 0
            
            # Almacenar historial
            for idx, row in tracked_gdf.iterrows():
                track_id = row['track_id']
                self.tracked_cells[track_id] = [row.to_dict()]
            
            return tracked_gdf
        
        # Encontrar correspondencias entre celdas actuales y tracks existentes
        matches = []  # [(current_cell_idx, track_id, similarity), ...]
        
        for current_idx, current_cell in current_cells.iterrows():
            best_match = None
            best_similarity = -1
            
            for track_id, track_history in self.tracked_cells.items():
                # Obtener la celda más reciente de este track
                last_cell = track_history[-1]
                
                # Calcular similitud
                similarity = self._calculate_similarity(current_cell, last_cell)
                
                # Añadir este código nuevo para mejorar la similitud con predicción
                if len(track_history) >= 2:
                    pred_lon, pred_lat = self._predict_next_position(track_history)
                    
                    if pred_lon is not None and pred_lat is not None:
                        # Calcular distancia a la posición predicha
                        pred_distance = self._calculate_distance(
                            current_cell['centroid_lon'], current_cell['centroid_lat'],
                            pred_lon, pred_lat
                        )
                        
                        # Bonus de similitud si está cerca de la posición predicha (hasta 0.2 adicional)
                        if pred_distance < self.max_distance_km * 0.8:
                            bonus = 0.2 * (1 - pred_distance / (self.max_distance_km * 0.8))
                            similarity += bonus
                
                # Actualizar mejor coincidencia
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = track_id
            
            # Y aquí reduce el umbral para ser más permisivo
            if best_similarity > 0.15:  # Cambia de 0.3 a 0.15
                matches.append((current_idx, best_match, best_similarity))
        
        # Resolver conflictos (varias celdas actuales coinciden con el mismo track)
        matched_tracks = set()
        final_matches = []
        
        # Ordenar por similitud descendente
        matches.sort(key=lambda x: x[2], reverse=True)
        
        for current_idx, track_id, similarity in matches:
            if track_id not in matched_tracks:
                final_matches.append((current_idx, track_id))
                matched_tracks.add(track_id)
        
        # Crear copia del GeoDataFrame actual
        tracked_gdf = current_cells.copy()
        tracked_gdf['track_id'] = -1  # Valor por defecto
        tracked_gdf['first_seen'] = timestamp
        tracked_gdf['age_minutes'] = 0
        
        # Procesar coincidencias
        for current_idx, track_id in final_matches:
            # Obtener historial del track
            track_history = self.tracked_cells[track_id]
            first_seen = track_history[0]['first_seen']
            
            # Actualizar GeoDataFrame
            tracked_gdf.at[current_idx, 'track_id'] = track_id
            tracked_gdf.at[current_idx, 'first_seen'] = first_seen
            
            # Calcular edad en minutos
            if isinstance(first_seen, pd.Timestamp):
                age = (timestamp - first_seen).total_seconds() / 60
            else:
                age = (timestamp - pd.Timestamp(first_seen)).total_seconds() / 60
            tracked_gdf.at[current_idx, 'age_minutes'] = age
            
            # Actualizar historial
            self.tracked_cells[track_id].append(tracked_gdf.loc[current_idx].to_dict())
        
        # Asignar nuevos IDs para celdas sin coincidencia
        new_cells = tracked_gdf[tracked_gdf['track_id'] == -1]
        for idx in new_cells.index:
            self.last_track_id += 1
            tracked_gdf.at[idx, 'track_id'] = self.last_track_id
            
            # Almacenar en historial
            self.tracked_cells[self.last_track_id] = [tracked_gdf.loc[idx].to_dict()]
        
        # Limpiar tracks antiguos que no se han visto en mucho tiempo
        cutoff_time = timestamp - pd.Timedelta(minutes=30)
        tracks_to_remove = []
        
        for track_id, history in self.tracked_cells.items():
            last_seen = history[-1]['end_time']
            if pd.Timestamp(last_seen) < cutoff_time:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_cells[track_id]
        
        logger.info(f"Tracked {len(final_matches)} existing cells, identified {len(new_cells)} new cells")
        
        return tracked_gdf
