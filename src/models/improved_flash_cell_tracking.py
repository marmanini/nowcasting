import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from datetime import datetime, timedelta
import logging
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)

class ImprovedFlashCellTracker:
    """
    Versión mejorada del rastreador de celdas con algoritmos más robustos.
    """
    
    def __init__(self, max_distance_km=30, max_speed_kmh=120, 
                 intensity_weight=0.3, size_weight=0.2, shape_weight=0.1,
                 prediction_weight=0.4, overlap_threshold=0.1):
        """
        Inicializa el rastreador mejorado.
        
        Args:
            max_distance_km: Distancia máxima para asociación (km)
            max_speed_kmh: Velocidad máxima realista de tormentas (km/h)
            intensity_weight: Peso para similitud de intensidad
            size_weight: Peso para similitud de tamaño
            shape_weight: Peso para similitud de forma
            prediction_weight: Peso para predicción de posición
            overlap_threshold: Umbral mínimo de solapamiento
        """
        self.max_distance_km = max_distance_km
        self.max_speed_kmh = max_speed_kmh
        self.intensity_weight = intensity_weight
        self.size_weight = size_weight
        self.shape_weight = shape_weight
        self.prediction_weight = prediction_weight
        self.overlap_threshold = overlap_threshold
        
        # Almacenamiento de tracks
        self.tracked_cells = {}
        self.last_track_id = 0
        self.time_delta_minutes = 10  # Ventana temporal típica
        
        # Parámetros adaptativos
        self.adaptive_thresholds = True
        self.min_track_length = 2
        
    def _calculate_haversine_distance(self, lon1, lat1, lon2, lat2):
        """
        Calcula distancia usando fórmula de Haversine más precisa.
        """
        R = 6371.0  # Radio de la Tierra en km
        
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)
        
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        return R * c
    
    def _predict_next_position(self, track_history):
        """
        Predicción mejorada de posición usando múltiples métodos.
        """
        if len(track_history) < 2:
            return None, None, 0.0, 0.0
        
        # Usar las últimas 3 posiciones si están disponibles
        n_points = min(len(track_history), 3)
        recent_history = track_history[-n_points:]
        
        if len(recent_history) == 2:
            # Velocidad simple
            last_cell = recent_history[-1]
            prev_cell = recent_history[-2]
            
            time_diff = (pd.to_datetime(last_cell['end_time']) - 
                        pd.to_datetime(prev_cell['end_time'])).total_seconds() / 3600.0  # horas
            
            if time_diff <= 0.001:
                return None, None, 0.0, 0.0
            
            vel_lon = (last_cell['centroid_lon'] - prev_cell['centroid_lon']) / time_diff
            vel_lat = (last_cell['centroid_lat'] - prev_cell['centroid_lat']) / time_diff
            
        else:
            # Regresión lineal con múltiples puntos
            times = [(pd.to_datetime(cell['end_time']) - 
                     pd.to_datetime(recent_history[0]['end_time'])).total_seconds() / 3600.0 
                     for cell in recent_history]
            
            lons = [cell['centroid_lon'] for cell in recent_history]
            lats = [cell['centroid_lat'] for cell in recent_history]
            
            # Ajuste lineal
            lon_coeff = np.polyfit(times, lons, 1)
            lat_coeff = np.polyfit(times, lats, 1)
            
            vel_lon = lon_coeff[0]  # Pendiente = velocidad
            vel_lat = lat_coeff[0]
        
        # Calcular posición predicha
        forecast_hours = self.time_delta_minutes / 60.0
        pred_lon = recent_history[-1]['centroid_lon'] + vel_lon * forecast_hours
        pred_lat = recent_history[-1]['centroid_lat'] + vel_lat * forecast_hours
        
        return pred_lon, pred_lat, vel_lon, vel_lat
    
    def _calculate_enhanced_similarity(self, current_cell, track_history):
        """
        Función de similitud mejorada basada en múltiples criterios.
        """
        if not track_history:
            return 0.0
        
        last_cell = track_history[-1]
        
        # 1. Factor de distancia básica
        distance = self._calculate_haversine_distance(
            current_cell['centroid_lon'], current_cell['centroid_lat'],
            last_cell['centroid_lon'], last_cell['centroid_lat']
        )
        
        # Verificar límite de velocidad realista
        time_diff_hours = self.time_delta_minutes / 60.0
        implied_speed = distance / time_diff_hours if time_diff_hours > 0 else float('inf')
        
        if implied_speed > self.max_speed_kmh:
            return 0.0  # Velocidad no realista
        
        # Factor de distancia (exponencial)
        distance_factor = np.exp(-distance / (self.max_distance_km / 3))
        
        # 2. Similitud de intensidad
        intensity_ratio = min(current_cell['n_flashes'], last_cell['n_flashes']) / \
                         max(current_cell['n_flashes'], last_cell['n_flashes'])
        
        # 3. Similitud de tamaño
        size_ratio = min(current_cell['area_km2'], last_cell['area_km2']) / \
                    max(current_cell['area_km2'], last_cell['area_km2'])
        
        # 4. Factor de predicción (si tenemos historial suficiente)
        prediction_bonus = 0.0
        if len(track_history) >= 2:
            pred_lon, pred_lat, _, _ = self._predict_next_position(track_history)
            if pred_lon is not None:
                pred_distance = self._calculate_haversine_distance(
                    current_cell['centroid_lon'], current_cell['centroid_lat'],
                    pred_lon, pred_lat
                )
                prediction_bonus = np.exp(-pred_distance / (self.max_distance_km / 2))
        
        # 5. Similitud de forma (basada en ratio de aspectos si está disponible)
        shape_factor = 1.0  # Por defecto
        if 'aspect_ratio' in current_cell and 'aspect_ratio' in last_cell:
            shape_factor = min(current_cell['aspect_ratio'], last_cell['aspect_ratio']) / \
                          max(current_cell['aspect_ratio'], last_cell['aspect_ratio'])
        
        # 6. Factor de solapamiento geométrico
        overlap_factor = 0.0
        try:
            if current_cell['geometry'].intersects(last_cell['geometry']):
                intersection = current_cell['geometry'].intersection(last_cell['geometry']).area
                union = current_cell['geometry'].union(last_cell['geometry']).area
                overlap_factor = intersection / union if union > 0 else 0.0
        except Exception:
            pass
        
        # Combinar factores con pesos
        similarity = (
            distance_factor * (1.0 - self.prediction_weight - self.intensity_weight - 
                              self.size_weight - self.shape_weight) +
            prediction_bonus * self.prediction_weight +
            intensity_ratio * self.intensity_weight +
            size_ratio * self.size_weight +
            shape_factor * self.shape_weight +
            overlap_factor * 0.1  # Factor de solapamiento adicional
        )
        
        # Penalizar si la distancia es muy grande
        if distance > self.max_distance_km:
            similarity *= 0.1
        
        return similarity
    
    def _solve_assignment_problem(self, similarity_matrix):
        """
        Resuelve el problema de asignación usando optimización combinatorial.
        """
        if similarity_matrix.size == 0:
            return []
        
        # Convertir matriz de similitud a matriz de costos
        cost_matrix = 1.0 - similarity_matrix
        
        # Resolver usando el algoritmo húngaro
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Filtrar asignaciones con similitud muy baja
        valid_assignments = []
        for row_idx, col_idx in zip(row_indices, col_indices):
            similarity = similarity_matrix[row_idx, col_idx]
            if similarity > 0.2:  # Umbral mínimo adaptativo
                valid_assignments.append((row_idx, col_idx, similarity))
        
        return valid_assignments
    
    def _adapt_thresholds(self, current_cells, existing_tracks):
        """
        Adapta umbrales basándose en las condiciones actuales.
        """
        if not self.adaptive_thresholds:
            return
        
        # Adaptar basándose en la densidad de celdas
        n_cells = len(current_cells)
        n_tracks = len(existing_tracks)
        
        if n_cells > 20:  # Muchas celdas, ser más restrictivo
            self.max_distance_km = min(self.max_distance_km, 25)
        elif n_cells < 5:  # Pocas celdas, ser más permisivo
            self.max_distance_km = max(self.max_distance_km, 35)
    
    def track_cells(self, current_cells, timestamp):
        """
        Realiza el tracking mejorado de celdas.
        """
        if current_cells.empty:
            logger.warning("No cells to track in current time window")
            return current_cells
        
        # Primera invocación - inicializar todos como nuevos
        if not self.tracked_cells:
            logger.info("First tracking step, initializing all cells as new")
            
            tracked_gdf = current_cells.copy()
            
            # Asignar IDs de seguimiento
            track_ids = []
            for _ in range(len(tracked_gdf)):
                self.last_track_id += 1
                track_ids.append(self.last_track_id)
            
            tracked_gdf['track_id'] = track_ids
            tracked_gdf['first_seen'] = timestamp
            tracked_gdf['age_minutes'] = 0
            tracked_gdf['predicted_lon'] = np.nan
            tracked_gdf['predicted_lat'] = np.nan
            tracked_gdf['prediction_error'] = np.nan
            
            # Almacenar historial
            for idx, row in tracked_gdf.iterrows():
                track_id = row['track_id']
                self.tracked_cells[track_id] = [row.to_dict()]
            
            return tracked_gdf
        
        # Adaptar umbrales si está habilitado
        self._adapt_thresholds(current_cells, self.tracked_cells)
        
        # Calcular matriz de similitud
        current_cells_list = [row for _, row in current_cells.iterrows()]
        active_tracks = list(self.tracked_cells.keys())
        
        if not active_tracks:
            # No hay tracks activos, crear nuevos
            tracked_gdf = current_cells.copy()
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
        
        # Construir matriz de similitud
        similarity_matrix = np.zeros((len(current_cells_list), len(active_tracks)))
        
        for i, current_cell in enumerate(current_cells_list):
            for j, track_id in enumerate(active_tracks):
                track_history = self.tracked_cells[track_id]
                similarity = self._calculate_enhanced_similarity(current_cell, track_history)
                similarity_matrix[i, j] = similarity
        
        # Resolver problema de asignación
        assignments = self._solve_assignment_problem(similarity_matrix)
        
        # Crear DataFrame de salida
        tracked_gdf = current_cells.copy()
        tracked_gdf['track_id'] = -1
        tracked_gdf['first_seen'] = timestamp
        tracked_gdf['age_minutes'] = 0
        tracked_gdf['predicted_lon'] = np.nan
        tracked_gdf['predicted_lat'] = np.nan
        tracked_gdf['prediction_error'] = np.nan
        
        # Procesar asignaciones válidas
        assigned_cells = set()
        assigned_tracks = set()
        
        for cell_idx, track_idx, similarity in assignments:
            track_id = active_tracks[track_idx]
            track_history = self.tracked_cells[track_id]
            first_seen = track_history[0]['first_seen']
            
            # Actualizar célula
            tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('track_id')] = track_id
            tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('first_seen')] = first_seen
            
            # Calcular edad
            if isinstance(first_seen, pd.Timestamp):
                age = (timestamp - first_seen).total_seconds() / 60
            else:
                age = (timestamp - pd.Timestamp(first_seen)).total_seconds() / 60
            tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('age_minutes')] = age
            
            # Agregar predicción y error si existe
            if len(track_history) >= 2:
                pred_lon, pred_lat, _, _ = self._predict_next_position(track_history)
                if pred_lon is not None:
                    tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('predicted_lon')] = pred_lon
                    tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('predicted_lat')] = pred_lat
                    
                    # Calcular error de predicción
                    current_cell = tracked_gdf.iloc[cell_idx]
                    pred_error = self._calculate_haversine_distance(
                        current_cell['centroid_lon'], current_cell['centroid_lat'],
                        pred_lon, pred_lat
                    )
                    tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('prediction_error')] = pred_error
            
            # Actualizar historial
            cell_dict = tracked_gdf.iloc[cell_idx].to_dict()
            cell_dict['timestamp'] = timestamp
            self.tracked_cells[track_id].append(cell_dict)
            
            assigned_cells.add(cell_idx)
            assigned_tracks.add(track_id)
        
        # Crear nuevos tracks para celdas no asignadas
        unassigned_cells = set(range(len(current_cells_list))) - assigned_cells
        for cell_idx in unassigned_cells:
            self.last_track_id += 1
            new_track_id = self.last_track_id
            
            tracked_gdf.iloc[cell_idx, tracked_gdf.columns.get_loc('track_id')] = new_track_id
            
            # Inicializar historial
            cell_dict = tracked_gdf.iloc[cell_idx].to_dict()
            cell_dict['timestamp'] = timestamp
            self.tracked_cells[new_track_id] = [cell_dict]
        
        # Limpiar tracks antiguos (no vistos en más de 30 minutos)
        cutoff_time = timestamp - pd.Timedelta(minutes=30)
        tracks_to_remove = []
        
        for track_id, history in self.tracked_cells.items():
            if track_id not in assigned_tracks:
                last_seen = pd.Timestamp(history[-1].get('timestamp', history[-1].get('end_time')))
                if last_seen < cutoff_time:
                    tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_cells[track_id]
        
        logger.info(f"Tracked {len(assignments)} cells, created {len(unassigned_cells)} new tracks, "
                   f"removed {len(tracks_to_remove)} old tracks")
        
        return tracked_gdf
    
    def get_track_statistics(self):
        """
        Devuelve estadísticas del tracking para diagnóstico.
        """
        if not self.tracked_cells:
            return {}
        
        track_lengths = [len(history) for history in self.tracked_cells.values()]
        
        stats = {
            'total_tracks': len(self.tracked_cells),
            'active_tracks': len(self.tracked_cells),
            'avg_track_length': np.mean(track_lengths),
            'max_track_length': max(track_lengths),
            'tracks_with_predictions': sum(1 for hist in self.tracked_cells.values() if len(hist) >= 2)
        }
        
        return stats