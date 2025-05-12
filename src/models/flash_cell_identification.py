# src/models/flash_cell_identification.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import logging
from shapely.geometry import Polygon, Point
import geopandas as gpd

# Configuración del logger
logger = logging.getLogger(__name__)

class FlashCellIdentifier:
    """
    Clase para identificar celdas de rayos mediante clustering.
    """
    
    def __init__(self, eps=0.05, min_samples=5, use_time_weight=True):
        """
        Inicializa el identificador de celdas de rayos.
        
        Args:
            eps (float): Distancia máxima entre dos muestras para considerarlas del mismo cluster (DBSCAN)
            min_samples (int): Número mínimo de muestras en un vecindario para considerarlas un core point (DBSCAN)
            use_time_weight (bool): Si se debe usar ponderación temporal en el clustering
        """
        self.eps = eps
        self.min_samples = min_samples
        self.use_time_weight = use_time_weight
        
    def normalize_coordinates(self, df):
        """
        Normaliza las coordenadas para el clustering.
        
        Args:
            df (pandas.DataFrame): DataFrame con datos de flashes
            
        Returns:
            numpy.ndarray: Array con coordenadas normalizadas
        """
        # Extraer coordenadas
        X = df[['flash_lon', 'flash_lat']].values
        
        if self.use_time_weight:
            # Convertir timestamps a segundos desde el primer evento
            times = df['time'].values.astype(np.int64) // 10**9  # Convertir a segundos
            min_time = np.min(times)
            times = times - min_time
            
            # Normalizar tiempo a una escala similar a las coordenadas
            # Ajustar factor de escala según sea necesario
            time_scale = 0.001  # Factor de escala temporal
            normalized_times = times * time_scale
            
            # Agregar dimensión temporal a las coordenadas
            X = np.column_stack((X, normalized_times.reshape(-1, 1)))
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        return X_scaled
    
    def identify_cells(self, flash_df):
        """
        Identifica celdas de rayos mediante clustering DBSCAN.
        Filtra los clusters pequeños para mostrar solo los sistemas principales.
        
        Args:
            flash_df (pandas.DataFrame): DataFrame con datos de flashes
            
        Returns:
            pandas.DataFrame: DataFrame original con columna de cluster
            list: Lista de polígonos (convex hull) para cada celda
            dict: Estadísticas de cada celda
        """
        if flash_df.empty:
            logger.warning("Empty flash DataFrame, cannot identify cells")
            return flash_df, [], {}
        
        # Normalizar coordenadas
        X_scaled = self.normalize_coordinates(flash_df)
        
        # Aplicar DBSCAN
        dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Agregar etiquetas de cluster al DataFrame
        flash_df = flash_df.copy()
        flash_df['cluster'] = clusters
        
        # Extraer polígonos y estadísticas para cada cluster
        polygons = []
        cell_stats = {}
        
        # Procesar cada cluster (excepto ruido, que es -1)
        unique_clusters = sorted(set(clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        # Calcular el tamaño de cada cluster
        cluster_sizes = {}
        for cluster_id in unique_clusters:
            cluster_size = np.sum(clusters == cluster_id)
            cluster_sizes[cluster_id] = cluster_size
        
        # Determinar un umbral para considerar si un cluster es significativo
        # Si hay muchos clusters, tomaremos solo el 10% superior en tamaño
        if len(unique_clusters) > 10:
            size_threshold = np.percentile(list(cluster_sizes.values()), 90)
        else:
            # Si hay pocos clusters, usar un valor mínimo absoluto
            size_threshold = max(self.min_samples * 2, 10)
        
        logger.info(f"Using cluster size threshold: {size_threshold} flashes")
        
        # Filtrar clusters significativos
        significant_clusters = [c for c in unique_clusters if cluster_sizes[c] >= size_threshold]
        logger.info(f"Found {len(significant_clusters)} significant clusters out of {len(unique_clusters)} total")
        
        # Procesar solo los clusters significativos
        for cluster_id in significant_clusters:
            # Obtener puntos de este cluster
            cluster_points = flash_df[flash_df['cluster'] == cluster_id]
            
            if len(cluster_points) >= 3:  # Necesitamos al menos 3 puntos para un polígono
                # Extraer coordenadas
                points = cluster_points[['flash_lon', 'flash_lat']].values
                
                try:
                    # Crear convex hull
                    hull = ConvexHull(points)
                    hull_points = points[hull.vertices]
                    
                    # Crear polígono shapely
                    polygon = Polygon(hull_points)
                    polygons.append((cluster_id, polygon))
                    
                    # Calcular estadísticas del cluster
                    stats = {
                        'n_flashes': len(cluster_points),
                        'centroid_lon': np.mean(cluster_points['flash_lon']),
                        'centroid_lat': np.mean(cluster_points['flash_lat']),
                        'total_energy': np.sum(cluster_points['flash_energy']),
                        'area_km2': polygon.area * 111 * 111,  # Aproximación área en km²
                        'start_time': cluster_points['time'].min(),
                        'end_time': cluster_points['time'].max()
                    }
                    
                    cell_stats[cluster_id] = stats
                    
                except Exception as e:
                    logger.warning(f"Error creating convex hull for cluster {cluster_id}: {e}")
                        
        logger.info(f"Identified {len(polygons)} significant flash cells")
        
        return flash_df, polygons, cell_stats
    
    def create_cell_geodataframe(self, polygons, cell_stats):
        """
        Crea un GeoDataFrame con las celdas identificadas.
        
        Args:
            polygons (list): Lista de tuplas (cluster_id, polygon)
            cell_stats (dict): Diccionario con estadísticas de celdas
            
        Returns:
            geopandas.GeoDataFrame: GeoDataFrame con celdas de rayos
        """
        if not polygons:
            return gpd.GeoDataFrame()
        
        # Crear lista de registros
        records = []
        
        for cluster_id, polygon in polygons:
            if cluster_id in cell_stats:
                stats = cell_stats[cluster_id]
                record = {
                    'cell_id': cluster_id,
                    'n_flashes': stats['n_flashes'],
                    'centroid_lon': stats['centroid_lon'],
                    'centroid_lat': stats['centroid_lat'],
                    'total_energy': stats['total_energy'],
                    'area_km2': stats['area_km2'],
                    'start_time': stats['start_time'],
                    'end_time': stats['end_time'],
                    'geometry': polygon
                }
                records.append(record)
        
        # Crear GeoDataFrame
        gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
        
        return gdf