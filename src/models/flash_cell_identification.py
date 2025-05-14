# src/models/flash_cell_identification.py

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from scipy.spatial import ConvexHull
import logging
from shapely.geometry import Polygon, Point
import geopandas as gpd
from scipy.spatial.distance import cdist
import logging

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
    
    # def identify_cells(self, flash_df):
    #     """
    #     Identifica celdas de rayos mediante clustering DBSCAN.
    #     Garantiza que se detecten todas las tormentas visibles.
        
    #     Args:
    #         flash_df (pandas.DataFrame): DataFrame con datos de flashes
            
    #     Returns:
    #         pandas.DataFrame: DataFrame original con columna de cluster
    #         list: Lista de polígonos (convex hull) para cada celda
    #         dict: Estadísticas de cada celda
    #     """
    #     if flash_df.empty:
    #         logger.warning("Empty flash DataFrame, cannot identify cells")
    #         return flash_df, [], {}
        
    #     # Normalizar coordenadas
    #     X_scaled = self.normalize_coordinates(flash_df)
        
    #     # Aplicar DBSCAN con parámetros actuales
    #     dbscan = DBSCAN(eps=self.eps, min_samples=self.min_samples)
    #     clusters = dbscan.fit_predict(X_scaled)
        
    #     # Agregar etiquetas de cluster al DataFrame
    #     flash_df = flash_df.copy()
    #     flash_df['cluster'] = clusters
        
    #     # Extraer polígonos y estadísticas para cada cluster
    #     polygons = []
    #     cell_stats = {}
        
    #     # Procesar cada cluster (excepto ruido, que es -1)
    #     unique_clusters = sorted(set(clusters))
    #     if -1 in unique_clusters:
    #         unique_clusters.remove(-1)
        
    #     # Contar cuántos flashes hay en cada cluster
    #     cluster_sizes = {}
    #     for cluster_id in unique_clusters:
    #         cluster_size = np.sum(clusters == cluster_id)
    #         cluster_sizes[cluster_id] = cluster_size
        
    #     # Si tenemos muchos clusters pequeños y pocos grandes, ajustar min_samples dinámicamente
    #     if len(unique_clusters) > 0:
    #         sizes = np.array(list(cluster_sizes.values()))
    #         if len(sizes) > 0:
    #             # Si el cluster más grande es 10 veces mayor que el promedio, ajustar dinámicamente
    #             max_size = np.max(sizes)
    #             avg_size = np.mean(sizes)
                
    #             # Calcular un umbral adaptativo para el tamaño mínimo de cluster
    #             if max_size > 10 * avg_size:
    #                 # Situación con un cluster dominante - usar un umbral más bajo
    #                 min_size_threshold = max(self.min_samples, int(avg_size * 0.5))
    #             else:
    #                 # Distribución más equilibrada - usar umbral basado en min_samples
    #                 min_size_threshold = self.min_samples
                    
    #             logger.info(f"Dynamic cluster size threshold: {min_size_threshold} (min_samples: {self.min_samples})")
    #         else:
    #             min_size_threshold = self.min_samples
    #     else:
    #         min_size_threshold = self.min_samples
        
    #     # IMPORTANTE: Procesar TODOS los clusters que superen el umbral mínimo
    #     for cluster_id in unique_clusters:
    #         if cluster_sizes[cluster_id] >= min_size_threshold:
    #             # Obtener puntos de este cluster
    #             cluster_points = flash_df[flash_df['cluster'] == cluster_id]
                
    #             if len(cluster_points) >= 3:  # Necesitamos al menos 3 puntos para un polígono
    #                 # Extraer coordenadas
    #                 points = cluster_points[['flash_lon', 'flash_lat']].values
                    
    #                 try:
    #                     # Crear convex hull
    #                     hull = ConvexHull(points)
    #                     hull_points = points[hull.vertices]
                        
    #                     # Crear polígono shapely
    #                     polygon = Polygon(hull_points)
    #                     polygons.append((cluster_id, polygon))
                        
    #                     # Calcular estadísticas del cluster
    #                     stats = {
    #                         'n_flashes': len(cluster_points),
    #                         'centroid_lon': np.mean(cluster_points['flash_lon']),
    #                         'centroid_lat': np.mean(cluster_points['flash_lat']),
    #                         'total_energy': np.sum(cluster_points['flash_energy']),
    #                         'area_km2': polygon.area * 111 * 111,  # Aproximación área en km²
    #                         'start_time': cluster_points['time'].min(),
    #                         'end_time': cluster_points['time'].max()
    #                     }
                        
    #                     cell_stats[cluster_id] = stats
                        
    #                 except Exception as e:
    #                     logger.warning(f"Error creating convex hull for cluster {cluster_id}: {e}")
        
    #     logger.info(f"Identified {len(polygons)} flash cells out of {len(unique_clusters)} total clusters")
        
    #     # Este mensaje nos ayudará a diagnosticar
    #     logger.info(f"Cluster sizes: {sorted(cluster_sizes.values(), reverse=True)}")
        
    #     # Si no se identificó ningún polígono pero hay clusters, intentar con parámetros más permisivos
    #     if len(polygons) == 0 and len(unique_clusters) > 0:
    #         logger.warning("No polygons created despite having clusters. Using more permissive parameters.")
    #         # Reducir el umbral para crear polígonos
    #         for cluster_id in unique_clusters:
    #             cluster_points = flash_df[flash_df['cluster'] == cluster_id]
    #             if len(cluster_points) >= 3:  # Solo necesitamos 3 puntos para un polígono
    #                 points = cluster_points[['flash_lon', 'flash_lat']].values
    #                 try:
    #                     hull = ConvexHull(points)
    #                     hull_points = points[hull.vertices]
    #                     polygon = Polygon(hull_points)
    #                     polygons.append((cluster_id, polygon))
                        
    #                     stats = {
    #                         'n_flashes': len(cluster_points),
    #                         'centroid_lon': np.mean(cluster_points['flash_lon']),
    #                         'centroid_lat': np.mean(cluster_points['flash_lat']),
    #                         'total_energy': np.sum(cluster_points['flash_energy']),
    #                         'area_km2': polygon.area * 111 * 111,
    #                         'start_time': cluster_points['time'].min(),
    #                         'end_time': cluster_points['time'].max()
    #                     }
                        
    #                     cell_stats[cluster_id] = stats
    #                 except Exception as e:
    #                     logger.warning(f"Error creating convex hull for cluster {cluster_id}: {e}")
        
    #     return flash_df, polygons, cell_stats

    def identify_cells(self, flash_df):
        """
        Identifica TODAS las celdas de rayos visibles, sin importar su tamaño.
        
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
        
        # Usar un eps mucho más pequeño de lo normal (0.01 o incluso menos)
        # Esto evitará agrupar tormentas distintas
        small_eps = min(self.eps, 0.015)  # Usar un valor máximo de 0.015
        min_points = max(3, self.min_samples - 2)  # Reducir el mínimo de puntos
        
        # Normalizar coordenadas (solo espacial, no temporal)
        X = flash_df[['flash_lon', 'flash_lat']].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Aplicar DBSCAN con parámetros muy restrictivos
        dbscan = DBSCAN(eps=small_eps, min_samples=min_points)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Imprimir diagnóstico detallado
        unique_clusters = sorted(set(clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
        
        logger.info(f"DIAGNÓSTICO: Usando eps={small_eps}, min_samples={min_points}")
        logger.info(f"DIAGNÓSTICO: Identificados {len(unique_clusters)} clusters iniciales")
        logger.info(f"DIAGNÓSTICO: {np.sum(clusters == -1)} puntos clasificados como ruido")
        
        # Contar rayos por cluster
        cluster_counts = {}
        for c in unique_clusters:
            count = np.sum(clusters == c)
            cluster_counts[c] = count
        
        # Imprimir información de los clusters más grandes
        sorted_clusters = sorted(cluster_counts.items(), key=lambda x: x[1], reverse=True)
        logger.info(f"DIAGNÓSTICO: Top 10 clusters por tamaño: {sorted_clusters[:10]}")
        
        # Agregar etiquetas de cluster al DataFrame
        flash_df = flash_df.copy()
        flash_df['cluster'] = clusters
        
        # Extraer polígonos y estadísticas
        polygons = []
        cell_stats = {}
        
        # Procesar CADA cluster, sin importar su tamaño
        for cluster_id in unique_clusters:
            # Obtener puntos de este cluster
            cluster_points = flash_df[flash_df['cluster'] == cluster_id]
            
            if len(cluster_points) >= 3:  # Solo necesitamos 3 puntos para un polígono
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
        
        logger.info(f"DIAGNÓSTICO: Creados {len(polygons)} polígonos de celdas")
        
        # Si no se creó ningún polígono, intentar agrupar puntos cercanos manualmente
        if not polygons and len(flash_df) >= 3:
            logger.warning("No storm cells identified with clustering. Trying manual grouping...")
            
            # Identificar grupos de puntos que estén cerca entre sí
            from scipy.spatial.distance import pdist, squareform
            
            # Usar solo un subconjunto si hay demasiados puntos
            max_points = 1000
            sample_df = flash_df
            if len(flash_df) > max_points:
                sample_df = flash_df.sample(max_points)
            
            # Calcular matriz de distancias
            points = sample_df[['flash_lon', 'flash_lat']].values
            distances = squareform(pdist(points))
            
            # Identificar grupos de puntos cercanos
            # Se considera que dos puntos están conectados si están a menos de 0.02 grados
            connected = distances < 0.02
            
            # Para cada punto, encontrar todos los puntos conectados
            groups = []
            visited = set()
            
            for i in range(len(sample_df)):
                if i in visited:
                    continue
                    
                # BFS para encontrar todos los puntos conectados
                group = {i}
                queue = [i]
                visited.add(i)
                
                while queue:
                    node = queue.pop(0)
                    for j in range(len(sample_df)):
                        if j not in visited and connected[node, j]:
                            group.add(j)
                            queue.append(j)
                            visited.add(j)
                
                if len(group) >= 3:
                    groups.append(group)
            
            # Crear polígonos para cada grupo
            for i, group in enumerate(groups):
                group_points = points[list(group)]
                
                try:
                    hull = ConvexHull(group_points)
                    hull_points = group_points[hull.vertices]
                    polygon = Polygon(hull_points)
                    
                    # Usar índices negativos para diferenciar estos polígonos manuales
                    manual_id = -(i + 1)
                    polygons.append((manual_id, polygon))
                    
                    stats = {
                        'n_flashes': len(group),
                        'centroid_lon': np.mean(group_points[:, 0]),
                        'centroid_lat': np.mean(group_points[:, 1]),
                        'total_energy': 0,  # No tenemos esta info para los puntos muestreados
                        'area_km2': polygon.area * 111 * 111,
                        'start_time': sample_df.iloc[list(group)[0]]['time'],
                        'end_time': sample_df.iloc[list(group)[0]]['time']
                    }
                    
                    cell_stats[manual_id] = stats
                    
                except Exception as e:
                    logger.warning(f"Error creating manual hull for group {i}: {e}")
            
            logger.info(f"DIAGNÓSTICO: Creados {len(polygons)} polígonos manualmente")
        
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