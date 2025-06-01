import folium
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from folium.plugins import HeatMap
from shapely.geometry import Point, Polygon
import geopandas as gpd
from datetime import datetime, timedelta
import os
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NowcastingEvaluator:
    """
    Clase para evaluar y visualizar el rendimiento de predicciones de nowcasting.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa el evaluador.
        
        Args:
            output_dir (str): Directorio para guardar visualizaciones de evaluación
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def calculate_prediction_metrics(self, actual_cells_df, predicted_cells_df, 
                                    position_tolerance_km=10.0):
        """
        Calcula métricas de precisión para predicciones de nowcasting.
        
        Args:
            actual_cells_df: DataFrame con células reales observadas
            predicted_cells_df: DataFrame con células predichas
            position_tolerance_km: Tolerancia en km para considerar acierto posicional
            
        Returns:
            DataFrame con métricas por predicción y métricas agregadas
        """
        metrics = []
        
        # Asegurar que tenemos columnas de timestamp
        if 'timestamp' not in actual_cells_df.columns:
            if 'time' in actual_cells_df.columns:
                actual_cells_df['timestamp'] = actual_cells_df['time']
            else:
                raise ValueError("No se encontró columna de tiempo en actual_cells_df")
        
        # Para cada predicción
        for _, pred in predicted_cells_df.iterrows():
            # Buscar células reales para el tiempo predicho
            pred_time = pred['pred_time']
            track_id = pred['track_id']
            
            # Encontrar células reales que coincidan en tiempo y track
            matching_cells = actual_cells_df[
                (actual_cells_df['timestamp'] == pred_time) & 
                (actual_cells_df['track_id'] == track_id)
            ]
            
            if matching_cells.empty:
                # La célula predicha no existe en realidad (falso positivo)
                metrics.append({
                    'track_id': track_id,
                    'pred_time': pred_time,
                    'lead_time_min': (pred_time - pred['last_time']).total_seconds() / 60,
                    'position_error_km': float('nan'),  # No hay célula real para comparar
                    'area_error_pct': float('nan'),
                    'intensity_error_pct': float('nan'),
                    'hit': False,
                    'false_positive': True,
                    'false_negative': False,
                    'position_hit': False,
                    'area_hit': False,
                    'intensity_hit': False
                })
            else:
                # Tomar la primera célula coincidente (si hay múltiples)
                actual = matching_cells.iloc[0]
                
                # Calcular error de posición
                pos_error_km = self._calculate_distance(
                    pred['pred_lon'], pred['pred_lat'],
                    actual['centroid_lon'], actual['centroid_lat']
                )
                
                # Calcular errores porcentuales de área e intensidad
                area_error_pct = abs(pred['pred_area'] - actual['area_km2']) / actual['area_km2'] * 100 if actual['area_km2'] > 0 else float('nan')
                intensity_error_pct = abs(pred['pred_n_flashes'] - actual['n_flashes']) / actual['n_flashes'] * 100 if actual['n_flashes'] > 0 else float('nan')
                
                # Determinar si la predicción es un acierto según tolerancias
                position_hit = pos_error_km <= position_tolerance_km
                area_hit = area_error_pct <= 50 if not np.isnan(area_error_pct) else False
                intensity_hit = intensity_error_pct <= 50 if not np.isnan(intensity_error_pct) else False
                
                # Considerar acierto general si la posición es correcta
                hit = position_hit
                
                metrics.append({
                    'track_id': track_id,
                    'pred_time': pred_time,
                    'lead_time_min': (pred_time - pred['last_time']).total_seconds() / 60,
                    'position_error_km': pos_error_km,
                    'area_error_pct': area_error_pct,
                    'intensity_error_pct': intensity_error_pct,
                    'hit': hit,
                    'false_positive': False,
                    'false_negative': False,
                    'position_hit': position_hit,
                    'area_hit': area_hit,
                    'intensity_hit': intensity_hit
                })
        
        # Buscar falsos negativos: células reales que no fueron predichas
        all_pred_times = predicted_cells_df['pred_time'].unique()
        all_pred_tracks = predicted_cells_df['track_id'].unique()
        
        for pred_time in all_pred_times:
            # Obtener células reales en este tiempo
            cells_at_time = actual_cells_df[actual_cells_df['timestamp'] == pred_time]
            
            for _, cell in cells_at_time.iterrows():
                if cell['track_id'] in all_pred_tracks:
                    # Verificar si esta célula ya fue considerada
                    matching_predictions = predicted_cells_df[
                        (predicted_cells_df['pred_time'] == pred_time) & 
                        (predicted_cells_df['track_id'] == cell['track_id'])
                    ]
                    
                    if matching_predictions.empty:
                        # Falso negativo: célula real sin predicción correspondiente
                        metrics.append({
                            'track_id': cell['track_id'],
                            'pred_time': pred_time,
                            'lead_time_min': float('nan'),  # No hay predicción
                            'position_error_km': float('nan'),
                            'area_error_pct': float('nan'),
                            'intensity_error_pct': float('nan'),
                            'hit': False,
                            'false_positive': False,
                            'false_negative': True,
                            'position_hit': False,
                            'area_hit': False,
                            'intensity_hit': False
                        })
        
        # Crear DataFrame de métricas individuales
        metrics_df = pd.DataFrame(metrics)
        
        # Calcular métricas agregadas
        if not metrics_df.empty:
            total_predictions = len(metrics_df)
            total_hits = sum(metrics_df['hit'])
            total_false_positives = sum(metrics_df['false_positive'])
            total_false_negatives = sum(metrics_df['false_negative'])
            
            # Métricas típicas de verificación
            pod = total_hits / (total_hits + total_false_negatives) if (total_hits + total_false_negatives) > 0 else 0
            far = total_false_positives / (total_hits + total_false_positives) if (total_hits + total_false_positives) > 0 else 0
            csi = total_hits / (total_hits + total_false_positives + total_false_negatives) if (total_hits + total_false_positives + total_false_negatives) > 0 else 0
            
            # Métricas posicionales solo para aciertos
            position_errors = metrics_df[metrics_df['hit']]['position_error_km']
            mean_position_error = position_errors.mean() if len(position_errors) > 0 else float('nan')
            median_position_error = position_errors.median() if len(position_errors) > 0 else float('nan')
            
            # Agregar métricas agregadas
            aggregated_metrics = {
                'total_predictions': total_predictions,
                'total_hits': total_hits,
                'total_false_positives': total_false_positives,
                'total_false_negatives': total_false_negatives,
                'probability_of_detection': pod,
                'false_alarm_ratio': far,
                'critical_success_index': csi,
                'mean_position_error_km': mean_position_error,
                'median_position_error_km': median_position_error
            }
        else:
            aggregated_metrics = {}
        
        return metrics_df, aggregated_metrics
    
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
    def plot_error_distribution(self, metrics_df, save_path=None):
        """
        Crea gráficos de distribución de errores.
        
        Args:
            metrics_df: DataFrame con métricas de evaluación
            save_path: Ruta para guardar la figura generada
        """
        if metrics_df.empty:
            print("No hay datos para graficar distribución de errores")
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histograma de errores de posición
        valid_pos_errors = metrics_df['position_error_km'].dropna()
        if not valid_pos_errors.empty:
            axs[0, 0].hist(valid_pos_errors, bins=15, color='blue', alpha=0.7)
            axs[0, 0].set_title('Distribución de Errores de Posición')
            axs[0, 0].set_xlabel('Error de Posición (km)')
            axs[0, 0].set_ylabel('Frecuencia')
            axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. Histograma de errores de área
        valid_area_errors = metrics_df['area_error_pct'].dropna()
        if not valid_area_errors.empty:
            axs[0, 1].hist(valid_area_errors, bins=15, color='green', alpha=0.7)
            axs[0, 1].set_title('Distribución de Errores de Área')
            axs[0, 1].set_xlabel('Error de Área (%)')
            axs[0, 1].set_ylabel('Frecuencia')
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Histograma de errores de intensidad
        valid_intensity_errors = metrics_df['intensity_error_pct'].dropna()
        if not valid_intensity_errors.empty:
            axs[1, 0].hist(valid_intensity_errors, bins=15, color='red', alpha=0.7)
            axs[1, 0].set_title('Distribución de Errores de Intensidad')
            axs[1, 0].set_xlabel('Error de Intensidad (%)')
            axs[1, 0].set_ylabel('Frecuencia')
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Errores vs tiempo de anticipación
        valid_lead_times = metrics_df[~metrics_df['lead_time_min'].isna() & ~metrics_df['position_error_km'].isna()]
        if not valid_lead_times.empty:
            axs[1, 1].scatter(valid_lead_times['lead_time_min'], valid_lead_times['position_error_km'], 
                            c=valid_lead_times['track_id'], cmap='tab10', alpha=0.7)
            axs[1, 1].set_title('Error de Posición vs Tiempo de Anticipación')
            axs[1, 1].set_xlabel('Tiempo de Anticipación (min)')
            axs[1, 1].set_ylabel('Error de Posición (km)')
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en {save_path}")
        
        plt.show()
    
    def create_error_map(self, metrics_df, actual_cells_gdf, predicted_cells_df):
        """
        Crea un mapa interactivo mostrando errores de predicción.
        
        Args:
            metrics_df: DataFrame con métricas de error
            actual_cells_gdf: GeoDataFrame de células reales
            predicted_cells_df: DataFrame de predicciones
            
        Returns:
            folium.Map: Mapa interactivo con visualización de errores
        """
        # Determinar centro del mapa
        if not actual_cells_gdf.empty:
            center_lat = actual_cells_gdf.centroid.y.mean()
            center_lon = actual_cells_gdf.centroid.x.mean()
        elif not predicted_cells_df.empty:
            center_lat = predicted_cells_df['pred_lat'].mean()
            center_lon = predicted_cells_df['pred_lon'].mean()
        else:
            center_lat = -34.0
            center_lon = -64.0
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Crear grupo para células reales
        real_group = folium.FeatureGroup(name='Células Reales')
        
        # Crear grupo para predicciones
        pred_group = folium.FeatureGroup(name='Predicciones')
        
        # Crear grupo para líneas de error
        error_group = folium.FeatureGroup(name='Errores de Predicción')
        
        # Lista de colores para track_ids
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080"
        ]
        
        # Procesar cada par de predicción-realidad
        for _, row in metrics_df.iterrows():
            if row['false_positive'] or row['false_negative']:
                continue  # Saltar si no hay par real-predicho
                
            track_id = row['track_id']
            pred_time = row['pred_time']
            error_km = row['position_error_km']
            
            # Obtener coordenadas predichas
            pred_matches = predicted_cells_df[
                (predicted_cells_df['track_id'] == track_id) & 
                (predicted_cells_df['pred_time'] == pred_time)
            ]
            
            if pred_matches.empty:
                continue
                
            pred_record = pred_matches.iloc[0]
            pred_lat = pred_record['pred_lat']
            pred_lon = pred_record['pred_lon']
            
            # Obtener coordenadas reales
            real_matches = actual_cells_gdf[
                (actual_cells_gdf['track_id'] == track_id) & 
                (actual_cells_gdf['timestamp'] == pred_time)
            ]
            
            if real_matches.empty:
                continue
                
            real_record = real_matches.iloc[0]
            real_lat = real_record['centroid_lat']
            real_lon = real_record['centroid_lon']
            
            # Color según track_id
            color = colors[track_id % len(colors)]
            
            # Determinar color de error según magnitud
            if error_km < 5:
                error_color = 'green'
            elif error_km < 15:
                error_color = 'orange'
            else:
                error_color = 'red'
            
            # Añadir marcador para posición real
            folium.CircleMarker(
                location=[real_lat, real_lon],
                radius=6,
                color='black',
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Celda Real #{real_record['cell_id']}<br>Track #{track_id}"
            ).add_to(real_group)
            
            # Añadir marcador para posición predicha
            folium.CircleMarker(
                location=[pred_lat, pred_lon],
                radius=6,
                color='white',
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Predicción para Track #{track_id}<br>Error: {error_km:.2f} km"
            ).add_to(pred_group)
            
            # Añadir línea de error
            folium.PolyLine(
                locations=[
                    [pred_lat, pred_lon],
                    [real_lat, real_lon]
                ],
                color=error_color,
                weight=3,
                opacity=0.8,
                popup=f"Error: {error_km:.2f} km<br>Track #{track_id}"
            ).add_to(error_group)
        
        # Añadir grupos al mapa
        real_group.add_to(m)
        pred_group.add_to(m)
        error_group.add_to(m)
        
        # Añadir control de capas
        folium.LayerControl().add_to(m)
        
        # Añadir leyenda para colores de error
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <h4>Leyenda de Errores</h4>
            <div><span style="background-color: green; display: inline-block; width: 15px; height: 15px;"></span> < 5 km</div>
            <div><span style="background-color: orange; display: inline-block; width: 15px; height: 15px;"></span> 5-15 km</div>
            <div><span style="background-color: red; display: inline-block; width: 15px; height: 15px;"></span> > 15 km</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    def plot_error_distribution(self, metrics_df, save_path=None):
        """
        Crea gráficos de distribución de errores.
        
        Args:
            metrics_df: DataFrame con métricas de evaluación
            save_path: Ruta para guardar la figura generada
        """
        if metrics_df.empty:
            print("No hay datos para graficar distribución de errores")
            return
        
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Histograma de errores de posición
        valid_pos_errors = metrics_df['position_error_km'].dropna()
        if not valid_pos_errors.empty:
            axs[0, 0].hist(valid_pos_errors, bins=15, color='blue', alpha=0.7)
            axs[0, 0].set_title('Distribución de Errores de Posición')
            axs[0, 0].set_xlabel('Error de Posición (km)')
            axs[0, 0].set_ylabel('Frecuencia')
            axs[0, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 2. Histograma de errores de área
        valid_area_errors = metrics_df['area_error_pct'].dropna()
        if not valid_area_errors.empty:
            axs[0, 1].hist(valid_area_errors, bins=15, color='green', alpha=0.7)
            axs[0, 1].set_title('Distribución de Errores de Área')
            axs[0, 1].set_xlabel('Error de Área (%)')
            axs[0, 1].set_ylabel('Frecuencia')
            axs[0, 1].grid(True, linestyle='--', alpha=0.7)
        
        # 3. Histograma de errores de intensidad
        valid_intensity_errors = metrics_df['intensity_error_pct'].dropna()
        if not valid_intensity_errors.empty:
            axs[1, 0].hist(valid_intensity_errors, bins=15, color='red', alpha=0.7)
            axs[1, 0].set_title('Distribución de Errores de Intensidad')
            axs[1, 0].set_xlabel('Error de Intensidad (%)')
            axs[1, 0].set_ylabel('Frecuencia')
            axs[1, 0].grid(True, linestyle='--', alpha=0.7)
        
        # 4. Errores vs tiempo de anticipación
        valid_lead_times = metrics_df[~metrics_df['lead_time_min'].isna() & ~metrics_df['position_error_km'].isna()]
        if not valid_lead_times.empty:
            axs[1, 1].scatter(valid_lead_times['lead_time_min'], valid_lead_times['position_error_km'], 
                            c=valid_lead_times['track_id'], cmap='tab10', alpha=0.7)
            axs[1, 1].set_title('Error de Posición vs Tiempo de Anticipación')
            axs[1, 1].set_xlabel('Tiempo de Anticipación (min)')
            axs[1, 1].set_ylabel('Error de Posición (km)')
            axs[1, 1].grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en {save_path}")
        
        plt.show()
    
    def create_error_map(self, metrics_df, actual_cells_gdf, predicted_cells_df):
        """
        Crea un mapa interactivo mostrando errores de predicción.
        
        Args:
            metrics_df: DataFrame con métricas de error
            actual_cells_gdf: GeoDataFrame de células reales
            predicted_cells_df: DataFrame de predicciones
            
        Returns:
            folium.Map: Mapa interactivo con visualización de errores
        """
        # Determinar centro del mapa
        if not actual_cells_gdf.empty:
            center_lat = actual_cells_gdf.centroid.y.mean()
            center_lon = actual_cells_gdf.centroid.x.mean()
        elif not predicted_cells_df.empty:
            center_lat = predicted_cells_df['pred_lat'].mean()
            center_lon = predicted_cells_df['pred_lon'].mean()
        else:
            center_lat = -34.0
            center_lon = -64.0
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Crear grupo para células reales
        real_group = folium.FeatureGroup(name='Células Reales')
        
        # Crear grupo para predicciones
        pred_group = folium.FeatureGroup(name='Predicciones')
        
        # Crear grupo para líneas de error
        error_group = folium.FeatureGroup(name='Errores de Predicción')
        
        # Lista de colores para track_ids
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080"
        ]
        
        # Procesar cada par de predicción-realidad
        for _, row in metrics_df.iterrows():
            if row['false_positive'] or row['false_negative']:
                continue  # Saltar si no hay par real-predicho
                
            track_id = row['track_id']
            pred_time = row['pred_time']
            error_km = row['position_error_km']
            
            # Obtener coordenadas predichas
            pred_matches = predicted_cells_df[
                (predicted_cells_df['track_id'] == track_id) & 
                (predicted_cells_df['pred_time'] == pred_time)
            ]
            
            if pred_matches.empty:
                continue
                
            pred_record = pred_matches.iloc[0]
            pred_lat = pred_record['pred_lat']
            pred_lon = pred_record['pred_lon']
            
            # Obtener coordenadas reales
            real_matches = actual_cells_gdf[
                (actual_cells_gdf['track_id'] == track_id) & 
                (actual_cells_gdf['timestamp'] == pred_time)
            ]
            
            if real_matches.empty:
                continue
                
            real_record = real_matches.iloc[0]
            real_lat = real_record['centroid_lat']
            real_lon = real_record['centroid_lon']
            
            # Color según track_id
            color = colors[track_id % len(colors)]
            
            # Determinar color de error según magnitud
            if error_km < 5:
                error_color = 'green'
            elif error_km < 15:
                error_color = 'orange'
            else:
                error_color = 'red'
            
            # Añadir marcador para posición real
            folium.CircleMarker(
                location=[real_lat, real_lon],
                radius=6,
                color='black',
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Celda Real #{real_record['cell_id']}<br>Track #{track_id}"
            ).add_to(real_group)
            
            # Añadir marcador para posición predicha
            folium.CircleMarker(
                location=[pred_lat, pred_lon],
                radius=6,
                color='white',
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=f"Predicción para Track #{track_id}<br>Error: {error_km:.2f} km"
            ).add_to(pred_group)
            
            # Añadir línea de error
            folium.PolyLine(
                locations=[
                    [pred_lat, pred_lon],
                    [real_lat, real_lon]
                ],
                color=error_color,
                weight=3,
                opacity=0.8,
                popup=f"Error: {error_km:.2f} km<br>Track #{track_id}"
            ).add_to(error_group)
        
        # Añadir grupos al mapa
        real_group.add_to(m)
        pred_group.add_to(m)
        error_group.add_to(m)
        
        # Añadir control de capas
        folium.LayerControl().add_to(m)
        
        # Añadir leyenda para colores de error
        legend_html = """
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <h4>Leyenda de Errores</h4>
            <div><span style="background-color: green; display: inline-block; width: 15px; height: 15px;"></span> < 5 km</div>
            <div><span style="background-color: orange; display: inline-block; width: 15px; height: 15px;"></span> 5-15 km</div>
            <div><span style="background-color: red; display: inline-block; width: 15px; height: 15px;"></span> > 15 km</div>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    def plot_contingency_table(self, contingency_data, title=None, save_path=None):
        """
        Visualiza una tabla de contingencia y sus métricas derivadas.
        
        Args:
            contingency_data: Diccionario con datos de tabla de contingencia y métricas
            title: Título opcional para el gráfico
            save_path: Ruta para guardar la figura generada
        """
        if not contingency_data:
            print("No hay datos para visualizar tabla de contingencia")
            return
        
        # Extraer datos
        table = contingency_data['contingency_table']
        metrics = contingency_data['metrics']
        
        # Crear figura
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Visualizar tabla de contingencia
        data = [
            [table['hits'], table['false_alarms']],
            [table['misses'], table['correct_negatives']]
        ]
        
        ax1.imshow(data, cmap='Blues')
        
        # Añadir etiquetas
        for i in range(2):
            for j in range(2):
                text = ax1.text(j, i, data[i][j], ha="center", va="center", color="black")
        
        # Configurar eje
        ax1.set_title('Tabla de Contingencia')
        ax1.set_xticks([0, 1])
        ax1.set_yticks([0, 1])
        ax1.set_xticklabels(['Observado Sí', 'Observado No'])
        ax1.set_yticklabels(['Predicho Sí', 'Predicho No'])
        
        # 2. Visualizar métricas
        metrics_names = list(metrics.keys())
        metrics_values = list(metrics.values())
        
        ax2.bar(metrics_names, metrics_values, color=['blue', 'red', 'green', 'orange'])
        
        # Añadir valores encima de las barras
        for i, v in enumerate(metrics_values):
            ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
        ax2.set_title('Métricas de Verificación')
        ax2.set_ylim([0, 1.1])
        ax2.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Título general
        if title:
            fig.suptitle(title, fontsize=14)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado en {save_path}")
        
        plt.show()
    
    def analyze_error_patterns(self, metrics_df, predicted_df, actual_df=None):
        """
        Analiza patrones de error en las predicciones.
        
        Args:
            metrics_df: DataFrame con métricas de error de predicciones
            predicted_df: DataFrame con datos de predicciones
            actual_df: DataFrame opcional con datos reales
            
        Returns:
            dict: Análisis de patrones de error
        """
        if metrics_df.empty:
            return {}
        
        # Inicializar resultados
        analysis = {
            'error_by_leadtime': {},
            'error_by_intensity': {},
            'error_by_movement_speed': {},
            'error_by_track_id': {},
            'common_error_patterns': []
        }
        
        # 1. Analizar error por tiempo de anticipación
        leadtime_groups = metrics_df.groupby(metrics_df['lead_time_min'].apply(lambda x: round(x / 10) * 10))
        
        for leadtime, group in leadtime_groups:
            position_errors = group['position_error_km'].dropna()
            if not position_errors.empty:
                analysis['error_by_leadtime'][leadtime] = {
                    'mean_error': position_errors.mean(),
                    'median_error': position_errors.median(),
                    'std_error': position_errors.std(),
                    'count': len(position_errors)
                }
        
        # 2. Analizar error por intensidad
        if 'last_n_flashes' in predicted_df.columns:
            # Definir categorías de intensidad
            predicted_df['intensity_category'] = pd.cut(
                predicted_df['last_n_flashes'],
                bins=[0, 10, 50, 100, 500, float('inf')],
                labels=['muy baja', 'baja', 'media', 'alta', 'muy alta']
            )
            
            # Fusionar con métricas
            merged = pd.merge(
                metrics_df,
                predicted_df[['track_id', 'pred_time', 'intensity_category']],
                on=['track_id', 'pred_time'],
                how='left'
            )
            
            # Calcular errores por categoría
            intensity_groups = merged.groupby('intensity_category')
            
            for category, group in intensity_groups:
                position_errors = group['position_error_km'].dropna()
                if not position_errors.empty:
                    analysis['error_by_intensity'][category] = {
                        'mean_error': position_errors.mean(),
                        'median_error': position_errors.median(),
                        'std_error': position_errors.std(),
                        'count': len(position_errors)
                    }
        
        # 3. Analizar error por velocidad de movimiento
        if 'velocity_lon' in predicted_df.columns and 'velocity_lat' in predicted_df.columns:
            # Calcular velocidad total
            predicted_df['movement_speed'] = np.sqrt(
                predicted_df['velocity_lon']**2 + predicted_df['velocity_lat']**2
            )
            
            # Definir categorías de velocidad
            predicted_df['speed_category'] = pd.cut(
                predicted_df['movement_speed'],
                bins=[0, 10, 20, 30, float('inf')],
                labels=['lenta', 'moderada', 'rápida', 'muy rápida']
            )
            
            # Fusionar con métricas
            merged = pd.merge(
                metrics_df,
                predicted_df[['track_id', 'pred_time', 'speed_category']],
                on=['track_id', 'pred_time'],
                how='left'
            )
            
            # Calcular errores por categoría
            speed_groups = merged.groupby('speed_category')
            
            for category, group in speed_groups:
                position_errors = group['position_error_km'].dropna()
                if not position_errors.empty:
                    analysis['error_by_movement_speed'][category] = {
                        'mean_error': position_errors.mean(),
                        'median_error': position_errors.median(),
                        'std_error': position_errors.std(),
                        'count': len(position_errors)
                    }
        
        # 4. Analizar error por track_id
        track_groups = metrics_df.groupby('track_id')
        
        for track_id, group in track_groups:
            position_errors = group['position_error_km'].dropna()
            if not position_errors.empty:
                analysis['error_by_track_id'][track_id] = {
                    'mean_error': position_errors.mean(),
                    'median_error': position_errors.median(),
                    'std_error': position_errors.std(),
                    'min_error': position_errors.min(),
                    'max_error': position_errors.max(),
                    'count': len(position_errors)
                }
        
        # 5. Identificar patrones comunes de error
        # Buscar situaciones donde el error es consistentemente alto
        high_error_threshold = 20  # km
        
        # Por tiempo de anticipación
        for leadtime, stats in analysis['error_by_leadtime'].items():
            if stats['mean_error'] > high_error_threshold:
                analysis['common_error_patterns'].append(
                    f"Error alto para tiempo de anticipación de {leadtime} min: {stats['mean_error']:.1f} km"
                )
        
        # Por intensidad
        for category, stats in analysis['error_by_intensity'].items():
            if stats['mean_error'] > high_error_threshold:
                analysis['common_error_patterns'].append(
                    f"Error alto para celdas de intensidad {category}: {stats['mean_error']:.1f} km"
                )
        
        # Por velocidad
        for category, stats in analysis['error_by_movement_speed'].items():
            if stats['mean_error'] > high_error_threshold:
                analysis['common_error_patterns'].append(
                    f"Error alto para celdas con movimiento {category}: {stats['mean_error']:.1f} km"
                )
        
        # Por track específico
        for track_id, stats in analysis['error_by_track_id'].items():
            if stats['mean_error'] > high_error_threshold:
                analysis['common_error_patterns'].append(
                    f"Error alto para track #{track_id}: {stats['mean_error']:.1f} km"
                )
        
        return analysis
    
if __name__ == "__main__":
    import glob
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Evaluar sistema de nowcasting de tormentas eléctricas.')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--predictions-dir', type=str, default='predictions', 
                        help='Directorio con archivos CSV de predicciones')
    parser.add_argument('--output-dir', type=str, default='evaluation', 
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Buscar archivos de datos
    geojson_files = sorted(glob.glob(f"{args.geojson_dir}/cells_*.geojson"))
    prediction_files = sorted(glob.glob(f"{args.predictions_dir}/predictions_*.csv"))
    
    if not geojson_files:
        print(f"No se encontraron archivos GeoJSON en {args.geojson_dir}")
    else:
        print(f"Se encontraron {len(geojson_files)} archivos GeoJSON")
    
    if not prediction_files:
        print(f"No se encontraron archivos de predicción en {args.predictions_dir}")
    else:
        print(f"Se encontraron {len(prediction_files)} archivos de predicción")
    
    # Evaluar rendimiento
    if geojson_files and prediction_files:
        print("\nIniciando evaluación de nowcasting...")
        results = evaluate_nowcasting_performance(
            geojson_files,
            prediction_files,
            output_dir=args.output_dir
        )
        
        # Imprimir resumen de resultados
        if results and 'aggregated_metrics' in results:
            print("\n=== Resumen de Evaluación de Nowcasting ===")
            print(f"Total de predicciones: {results['aggregated_metrics']['total_predictions']}")
            print(f"Aciertos: {results['aggregated_metrics']['total_hits']}")
            print(f"Falsas alarmas: {results['aggregated_metrics']['total_false_positives']}")
            print(f"Falsos negativos: {results['aggregated_metrics']['total_false_negatives']}")
            print(f"POD: {results['aggregated_metrics']['probability_of_detection']:.3f}")
            print(f"FAR: {results['aggregated_metrics']['false_alarm_ratio']:.3f}")
            print(f"CSI: {results['aggregated_metrics']['critical_success_index']:.3f}")
            print(f"Error medio de posición: {results['aggregated_metrics']['mean_position_error_km']:.2f} km")
            print(f"Error mediano de posición: {results['aggregated_metrics']['median_position_error_km']:.2f} km")
            print("\nResultados guardados en el directorio: {args.output_dir}")
            
            # Mostrar archivos generados
            print("\nArchivos generados:")
            for key, file_path in results.items():
                if isinstance(file_path, str) and os.path.exists(file_path):
                    print(f"- {key}: {file_path}")
        else:
            print("No se pudieron generar resultados de evaluación")