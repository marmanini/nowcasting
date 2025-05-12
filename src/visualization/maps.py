# src/visualization/maps.py (versión completa corregida)

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import logging
import os
from datetime import datetime

# Configuración del logger
logger = logging.getLogger(__name__)

class LightningVisualizer:
    """
    Clase para visualizar datos de rayos, celdas y predicciones.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa el visualizador.
        
        Args:
            output_dir (str): Directorio para guardar las visualizaciones
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    
    def create_interactive_map(self, flash_df=None, cells_gdf=None, predictions_gdf=None, 
                              start_time=None, end_time=None):
        """
        Crea un mapa interactivo con datos de rayos, celdas y predicciones.
        
        Args:
            flash_df (pandas.DataFrame): DataFrame con datos de flashes
            cells_gdf (geopandas.GeoDataFrame): GeoDataFrame con celdas identificadas
            predictions_gdf (geopandas.GeoDataFrame): GeoDataFrame con predicciones
            start_time (datetime): Tiempo de inicio para el título
            end_time (datetime): Tiempo de fin para el título
            
        Returns:
            folium.Map: Mapa interactivo
        """
        # Determinar el centro del mapa
        if flash_df is not None and not flash_df.empty:
            center_lat = flash_df['flash_lat'].mean()
            center_lon = flash_df['flash_lon'].mean()
        elif cells_gdf is not None and not cells_gdf.empty:
            center_lat = cells_gdf.centroid.y.mean()
            center_lon = cells_gdf.centroid.x.mean()
        elif predictions_gdf is not None and not predictions_gdf.empty:
            center_lat = predictions_gdf.geometry.y.mean()
            center_lon = predictions_gdf.geometry.x.mean()
        else:
            # Valores por defecto (aproximadamente el centro de Argentina)
            center_lat = -34.0
            center_lon = -64.0
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Agregar título
        title_html = ''
        if start_time and end_time:
            title_html = f'''
                <h3 align="center" style="font-size:16px">
                    <b>GLM Lightning Nowcasting</b><br>
                    {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M UTC')}
                </h3>
            '''
            folium.Element(title_html).add_to(m)
        
        # Agregar datos de flashes individuales
        if flash_df is not None and not flash_df.empty:
            flash_group = folium.FeatureGroup(name='Lightning Flashes')
            
            # Limitar a máximo 2000 flashes para rendimiento
            plot_df = flash_df
            if len(flash_df) > 2000:
                plot_df = flash_df.sample(2000)
                
            for _, flash in plot_df.iterrows():
                # Color según cluster
                if 'cluster' in flash and flash['cluster'] != -1:
                    color = f"#{hash(flash['cluster']) % 0xFFFFFF:06x}"
                else:
                    color = 'gray'
                
                folium.CircleMarker(
                    location=[flash['flash_lat'], flash['flash_lon']],
                    radius=2,
                    color=color,
                    fill=True,
                    fill_opacity=0.5,
                    popup=f"Flash ID: {flash['flash_id']}<br>Energy: {flash['flash_energy']:.2f}"
                ).add_to(flash_group)
            
            flash_group.add_to(m)
        
        # Agregar celdas identificadas
        if cells_gdf is not None and not cells_gdf.empty:
            cell_group = folium.FeatureGroup(name='Lightning Cells')
            
            for _, cell in cells_gdf.iterrows():
                # Color único para cada celda
                color = f"#{hash(cell['cell_id']) % 0xFFFFFF:06x}"
                
                # Convertir geometría a lista de coordenadas
                if isinstance(cell.geometry, Polygon):
                    coords = [(y, x) for x, y in list(cell.geometry.exterior.coords)]
                    
                    # Crear polígono
                    folium.Polygon(
                        locations=coords,
                        color=color,
                        weight=2,
                        fill=True,
                        fill_opacity=0.2,
                        popup=f"Cell ID: {cell['cell_id']}<br>Track ID: {cell.get('track_id', 'N/A')}<br>Flashes: {cell['n_flashes']}<br>Area: {cell['area_km2']:.2f} km²"
                    ).add_to(cell_group)
                    
                    # Agregar marcador para el centroide
                    folium.CircleMarker(
                        location=[cell['centroid_lat'], cell['centroid_lon']],
                        radius=5,
                        color=color,
                        fill=True,
                        fill_opacity=0.8,
                        popup=f"Cell ID: {cell['cell_id']}<br>Track ID: {cell.get('track_id', 'N/A')}"
                    ).add_to(cell_group)
            
            cell_group.add_to(m)
        
        # Agregar predicciones
        if predictions_gdf is not None and not predictions_gdf.empty:
            pred_group = folium.FeatureGroup(name='Predictions')
            
            for _, pred in predictions_gdf.iterrows():
                # Color único para cada track
                color = f"#{hash(pred['track_id']) % 0xFFFFFF:06x}"
                
                # Crear línea desde la última posición a la predicha
                folium.PolyLine(
                    locations=[
                        [pred['last_lat'], pred['last_lon']],
                        [pred['pred_lat'], pred['pred_lon']]
                    ],
                    color=color,
                    weight=3,
                    opacity=0.7,
                    dash_array='5, 10'
                ).add_to(pred_group)
                
                # Crear marcador para la posición predicha
                folium.CircleMarker(
                    location=[pred['pred_lat'], pred['pred_lon']],
                    radius=7,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=f"Track ID: {pred['track_id']}<br>Pred. Time: {pred['pred_time']}<br>Velocity: {np.sqrt(pred['velocity_lon']**2 + pred['velocity_lat']**2):.2f} °/h"
                ).add_to(pred_group)
            
            pred_group.add_to(m)
        
        # Agregar controles de capas
        folium.LayerControl().add_to(m)
        
        return m
    
    def save_interactive_map(self, folium_map, filename=None):
        """
        Guarda un mapa interactivo en un archivo HTML.
        
        Args:
            folium_map (folium.Map): Mapa a guardar
            filename (str): Nombre del archivo (opcional)
            
        Returns:
            str: Ruta al archivo guardado
        """
        if not self.output_dir:
            logger.warning("Output directory not specified, cannot save map")
            return None
        
        if not filename:
            # Generar nombre de archivo basado en timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"lightning_nowcast_{timestamp}.html"
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            folium_map.save(file_path)
            logger.info(f"Map saved to {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error saving map: {e}")
            return None
    
    def create_density_map(self, flash_df, resolution=0.05):
        """
        Crea un mapa de densidad de rayos.
        
        Args:
            flash_df (pandas.DataFrame): DataFrame con datos de flashes
            resolution (float): Resolución del grid para densidad
            
        Returns:
            folium.Map: Mapa de densidad
        """
        if flash_df.empty:
            logger.warning("Empty flash DataFrame, cannot create density map")
            return folium.Map(location=[-34.0, -64.0], zoom_start=5)
        
        # Calcular centro
        center_lat = flash_df['flash_lat'].mean()
        center_lon = flash_df['flash_lon'].mean()
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Método simple: crear marcadores para cada flash
        # Esto siempre funcionará, incluso sin dependencias adicionales
        for _, flash in flash_df.iterrows():
            folium.CircleMarker(
                location=[flash['flash_lat'], flash['flash_lon']],
                radius=2,
                color='red',
                fill=True,
                fill_opacity=0.5
            ).add_to(m)
        
        # Agregar información
        title_html = f'''
            <h3 align="center" style="font-size:16px">
                <b>GLM Lightning Density Map</b><br>
                {len(flash_df)} flashes
            </h3>
        '''
        folium.Element(title_html).add_to(m)
        
        return m

    def _get_heat_color(self, intensity):
        """
        Devuelve un color basado en intensidad (0-1).
        """
        if intensity < 0.4:
            return 'blue'
        elif intensity < 0.65:
            return 'lime'
        elif intensity < 0.8:
            return 'yellow'
        else:
            return 'red'