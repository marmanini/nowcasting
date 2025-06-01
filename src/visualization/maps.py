# src/visualization/maps.py (versión mejorada con trackeo y predicciones)

import folium
from folium.plugins import TimestampedGeoJson, MeasureControl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import logging
import os
from datetime import datetime, timedelta
import json
import random
import branca.colormap as cm

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
                            start_time=None, end_time=None, show_uncertainty=False,
                            uncertainty_data=None):
        """
        Crea un mapa interactivo con datos de rayos, celdas y predicciones.
        
        Args:
            flash_df (pandas.DataFrame): DataFrame con datos de flashes
            cells_gdf (geopandas.GeoDataFrame): GeoDataFrame con celdas identificadas
            predictions_gdf (geopandas.GeoDataFrame): GeoDataFrame con predicciones
            start_time (datetime): Tiempo de inicio para el título
            end_time (datetime): Tiempo de fin para el título
            show_uncertainty (bool): Si se deben mostrar zonas de incertidumbre
            uncertainty_data (dict): Datos adicionales de incertidumbre
            
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
        
        # Agregar control de medición
        m.add_child(MeasureControl())
        
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
        
        # Crear colormap para intensidad de celdas
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=0,
            vmax=500  # Ajustar según el máximo n_flashes en tus datos
        )
        colormap.caption = 'Intensidad (número de rayos)'
        m.add_child(colormap)
        
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
                    popup=f"Flash ID: {flash.get('flash_id', 'N/A')}<br>Energy: {flash.get('flash_energy', 0):.2e}"
                ).add_to(flash_group)
            
            flash_group.add_to(m)
        
        # Agregar celdas identificadas
        if cells_gdf is not None and not cells_gdf.empty:
            # Crear una lista de colores fijos brillantes
            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
                "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
                "#80FF00", "#00FF80", "#FF8080", "#80FF80", "#8080FF", 
                "#FFFF80", "#FF80FF", "#80FFFF", "#FF4000", "#40FF00"
            ]
            
            # Solo para validar que estamos procesando los datos correctamente
            print(f"Número total de celdas: {len(cells_gdf)}")
            
            # Crear grupo para las celdas
            cells_group = folium.FeatureGroup(name='Storm Cells')
            
            # Verificar si existe la columna track_id
            has_cell_track_id = 'track_id' in cells_gdf.columns
            
            # En lugar de crear grupos complejos, simplemente iteramos y creamos cada polígono directamente
            for i, (index, cell) in enumerate(cells_gdf.iterrows()):
                # Asignar color basado en la intensidad o usar un color fijo si hay track_id
                if has_cell_track_id:
                    # Usar color consistente basado en track_id para seguimiento visual
                    track_id = cell.get('track_id', i)
                    color = colors[track_id % len(colors)]
                else:
                    # Color basado en intensidad
                    color = colormap(cell['n_flashes'])
                
                # Verificar tipo de geometría e imprimir información para depuración
                print(f"Procesando celda {i}: ID={cell['cell_id']}, Tipo geometría={type(cell.geometry)}")
                
                # Crear popup con información detallada
                popup_html = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>Celda #{cell['cell_id']}</h4>
                    <b>Rayos:</b> {cell['n_flashes']}<br>
                    <b>Área:</b> {cell['area_km2']:.1f} km²<br>
                    <b>Energía:</b> {cell['total_energy']:.2e}<br>
                    <b>Inicio:</b> {cell.get('start_time', 'N/A')}<br>
                    <b>Fin:</b> {cell.get('end_time', 'N/A')}<br>
                """
                
                # Agregar información de track si está disponible
                if has_cell_track_id:
                    popup_html += f"<b>Track ID:</b> {cell['track_id']}<br>"
                if 'age_minutes' in cell:
                    popup_html += f"<b>Edad:</b> {cell['age_minutes']:.1f} min<br>"
                
                popup_html += "</div>"
                
                # Convertir geometría a lista de coordenadas
                if isinstance(cell.geometry, Polygon):
                    # Obtener las coordenadas y crear el polígono
                    coords = [(y, x) for x, y in list(cell.geometry.exterior.coords)]
                    
                    # Crear polígono en el grupo
                    folium.Polygon(
                        locations=coords,
                        color=color,
                        weight=3,
                        opacity=1.0,
                        fill=True,
                        fill_opacity=0.7,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(cells_group)
                    
                    # Agregar centroide
                    folium.CircleMarker(
                        location=[cell['centroid_lat'], cell['centroid_lon']],
                        radius=6,
                        color='white',
                        fill=True,
                        fill_opacity=1.0,
                        stroke=True,
                        weight=1,
                        popup=folium.Popup(popup_html, max_width=300)
                    ).add_to(cells_group)
                    
                    # Agregar etiqueta con ID
                    folium.Marker(
                        location=[cell['centroid_lat'], cell['centroid_lon']],
                        icon=folium.DivIcon(
                            icon_size=(20, 20),
                            icon_anchor=(10, 10),
                            html=f'<div style="font-size: 10pt; color: black; background-color: white; border-radius: 50%; padding: 3px; border: 1px solid black; text-align: center;">{cell["cell_id"]}</div>'
                        )
                    ).add_to(cells_group)
                else:
                    print(f"ADVERTENCIA: Celda {cell['cell_id']} no tiene geometría de tipo Polygon")
            
            cells_group.add_to(m)
        
        # Añadir control de escala

        # Agregar predicciones
        if predictions_gdf is not None and not predictions_gdf.empty:
            pred_group = folium.FeatureGroup(name='Predictions')
            
            # Lista de colores fijos para mantener consistencia con las celdas
            colors = [
                "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
                "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
                "#80FF00", "#00FF80", "#FF8080", "#80FF80", "#8080FF", 
                "#FFFF80", "#FF80FF", "#80FFFF", "#FF4000", "#40FF00"
            ]
            
            # Verificar si la columna track_id está presente
            has_track_id = 'track_id' in predictions_gdf.columns
            
            for i, (_, pred) in enumerate(predictions_gdf.iterrows()):
                # Usar el mismo esquema de colores que las celdas para consistencia visual
                if has_track_id:
                    track_id = pred['track_id']
                    track_label = f"Track #{track_id}"
                else:
                    track_id = i  # Usar el índice como identificador alternativo
                    track_label = f"Predicción #{i+1}"
                    
                color = colors[track_id % len(colors)]
                
                # Crear popup con información detallada de la predicción
                pred_popup_html = f"""
                <div style="font-family: Arial; width: 220px;">
                    <h4>Predicción - {track_label}</h4>
                    <b>Celda Original:</b> #{pred.get('last_cell_id', 'N/A')}<br>
                    <b>Tiempo Actual:</b> {pred.get('last_time', 'N/A')}<br>
                    <b>Tiempo Predicción:</b> {pred.get('pred_time', 'N/A')}<br>
                    <hr style="margin: 5px 0;">
                    <b>Pos. Actual:</b> [{pred.get('last_lat', 0):.4f}, {pred.get('last_lon', 0):.4f}]<br>
                    <b>Pos. Predicha:</b> [{pred['pred_lat']:.4f}, {pred['pred_lon']:.4f}]<br>
                    <hr style="margin: 5px 0;">
                    <b>Rayos (actual):</b> {pred.get('last_n_flashes', 'N/A')}<br>
                    <b>Rayos (predicción):</b> {pred.get('pred_n_flashes', 'N/A')}<br>
                    <b>Área (actual):</b> {pred.get('last_area', 0):.1f} km²<br>
                    <b>Área (predicción):</b> {pred.get('pred_area', 0):.1f} km²<br>
                    <hr style="margin: 5px 0;">
                """
                
                # Añadir información de velocidad si está disponible
                if 'velocity_lon' in pred and 'velocity_lat' in pred:
                    velocity_magnitude = np.sqrt(pred['velocity_lon']**2 + pred['velocity_lat']**2)
                    direction = self._get_direction(pred['velocity_lon'], pred['velocity_lat'])
                    pred_popup_html += f"""
                    <b>Velocidad:</b> {velocity_magnitude:.2f} °/h<br>
                    <b>Dirección:</b> {direction}
                    """
                
                pred_popup_html += "</div>"
                
                # Crear línea desde la última posición a la predicha
                if 'last_lat' in pred and 'last_lon' in pred:
                    folium.PolyLine(
                        locations=[
                            [pred['last_lat'], pred['last_lon']],
                            [pred['pred_lat'], pred['pred_lon']]
                        ],
                        color=color,
                        weight=3,
                        opacity=0.7,
                        dash_array='5, 10',
                        popup=folium.Popup(pred_popup_html, max_width=300)
                    ).add_to(pred_group)
                    
                    # Añadir símbolo de flecha direccional
                    self._add_arrow(
                        m, 
                        pred['last_lat'], pred['last_lon'],
                        pred['pred_lat'], pred['pred_lon'],
                        color=color
                    )
                
                # Crear marcador para la posición predicha
                folium.CircleMarker(
                    location=[pred['pred_lat'], pred['pred_lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fill_opacity=0.8,
                    popup=folium.Popup(pred_popup_html, max_width=300)
                ).add_to(pred_group)
            
            pred_group.add_to(m)
        
        else:
            # Añadir mensaje informativo cuando no hay predicciones
            info_html = f'''
                <div style="position: fixed; top: 50px; right: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 1px solid orange; border-radius: 5px;">
                    <h4 style="color: orange; margin: 0;">Información</h4>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">
                        No hay predicciones disponibles para esta ventana de tiempo.
                        {'' if not show_uncertainty else 'Visualización de incertidumbre habilitada pero sin datos.'}
                    </p>
                </div>
            '''
            folium.Element(info_html).add_to(m)

        # Añadir información sobre predicciones
        if predictions_gdf is None or predictions_gdf.empty:
            title_html = f'''
                <div style="position: fixed; top: 50px; right: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 1px solid red; border-radius: 5px;">
                    <h4 style="color: red; margin: 0;">No hay predicciones disponibles</h4>
                    <p style="margin: 5px 0 0 0; font-size: 12px;">
                        Posible causa: Datos insuficientes para el modelo VAR
                    </p>
                </div>
            '''
            folium.Element(title_html).add_to(m)

        # Si se solicita mostrar incertidumbre y tenemos datos
        if show_uncertainty and uncertainty_data is not None and predictions_gdf is not None and not predictions_gdf.empty:
            uncertainty_group = folium.FeatureGroup(name='Zonas de Incertidumbre')
            
            # Usar UncertaintyModeling para crear elipses
            try:
                from src.models.uncertainty_modeling import UncertaintyModeling
                uncertainty_model = UncertaintyModeling()
                
                for i, (_, pred) in enumerate(predictions_gdf.iterrows()):
                    # Verificar si hay datos de incertidumbre
                    if 'expected_error_km' in pred:
                        # Usar track_id con verificación segura
                        if has_track_id:
                            track_id = pred['track_id']
                            track_label = f"Track #{track_id}"
                        else:
                            track_id = i
                            track_label = f"Predicción #{i+1}"
                        
                        # Obtener color consistente con el track
                        color = colors[track_id % len(colors)]
                        
                        # Crear elipses para diferentes niveles de confianza
                        confidence_levels = [
                            ('error_80ci_km', 0.2, '80% confianza'),
                            ('error_60ci_km', 0.3, '60% confianza'),
                            ('error_40ci_km', 0.4, '40% confianza')
                        ]
                        
                        # Usar el IC del 90% por defecto si no hay otros niveles
                        if not any(level in pred for level, _, _ in confidence_levels):
                            if 'error_90ci_km' in pred:
                                ellipse_points = uncertainty_model.create_uncertainty_ellipse(
                                    pred['pred_lat'], pred['pred_lon'], pred['error_90ci_km']
                                )
                                
                                folium.Polygon(
                                    locations=ellipse_points,
                                    color=color,
                                    weight=1,
                                    fill=True,
                                    fill_opacity=0.2,
                                    popup=f"Incertidumbre - 90% IC - {track_label}"
                                ).add_to(uncertainty_group)
                        else:
                            # Crear elipses para cada nivel de confianza configurado
                            for error_field, opacity, label in confidence_levels:
                                if error_field in pred:
                                    ellipse_points = uncertainty_model.create_uncertainty_ellipse(
                                        pred['pred_lat'], pred['pred_lon'], pred[error_field]
                                    )
                                    
                                    folium.Polygon(
                                        locations=ellipse_points,
                                        color=color,
                                        weight=1,
                                        fill=True,
                                        fill_opacity=opacity,
                                        popup=f"Incertidumbre - {label} - {track_label}"
                                    ).add_to(uncertainty_group)
                
                uncertainty_group.add_to(m)
                
                # Agregar leyenda de incertidumbre
                legend_html = '''
                <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 1px solid grey; border-radius: 5px;">
                    <h4 style="margin-top: 0;">Áreas de confianza</h4>
                    <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.4);"></span> 40% confianza</div>
                    <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.3);"></span> 60% confianza</div>
                    <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.2);"></span> 80% confianza</div>
                    <div style="margin-top: 5px;"><b>Pronóstico:</b> +15, +30, +45 min</div>
                </div>
                '''
                m.get_root().html.add_child(folium.Element(legend_html))
                
            except ImportError:
                logger.warning("No se pudo importar UncertaintyModeling para mostrar incertidumbre")
            except Exception as e:
                logger.error(f"Error al agregar incertidumbre: {e}")
        
        # Agregar controles de capas
        folium.LayerControl().add_to(m)
        
        # Añadir control de escala
        folium.plugins.MeasureControl(position='bottomleft', primary_length_unit='kilometers').add_to(m)

        return m
    
    def _get_direction(self, velocity_lon, velocity_lat):
        """
        Devuelve la dirección cardinal basada en componentes de velocidad.
        """
        angle = np.degrees(np.arctan2(velocity_lat, velocity_lon))
        
        # Convertir ángulo a dirección cardinal (N, NE, E, SE, S, SW, W, NW)
        directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE', 'E']
        idx = int(np.round((angle + 180) / 45)) % 8
        
        return directions[idx]
    
    def _add_arrow(self, folium_map, start_lat, start_lon, end_lat, end_lon, color='red'):
        """
        Añade un símbolo de flecha entre dos puntos en el mapa.
        """
        # Crear un marcador con icono de flecha
        arrow_icon = folium.features.DivIcon(
            icon_size=(20, 20),
            icon_anchor=(10, 10),
            html=f'<div style="font-size: 12pt; color: {color}; transform: rotate({self._calculate_bearing(start_lat, start_lon, end_lat, end_lon)}deg);">➤</div>'
        )
        
        # Calcular posición media para la flecha
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        
        # Añadir marcador con la flecha
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=arrow_icon
        ).add_to(folium_map)
    
    def _calculate_bearing(self, start_lat, start_lon, end_lat, end_lon):
        """
        Calcula el ángulo de orientación entre dos puntos.
        """
        start_lat = np.radians(start_lat)
        start_lon = np.radians(start_lon)
        end_lat = np.radians(end_lat)
        end_lon = np.radians(end_lon)
        
        d_lon = end_lon - start_lon
        
        y = np.sin(d_lon) * np.cos(end_lat)
        x = np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(d_lon)
        
        bearing = np.degrees(np.arctan2(y, x))
        
        return bearing
    
    def create_track_visualization(self, cells_gdf_list, timestamps, predictions_df_list=None):
        """
        Crea un mapa interactivo mostrando el trackeo de celdas y predicciones.
        
        Args:
            cells_gdf_list (list): Lista de GeoDataFrames con celdas en cada tiempo
            timestamps (list): Lista de timestamps para cada GeoDataFrame
            predictions_df_list (list): Lista de DataFrames con predicciones
            
        Returns:
            folium.Map: Mapa interactivo
        """
        # Verificar datos de entrada
        if not cells_gdf_list or len(cells_gdf_list) == 0:
            logger.warning("No cell data provided for track visualization")
            return folium.Map(location=[-34.0, -64.0], zoom_start=5)
        
        # Asegurar que tenemos timestamps para todos los datos
        if len(timestamps) != len(cells_gdf_list):
            logger.warning("Number of timestamps does not match number of cell data frames")
            timestamps = [datetime.now() + timedelta(minutes=i*10) for i in range(len(cells_gdf_list))]
        
        # Determinar el centro del mapa (usando el primer conjunto de datos)
        first_gdf = cells_gdf_list[0]
        if not first_gdf.empty:
            center_lat = first_gdf.centroid.y.mean()
            center_lon = first_gdf.centroid.x.mean()
        else:
            # Valores por defecto
            center_lat = -34.0
            center_lon = -64.0
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=7,
            tiles='CartoDB positron'
        )
        
        # Agregar control de medición
        m.add_child(MeasureControl())
        
        # Crear colormap para intensidad de celdas
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=0,
            vmax=500  # Ajustar según el máximo n_flashes en tus datos
        )
        colormap.caption = 'Intensidad (número de rayos)'
        m.add_child(colormap)
        
        # Lista de colores fijos para tracks
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
            "#80FF00", "#00FF80", "#FF8080", "#80FF80", "#8080FF", 
            "#FFFF80", "#FF80FF", "#80FFFF", "#FF4000", "#40FF00"
        ]
        
        # Preparar datos para TimestampedGeoJson
        features = []
        
        # Procesar cada intervalo de tiempo
        for i, (cells_gdf, timestamp) in enumerate(zip(cells_gdf_list, timestamps)):
            # Formatear timestamp como string
            time_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Procesar celdas detectadas
            if not cells_gdf.empty:
                for _, cell in cells_gdf.iterrows():
                    # Determinar color basado en track_id o intensidad
                    if 'track_id' in cells_gdf.columns:
                        track_id = cell.get('track_id', 0)
                        color = colors[track_id % len(colors)]
                    else:
                        color = colormap(cell['n_flashes'])
                    
                    # Crear popup con información detallada
                    popup_html = f"""
                    <div style="font-family: Arial; width: 200px;">
                        <h4>Celda #{cell['cell_id']}</h4>
                        <b>Rayos:</b> {cell['n_flashes']}<br>
                        <b>Área:</b> {cell['area_km2']:.1f} km²<br>
                        <b>Energía:</b> {cell['total_energy']:.2e}<br>
                        <b>Tiempo:</b> {time_str}<br>
                    """
                    
                    # Agregar información de track si está disponible
                    if 'track_id' in cell:
                        popup_html += f"<b>Track ID:</b> {cell['track_id']}<br>"
                    if 'age_minutes' in cell:
                        popup_html += f"<b>Edad:</b> {cell['age_minutes']:.1f} min<br>"
                    
                    popup_html += "</div>"
                    
                    # Crear feature para el polígono de la celda
                    if isinstance(cell.geometry, Polygon):
                        features.append({
                            'type': 'Feature',
                            #'geometry': json.loads(cell.geometry.to_json()),
                            'geometry': cell.geometry.__geo_interface__,
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': color,
                                    'fillOpacity': 0.5,
                                    'stroke': True,
                                    'color': 'black',
                                    'weight': 1
                                },
                                'style': {'weight': 0},
                                'popup': popup_html
                            }
                        })
                        
                        # Crear feature para el centroide
                        features.append({
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [cell['centroid_lon'], cell['centroid_lat']]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': 'white',
                                    'fillOpacity': 0.7,
                                    'stroke': True,
                                    'color': 'black',
                                    'radius': 3
                                },
                                'popup': popup_html
                            }
                        })
                        
                        # Agregar etiqueta con ID de celda y track
                        if 'track_id' in cell:
                            label_text = f"{cell['cell_id']} (T{cell['track_id']})"
                        else:
                            label_text = f"{cell['cell_id']}"
                            
                        features.append({
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [cell['centroid_lon'], cell['centroid_lat']]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'html': f'<div style="font-size: 10pt; color: black; background-color: white; border-radius: 3px; padding: 2px; border: 1px solid black;">{label_text}</div>',
                                    'iconSize': [30, 15],
                                    'iconAnchor': [15, 7]
                                }
                            }
                        })
            
            # Procesar predicciones si están disponibles para este tiempo
            if predictions_df_list and i < len(predictions_df_list) and not predictions_df_list[i].empty:
                pred_df = predictions_df_list[i]
                for _, pred in pred_df.iterrows():
                    # Usar el mismo color que el track correspondiente
                    track_id = pred['track_id']
                    color = colors[track_id % len(colors)]
                    
                    # Crear popup para la predicción
                    pred_popup = f"""
                    <div style="font-family: Arial; width: 220px;">
                        <h4>Predicción - Track #{pred['track_id']}</h4>
                        <b>Celda Original:</b> #{pred['last_cell_id']}<br>
                        <b>Tiempo Actual:</b> {pred['last_time']}<br>
                        <b>Tiempo Predicción:</b> {pred['pred_time']}<br>
                        <hr style="margin: 5px 0;">
                        <b>Rayos (actual):</b> {pred['last_n_flashes']}<br>
                        <b>Rayos (predicción):</b> {pred['pred_n_flashes']}<br>
                        <b>Área (actual):</b> {pred['last_area']:.1f} km²<br>
                        <b>Área (predicción):</b> {pred['pred_area']:.1f} km²<br>
                        <hr style="margin: 5px 0;">
                        <b>Velocidad:</b> {np.sqrt(pred['velocity_lon']**2 + pred['velocity_lat']**2):.2f} °/h<br>
                        <b>Dirección:</b> {self._get_direction(pred['velocity_lon'], pred['velocity_lat'])}
                    </div>
                    """
                    
                    # Crear feature para la posición predicha
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'Point',
                            'coordinates': [pred['pred_lon'], pred['pred_lat']]
                        },
                        'properties': {
                            'time': time_str,
                            'icon': 'circle',
                            'iconstyle': {
                                'fillColor': '#ff00ff',  # Magenta para predicciones
                                'fillOpacity': 0.8,
                                'stroke': True,
                                'color': 'black',
                                'radius': 8
                            },
                            'popup': pred_popup
                        }
                    })
                    
                    # Crear feature para la línea de trayectoria predicha
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': [
                                [pred['last_lon'], pred['last_lat']],
                                [pred['pred_lon'], pred['pred_lat']]
                            ]
                        },
                        'properties': {
                            'time': time_str,
                            'style': {
                                'color': color,
                                'weight': 3,
                                'opacity': 0.7,
                                'dashArray': '5, 5'  # Línea punteada
                            },
                            'popup': pred_popup
                        }
                    })
        
                # Procesar predicciones si están disponibles para este tiempo
                if predictions_df_list and i < len(predictions_df_list) and not predictions_df_list[i].empty:
                    pred_df = predictions_df_list[i]
                    for _, pred in pred_df.iterrows():
                        # [Código existente para dibujar puntos y líneas...]
                        
                        # Crear elipses de incertidumbre con diferentes niveles de confianza si están disponibles
                        if 'error_40ci_km' in pred:
                            # Obtener el mismo color que el track pero con diferentes opacidades
                            track_id = pred['track_id']
                            color = colors[track_id % len(colors)]
                            
                            # Crear elipse para 80% de confianza (menos opaca)
                            if 'error_80ci_km' in pred:
                                ellipse_points = self._create_uncertainty_ellipse(
                                    pred['pred_lat'], pred['pred_lon'], pred['error_80ci_km']
                                )
                                features.append({
                                    'type': 'Feature',
                                    'geometry': {
                                        'type': 'Polygon',
                                        'coordinates': [[lon, lat] for lat, lon in ellipse_points]
                                    },
                                    'properties': {
                                        'time': time_str,
                                        'style': {
                                            'color': color,
                                            'weight': 1,
                                            'fillColor': color,
                                            'fillOpacity': 0.15,
                                            'customLabel': '80% confianza'
                                        },
                                        'popup': f"Área de incertidumbre 80% - Track #{pred['track_id']}"
                                    }
                                })
                            
                            # Crear elipse para 60% de confianza (opacidad media)
                            if 'error_60ci_km' in pred:
                                ellipse_points = self._create_uncertainty_ellipse(
                                    pred['pred_lat'], pred['pred_lon'], pred['error_60ci_km']
                                )
                                features.append({
                                    'type': 'Feature',
                                    'geometry': {
                                        'type': 'Polygon',
                                        'coordinates': [[lon, lat] for lat, lon in ellipse_points]
                                    },
                                    'properties': {
                                        'time': time_str,
                                        'style': {
                                            'color': color,
                                            'weight': 1,
                                            'fillColor': color,
                                            'fillOpacity': 0.25,
                                            'customLabel': '60% confianza'
                                        },
                                        'popup': f"Área de incertidumbre 60% - Track #{pred['track_id']}"
                                    }
                                })
                            
                            # Crear elipse para 40% de confianza (más opaca)
                            ellipse_points = self._create_uncertainty_ellipse(
                                pred['pred_lat'], pred['pred_lon'], pred['error_40ci_km']
                            )
                            features.append({
                                'type': 'Feature',
                                'geometry': {
                                    'type': 'Polygon',
                                    'coordinates': [[lon, lat] for lat, lon in ellipse_points]
                                },
                                'properties': {
                                    'time': time_str,
                                    'style': {
                                        'color': color,
                                        'weight': 1,
                                        'fillColor': color,
                                        'fillOpacity': 0.35,
                                        'customLabel': '40% confianza'
                                    },
                                    'popup': f"Área de incertidumbre 40% - Track #{pred['track_id']}"
                                }
                            })



        # Crear líneas de trayectoria para cada track
        track_lines = self._create_track_lines(cells_gdf_list, timestamps)
        for track_line in track_lines:
            features.append(track_line)
        
        # Crear TimestampedGeoJson y agregarlo al mapa
        TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period='PT10M',  # Intervalo de tiempo (10 minutos)
            duration='PT1M',  # Duración de la transición (1 minuto)
            auto_play=False,
            loop=False
        ).add_to(m)
        
        # Agregar título
        if timestamps:
            start_time = min(timestamps)
            end_time = max(timestamps)
            title_html = f'''
                <h3 align="center" style="font-size:16px">
                    <b>GLM Lightning Tracking & Nowcasting</b><br>
                    {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%MUTC')}
                </h3>
            '''
            folium.Element(title_html).add_to(m)
        

        # Añadir trayectorias históricas
        for track_id in set(sum([list(gdf['track_id'].unique()) for gdf in cells_gdf_list if not gdf.empty], [])):
            track_points = []
            for i, gdf in enumerate(cells_gdf_list):
                if not gdf.empty and track_id in gdf['track_id'].values:
                    cell = gdf[gdf['track_id'] == track_id].iloc[0]
                    track_points.append({
                        'lat': cell['centroid_lat'],
                        'lon': cell['centroid_lon'],
                        'time': timestamps[i]
                    })
            
            # Ordenar por tiempo
            track_points.sort(key=lambda x: x['time'])
            
            # Dibujar línea de trayectoria si hay al menos 2 puntos
            if len(track_points) >= 2:
                folium.PolyLine(
                    locations=[[p['lat'], p['lon']] for p in track_points],
                    color=colors[track_id % len(colors)],
                    weight=3,
                    opacity=0.8,
                    tooltip=f"Track #{track_id} - Trayectoria"
                ).add_to(m)

        # Agregar leyenda de incertidumbre
        legend_html = '''
        <div style="position: fixed; bottom: 50px; right: 50px; z-index: 1000; background-color: white; 
        padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <h4 style="margin-top: 0;">Áreas de confianza</h4>
            <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.35);"></span> 40% confianza</div>
            <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.25);"></span> 60% confianza</div>
            <div><span style="display: inline-block; width: 15px; height: 15px; border-radius: 50%; background-color: rgba(255,0,0,0.15);"></span> 80% confianza</div>
            <div style="margin-top: 5px;"><b>Pronóstico:</b> +15, +30, +45 min</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))

        # Agregar leyenda
        colormap.add_to(m)
        
        # Agregar control de capas
        folium.LayerControl().add_to(m)
        
        return m
    
    
    def _create_uncertainty_ellipse(self, center_lat, center_lon, radius_km, num_points=36):
        """
        Crea coordenadas para una elipse de incertidumbre alrededor de un punto.
        
        Args:
            center_lat, center_lon: Coordenadas del centro de la elipse
            radius_km: Radio de incertidumbre en kilómetros
            num_points: Número de puntos para la elipse
            
        Returns:
            list: Lista de coordenadas [lat, lon] para la elipse
        """
        # Convertir km a grados (aproximación)
        # 1 grado de latitud ≈ 111 km
        # 1 grado de longitud ≈ 111 * cos(latitud) km
        lat_radius = radius_km / 111.0
        lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Generar puntos de la elipse
        ellipse_points = []
        for angle in np.linspace(0, 2*np.pi, num_points):
            lat = center_lat + lat_radius * np.sin(angle)
            lon = center_lon + lon_radius * np.cos(angle)
            ellipse_points.append([lat, lon])
        
        # Cerrar el polígono
        ellipse_points.append(ellipse_points[0])
        
        return ellipse_points
    
    def _create_track_lines(self, cells_gdf_list, timestamps):
        """
        Crea líneas que muestran la trayectoria histórica de cada track.
        
        Args:
            cells_gdf_list (list): Lista de GeoDataFrames con celdas detectadas
            timestamps (list): Lista de timestamps
        
        Returns:
            list: Lista de features GeoJSON para las líneas de trayectoria
        """
        features = []
        
        # Crear diccionario para almacenar posiciones por track_id
        track_positions = {}
        
        # Lista de colores fijos para tracks
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080"
        ]
        
        # Recopilar posiciones para cada track
        for gdf, timestamp in zip(cells_gdf_list, timestamps):
            if not gdf.empty and 'track_id' in gdf.columns:
                for _, cell in gdf.iterrows():
                    track_id = cell['track_id']
                    if track_id not in track_positions:
                        track_positions[track_id] = []
                    
                    track_positions[track_id].append({
                        'coords': [cell['centroid_lon'], cell['centroid_lat']],
                        'time': timestamp,
                        'n_flashes': cell['n_flashes'],
                        'cell_id': cell['cell_id']
                    })
        
        # Crear líneas para tracks con múltiples posiciones
        for track_id, positions in track_positions.items():
            if len(positions) > 1:
                # Ordenar por tiempo
                positions.sort(key=lambda x: x['time'])
                
                # Usar color consistente para cada track
                color = colors[track_id % len(colors)]
                
                # Crear una línea para cada intervalo de tiempo
                for i in range(1, len(positions)):
                    time_str = positions[i]['time'].strftime('%Y-%m-%dT%H:%M:%S')
                    
                    # Crear popup para la línea de trayectoria
                    track_popup = f"""
                    <div style="font-family: Arial; width: 180px;">
                        <h4>Trayectoria - Track #{track_id}</h4>
                        <b>Celda anterior:</b> #{positions[i-1]['cell_id']}<br>
                        <b>Celda actual:</b> #{positions[i]['cell_id']}<br>
                        <b>Tiempo:</b> {time_str}<br>
                        <b>Rayos actual:</b> {positions[i]['n_flashes']}<br>
                        <b>Distancia:</b> {self._calculate_distance(
                            positions[i-1]['coords'][0], positions[i-1]['coords'][1],
                            positions[i]['coords'][0], positions[i]['coords'][1]
                        ):.1f} km
                    </div>
                    """
                    
                    # Crear feature para la línea de trayectoria
                    features.append({
                        'type': 'Feature',
                        'geometry': {
                            'type': 'LineString',
                            'coordinates': [positions[i-1]['coords'], positions[i]['coords']]
                        },
                        'properties': {
                            'time': time_str,
                            'style': {
                                'color': color,
                                'weight': 3,
                                'opacity': 0.8
                            },
                            'popup': track_popup
                        }
                    })
        
        return features
    
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
    
def visualize_tracking_and_predictions(geojson_files, prediction_files, output_dir='./maps'):
    """
    Carga datos de archivos GeoJSON y CSV, y genera visualizaciones de trackeo y predicciones.
    
    Args:
        geojson_files (list): Lista de rutas a archivos GeoJSON con celdas
        prediction_files (list): Lista de rutas a archivos CSV con predicciones
        output_dir (str): Directorio para guardar las visualizaciones
        
    Returns:
        str: Ruta al archivo HTML generado
    """
    import pandas as pd
    import geopandas as gpd
    from datetime import datetime
    import os
    
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear visualizador
    visualizer = LightningVisualizer(output_dir=output_dir)
    
    # Cargar datos
    cells_gdf_list = []
    predictions_df_list = []
    timestamps = []
    
    # Ordenar archivos para asegurar secuencia temporal correcta
    geojson_files.sort()
    prediction_files.sort()
    
    # Cargar datos de celdas
    for geojson_file in geojson_files:
        # Extraer timestamp del nombre del archivo
        # Asumiendo formato como "cells_20241223_224000.geojson"
        filename = os.path.basename(geojson_file)
        time_str = filename.split('_')[-1].split('.')[0]
        year = int(time_str[:4])
        month = int(time_str[4:6])
        day = int(time_str[6:8])
        hour = int(time_str[8:10])
        minute = int(time_str[10:12])
        
        timestamp = datetime(year, month, day, hour, minute)
        timestamps.append(timestamp)
        
        # Cargar GeoJSON
        gdf = gpd.read_file(geojson_file)
        
        # Asignar track_id si hay predicción correspondiente
        pred_file = None
        for pf in prediction_files:
            if time_str in os.path.basename(pf):
                pred_file = pf
                break
        
        if pred_file:
            # Cargar predicciones
            pred_df = pd.read_csv(pred_file)
            
            # Agregar columna track_id al GeoDataFrame de celdas
            if 'track_id' not in gdf.columns and not pred_df.empty:
                # Crear diccionario cell_id -> track_id
                cell_to_track = {}
                for _, pred in pred_df.iterrows():
                    cell_to_track[pred['last_cell_id']] = pred['track_id']
                
                # Asignar track_id a cada celda
                gdf['track_id'] = gdf['cell_id'].apply(lambda x: cell_to_track.get(x, -1))
                
                # Calcular edad en minutos si hay track_id
                gdf['age_minutes'] = 0  # Inicializar
        
        cells_gdf_list.append(gdf)
        
        # Cargar predicciones
        if pred_file:
            pred_df = pd.read_csv(pred_file)
            predictions_df_list.append(pred_df)
        else:
            predictions_df_list.append(pd.DataFrame())  # DataFrame vacío como placeholder
    
    # Crear mapa con trackeo temporal
    track_map = visualizer.create_track_visualization(cells_gdf_list, timestamps, predictions_df_list)
    
    # Guardar mapa
    output_filename = f"storm_tracking_nowcast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    output_path = visualizer.save_interactive_map(track_map, filename=output_filename)
    
    print(f"Mapa generado y guardado en: {output_path}")
    
    return output_path


if __name__ == "__main__":
    import glob
    
    # Buscar archivos de datos
    geojson_files = sorted(glob.glob("data/cells_*.geojson"))
    prediction_files = sorted(glob.glob("data/predictions_*.csv"))
    
    # Generar visualización
    output_path = visualize_tracking_and_predictions(
        geojson_files,
        prediction_files,
        output_dir="./maps"
    )
    
    print(f"Mapa interactivo guardado en: {output_path}")
    print("Abre este archivo en tu navegador para ver la visualización.")