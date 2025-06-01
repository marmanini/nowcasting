# src/visualization/enhanced_lightning_visualizer.py

import folium
from folium.plugins import TimestampedGeoJson, MeasureControl, HeatMap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
import logging
import os
from datetime import datetime, timedelta
import json
import branca.colormap as cm

logger = logging.getLogger(__name__)

class EnhancedLightningVisualizer:
    """
    Visualizador mejorado con métricas de incertidumbre y rendimiento.
    """
    
    def __init__(self, output_dir=None):
        """
        Inicializa el visualizador mejorado.
        
        Args:
            output_dir (str): Directorio para guardar las visualizaciones
        """
        self.output_dir = output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Colores consistentes para tracks
        self.track_colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
            "#80FF00", "#00FF80", "#FF8080", "#80FF80", "#8080FF", 
            "#FFFF80", "#FF80FF", "#80FFFF", "#FF4000", "#40FF00"
        ]
        
        # Configuración de intervalos de confianza
        self.confidence_levels = [
            ('error_90ci_km', 0.1, '90% confianza', '#FF0000'),
            ('error_80ci_km', 0.15, '80% confianza', '#FF4000'),
            ('error_60ci_km', 0.25, '60% confianza', '#FF8000'),
            ('error_40ci_km', 0.35, '40% confianza', '#FFFF00')
        ]
    
    def create_consolidated_nowcast_map(self, historical_data, performance_metrics, 
                                      show_uncertainty=True, show_verification=True):
        """
        Crea un mapa consolidado que muestra el historial de tracking, 
        pronósticos actuales con incertidumbre, y métricas de rendimiento.
        
        Args:
            historical_data: Lista de datos históricos de ventanas temporales
            performance_metrics: Diccionario con métricas de rendimiento
            show_uncertainty: Si mostrar áreas de incertidumbre
            show_verification: Si mostrar resultados de verificación
            
        Returns:
            folium.Map: Mapa consolidado
        """
        if not historical_data:
            logger.warning("No hay datos históricos para visualizar")
            return self._create_empty_map()
        
        # Determinar centro del mapa
        latest_data = historical_data[-1]
        if not latest_data['cells_gdf'].empty:
            center_lat = latest_data['cells_gdf']['centroid_lat'].mean()
            center_lon = latest_data['cells_gdf']['centroid_lon'].mean()
        else:
            center_lat, center_lon = -34.0, -64.0
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='CartoDB positron'
        )
        
        # Agregar controles
        m.add_child(MeasureControl())
        
        # Agregar título y tiempo
        start_time = historical_data[0]['timestamp']
        end_time = historical_data[-1]['timestamp']
        self._add_title(m, start_time, end_time, len(historical_data))
        
        # Crear colormap para intensidad
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=0,
            vmax=500
        )
        colormap.caption = 'Intensidad de rayos'
        m.add_child(colormap)
        
        # 1. Agregar trayectorias históricas de tracks
        self._add_historical_tracks(m, historical_data)
        
        # 2. Agregar celdas del último tiempo
        self._add_current_cells(m, latest_data['cells_gdf'])
        
        # 3. Agregar pronósticos con incertidumbre
        if not latest_data['predictions_df'].empty:
            self._add_predictions_with_uncertainty(
                m, latest_data['predictions_df'], show_uncertainty
            )
        
        # 4. Agregar verificaciones si se solicita
        if show_verification:
            self._add_verification_results(m, historical_data)
        
        # 5. Agregar panel de métricas de rendimiento
        self._add_performance_panel(m, performance_metrics, latest_data)
        
        # 6. Agregar leyenda de incertidumbre
        if show_uncertainty:
            self._add_uncertainty_legend(m)
        
        # 7. Agregar control de capas
        folium.LayerControl().add_to(m)
        
        return m
    
    def _create_empty_map(self):
        """Crea un mapa vacío por defecto."""
        m = folium.Map(location=[-34.0, -64.0], zoom_start=5)
        folium.Marker(
            [-34.0, -64.0],
            popup="No hay datos disponibles",
            icon=folium.Icon(color='red', icon='info-sign')
        ).add_to(m)
        return m
    
    def _add_title(self, folium_map, start_time, end_time, n_windows):
        """Agrega título al mapa."""
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                    z-index: 1000; background-color: rgba(255,255,255,0.95); 
                    padding: 10px 20px; border: 2px solid #333; border-radius: 10px; 
                    font-family: Arial; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <h2 style="margin: 0; color: #333;">⚡ Sistema de Nowcasting GLM ⚡</h2>
            <div style="margin-top: 5px; font-size: 14px; color: #666;">
                📅 {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%H:%M UTC')} 
                | 📊 {n_windows} ventanas | ⏱️ {(n_windows * 10)} minutos
            </div>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(title_html))
    
    def _add_historical_tracks(self, folium_map, historical_data):
        """Agrega trayectorias históricas de todos los tracks."""
        track_group = folium.FeatureGroup(name='🛤️ Trayectorias Históricas')
        
        # Recopilar todas las posiciones por track_id
        track_positions = {}
        
        for window_data in historical_data:
            cells_gdf = window_data['cells_gdf']
            timestamp = window_data['timestamp']
            
            if not cells_gdf.empty and 'track_id' in cells_gdf.columns:
                for _, cell in cells_gdf.iterrows():
                    track_id = cell['track_id']
                    if track_id not in track_positions:
                        track_positions[track_id] = []
                    
                    track_positions[track_id].append({
                        'timestamp': timestamp,
                        'lat': cell['centroid_lat'],
                        'lon': cell['centroid_lon'],
                        'n_flashes': cell['n_flashes'],
                        'cell_id': cell['cell_id'],
                        'age_minutes': cell.get('age_minutes', 0)
                    })
        
        # Crear líneas para tracks con múltiples posiciones
        for track_id, positions in track_positions.items():
            if len(positions) > 1:
                # Ordenar por tiempo
                positions.sort(key=lambda x: x['timestamp'])
                
                # Color consistente para el track
                color = self.track_colors[track_id % len(self.track_colors)]
                
                # Crear línea de trayectoria
                coords = [[pos['lat'], pos['lon']] for pos in positions]
                
                # Información del track
                track_info = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>🛤️ Track #{track_id}</h4>
                    <b>Duración:</b> {len(positions)} observaciones<br>
                    <b>Tiempo total:</b> {(positions[-1]['timestamp'] - positions[0]['timestamp']).total_seconds()/60:.1f} min<br>
                    <b>Intensidad final:</b> {positions[-1]['n_flashes']} rayos<br>
                    <b>Distancia recorrida:</b> {self._calculate_track_distance(coords):.1f} km
                </div>
                """
                
                folium.PolyLine(
                    locations=coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=folium.Popup(track_info, max_width=250)
                ).add_to(track_group)
                
                # Agregar marcadores en posiciones clave
                # Posición inicial
                folium.CircleMarker(
                    location=[positions[0]['lat'], positions[0]['lon']],
                    radius=8,
                    color='green',
                    fill=True,
                    fillColor='lightgreen',
                    popup=f"🟢 Inicio Track #{track_id}"
                ).add_to(track_group)
                
                # Posición final
                folium.CircleMarker(
                    location=[positions[-1]['lat'], positions[-1]['lon']],
                    radius=8,
                    color=color,
                    fill=True,
                    fillColor=color,
                    popup=f"🔴 Actual Track #{track_id}"
                ).add_to(track_group)
        
        track_group.add_to(folium_map)
    
    def _add_current_cells(self, folium_map, cells_gdf):
        """Agrega celdas del tiempo actual."""
        if cells_gdf.empty:
            return
        
        cells_group = folium.FeatureGroup(name='⛈️ Celdas Actuales')
        
        for _, cell in cells_gdf.iterrows():
            # Color basado en track_id o intensidad
            if 'track_id' in cell:
                color = self.track_colors[cell['track_id'] % len(self.track_colors)]
            else:
                color = self._get_intensity_color(cell['n_flashes'])
            
            # Popup con información detallada
            popup_html = f"""
            <div style="font-family: Arial; width: 220px;">
                <h4>⛈️ Celda #{cell['cell_id']}</h4>
                <b>⚡ Rayos:</b> {cell['n_flashes']}<br>
                <b>📏 Área:</b> {cell['area_km2']:.1f} km²<br>
                <b>🔋 Energía:</b> {cell['total_energy']:.2e}<br>
                <b>📍 Posición:</b> [{cell['centroid_lat']:.4f}, {cell['centroid_lon']:.4f}]<br>
            """
            
            if 'track_id' in cell:
                popup_html += f"<b>🆔 Track:</b> #{cell['track_id']}<br>"
            if 'age_minutes' in cell:
                popup_html += f"<b>⏱️ Edad:</b> {cell['age_minutes']:.1f} min<br>"
            if 'prediction_error' in cell and not pd.isna(cell['prediction_error']):
                popup_html += f"<b>📊 Error anterior:</b> {cell['prediction_error']:.1f} km<br>"
            
            popup_html += "</div>"
            
            # Agregar polígono de la celda
            if isinstance(cell.geometry, Polygon):
                coords = [(y, x) for x, y in list(cell.geometry.exterior.coords)]
                
                folium.Polygon(
                    locations=coords,
                    color=color,
                    weight=3,
                    opacity=1.0,
                    fill=True,
                    fill_opacity=0.6,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(cells_group)
                
                # Centroide con etiqueta
                folium.CircleMarker(
                    location=[cell['centroid_lat'], cell['centroid_lon']],
                    radius=8,
                    color='white',
                    fill=True,
                    fillColor='white',
                    stroke=True,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=300)
                ).add_to(cells_group)
                
                # Etiqueta con información
                track_label = f"T{cell['track_id']}" if 'track_id' in cell else f"C{cell['cell_id']}"
                folium.Marker(
                    location=[cell['centroid_lat'], cell['centroid_lon']],
                    icon=folium.DivIcon(
                        icon_size=(30, 20),
                        icon_anchor=(15, 10),
                        html=f'<div style="font-size: 10pt; color: black; background-color: white; border-radius: 3px; padding: 2px; border: 1px solid black; text-align: center;"><b>{track_label}</b><br>{cell["n_flashes"]}⚡</div>'
                    )
                ).add_to(cells_group)
        
        cells_group.add_to(folium_map)
    
    def _add_predictions_with_uncertainty(self, folium_map, predictions_df, show_uncertainty=True):
        """Agrega pronósticos con zonas de incertidumbre - VERSIÓN CORREGIDA."""
        pred_group = folium.FeatureGroup(name='🔮 Pronósticos')
        
        for _, pred in predictions_df.iterrows():
            track_id = pred['track_id']
            color = self.track_colors[track_id % len(self.track_colors)]
            
            # ← CORRECCIÓN: Manejar información del pronóstico con columnas faltantes
            forecast_method = pred.get('forecast_method', 'unknown')
            
            # Buscar confianza en diferentes columnas posibles
            if 'confidence_percentage' in pred:
                confidence = pred['confidence_percentage']
            elif 'confidence_level' in pred:
                confidence = pred['confidence_level'] * 100  # Convertir a porcentaje
            elif 'overall_confidence' in pred:
                confidence = pred['overall_confidence'] * 100  # Convertir a porcentaje
            else:
                confidence = 75.0  # Valor por defecto
            
            expected_error = pred.get('expected_error_km', pred.get('uncertainty_lat', 0) * 111)  # Fallback
            
            pred_popup = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4>🔮 Pronóstico - Track #{track_id}</h4>
                <div style="background-color: #f0f8ff; padding: 8px; border-radius: 5px; margin: 5px 0;">
                    <b>📊 Confianza:</b> <span style="color: {'green' if confidence > 70 else 'orange' if confidence > 50 else 'red'}; font-weight: bold;">{confidence:.1f}%</span><br>
                    <b>🎯 Método:</b> {forecast_method}<br>
                    <b>📏 Error esperado:</b> ±{expected_error:.1f} km
                </div>
                <hr style="margin: 8px 0;">
                <b>📍 Posición actual:</b> [{pred.get('last_lat', 0):.4f}, {pred.get('last_lon', 0):.4f}]<br>
                <b>🎯 Posición predicha:</b> [{pred['pred_lat']:.4f}, {pred['pred_lon']:.4f}]<br>
                <b>⏰ Tiempo pronóstico:</b> {pred.get('pred_time', 'N/A')}<br>
                <hr style="margin: 8px 0;">
                <b>⚡ Rayos actual:</b> {pred.get('last_n_flashes', 'N/A')}<br>
                <b>⚡ Rayos predicho:</b> {pred.get('pred_n_flashes', 'N/A')}<br>
                <b>📏 Área actual:</b> {pred.get('last_area', 0):.1f} km²<br>
                <b>📏 Área predicha:</b> {pred.get('pred_area', 0):.1f} km²
            </div>
            """

            # Línea de trayectoria predicha
            if 'last_lat' in pred and 'last_lon' in pred:
                folium.PolyLine(
                    locations=[
                        [pred['last_lat'], pred['last_lon']],
                        [pred['pred_lat'], pred['pred_lon']]
                    ],
                    color=color,
                    weight=4,
                    opacity=0.8,
                    dash_array='10, 5',
                    popup=folium.Popup(pred_popup, max_width=300)
                ).add_to(pred_group)
                
                # Flecha direccional
                self._add_arrow_marker(
                    pred_group, 
                    pred['last_lat'], pred['last_lon'],
                    pred['pred_lat'], pred['pred_lon'],
                    color
                )
            
            # Marcador de posición predicha
            folium.CircleMarker(
                location=[pred['pred_lat'], pred['pred_lon']],
                radius=10,
                color='white',
                fill=True,
                fillColor=color,
                stroke=True,
                weight=3,
                popup=folium.Popup(pred_popup, max_width=300)
            ).add_to(pred_group)
            
            # Etiqueta con confianza
            folium.Marker(
                location=[pred['pred_lat'], pred['pred_lon']],
                icon=folium.DivIcon(
                    icon_size=(40, 25),
                    icon_anchor=(20, 12),
                    html=f'<div style="font-size: 9pt; color: white; background-color: {color}; border-radius: 3px; padding: 3px; border: 2px solid white; text-align: center;"><b>🔮 {confidence:.0f}%</b></div>'
                )
            ).add_to(pred_group)
            
            # Agregar zonas de incertidumbre si se solicita
            if show_uncertainty:
                self._add_uncertainty_zones(pred_group, pred, color)
        
        pred_group.add_to(folium_map)
    
    def _add_uncertainty_zones(self, feature_group, prediction, color):
        """Agrega zonas de incertidumbre como elipses concéntricas - VERSIÓN CORREGIDA."""
        
        # ← CORRECCIÓN: Verificar qué campos de incertidumbre existen
        available_uncertainty_fields = []
        
        for error_field, opacity, label, zone_color in self.confidence_levels:
            if error_field in prediction and prediction[error_field] > 0:
                available_uncertainty_fields.append((error_field, opacity, label, zone_color))
        
        # Si no hay campos específicos, usar campo genérico
        if not available_uncertainty_fields:
            # Buscar campos alternativos
            if 'expected_error_km' in prediction and prediction['expected_error_km'] > 0:
                available_uncertainty_fields.append(
                    ('expected_error_km', 0.3, 'Zona de incertidumbre', '#FF8000')
                )
            elif 'uncertainty_lat' in prediction:
                # Convertir uncertainty_lat (grados) a km aproximadamente
                uncertainty_km = prediction['uncertainty_lat'] * 111
                if uncertainty_km > 0:
                    available_uncertainty_fields.append(
                        ('uncertainty_lat_km', 0.3, 'Zona de incertidumbre', '#FF8000')
                    )
                    # Crear campo temporal
                    prediction['uncertainty_lat_km'] = uncertainty_km
        
        # Crear elipses para los campos disponibles
        for error_field, opacity, label, zone_color in available_uncertainty_fields:
            if error_field in prediction and prediction[error_field] > 0:
                # Crear elipse de incertidumbre
                ellipse_points = self._create_uncertainty_ellipse(
                    prediction['pred_lat'], 
                    prediction['pred_lon'], 
                    prediction[error_field]
                )
                
                folium.Polygon(
                    locations=ellipse_points,
                    color=zone_color,
                    weight=1,
                    fill=True,
                    fill_color=zone_color,
                    fill_opacity=opacity,
                    popup=f"Zona de {label} - Track #{prediction['track_id']}"
                ).add_to(feature_group)
    
    def _add_verification_results(self, folium_map, historical_data):
        """Agrega marcadores mostrando resultados de verificación."""
        verification_group = folium.FeatureGroup(name='✅ Verificaciones')
        
        for window_data in historical_data:
            verifications = window_data.get('verification_results', [])
            
            for verification in verifications:
                track_id = verification['track_id']
                error_km = verification['position_error_km']
                was_successful = verification['was_within_uncertainty']
                
                # Color basado en éxito/fallo
                marker_color = 'green' if was_successful else 'red'
                icon_symbol = 'ok' if was_successful else 'remove'
                
                # Información de verificación
                verification_popup = f"""
                <div style="font-family: Arial; width: 200px;">
                    <h4>✅ Verificación - Track #{track_id}</h4>
                    <b>Estado:</b> <span style="color: {marker_color};">{'✅ Exitosa' if was_successful else '❌ Fallida'}</span><br>
                    <b>Error posición:</b> {error_km:.1f} km<br>
                    <b>Error intensidad:</b> {verification['intensity_error_percentage']:.1f}%<br>
                    <b>Método:</b> {verification['forecast_method']}<br>
                    <b>Confianza:</b> {verification['predicted_confidence']:.1f}%<br>
                    <b>Tiempo:</b> {verification['observation_time'].strftime('%H:%M')}<br>
                </div>
                """
                
                # Buscar posición actual de la celda verificada
                cells_gdf = window_data['cells_gdf']
                track_cells = cells_gdf[cells_gdf['track_id'] == track_id]
                
                if not track_cells.empty:
                    cell = track_cells.iloc[0]
                    folium.Marker(
                        location=[cell['centroid_lat'], cell['centroid_lon']],
                        icon=folium.Icon(
                            color=marker_color, 
                            icon=icon_symbol,
                            prefix='glyphicon'
                        ),
                        popup=folium.Popup(verification_popup, max_width=250)
                    ).add_to(verification_group)
        
        verification_group.add_to(folium_map)
    
    def _add_performance_panel(self, folium_map, performance_metrics, latest_data):
        """Agrega panel con métricas de rendimiento en tiempo real - VERSIÓN CORREGIDA."""
        
        # Calcular estadísticas actuales
        total_preds = performance_metrics.get('total_predictions', 0)
        successful_verifs = performance_metrics.get('successful_verifications', 0)
        accuracy_rate = (successful_verifs / total_preds * 100) if total_preds > 0 else 0
        
        mean_pos_error = performance_metrics.get('mean_position_error_km', 0)
        mean_int_error = performance_metrics.get('mean_intensity_error_pct', 0)
        
        # ← CORRECCIÓN: Manejar predicciones con columnas faltantes
        latest_predictions = latest_data.get('predictions_df', pd.DataFrame())
        if not latest_predictions.empty:
            # Verificar qué columnas existen para confianza
            if 'confidence_percentage' in latest_predictions.columns:
                avg_confidence = latest_predictions['confidence_percentage'].mean()
            elif 'confidence_level' in latest_predictions.columns:
                # Convertir confidence_level (0-1) a porcentaje
                avg_confidence = latest_predictions['confidence_level'].mean() * 100
            elif 'overall_confidence' in latest_predictions.columns:
                # Convertir overall_confidence (0-1) a porcentaje
                avg_confidence = latest_predictions['overall_confidence'].mean() * 100
            else:
                # Valor por defecto si no hay columna de confianza
                avg_confidence = 75.0  # Asumir confianza moderada
            
            n_current_predictions = len(latest_predictions)
            
            # Verificar si existe la columna forecast_method
            if 'forecast_method' in latest_predictions.columns:
                methods_used = latest_predictions['forecast_method'].value_counts().to_dict()
                methods_str = ', '.join([f"{k}({v})" for k, v in methods_used.items()])
            else:
                methods_str = "Métodos no especificados"
        else:
            avg_confidence = 0
            n_current_predictions = 0
            methods_str = "Ninguno"

        # HTML del panel
        panel_html = f'''
        <div style="position: fixed; top: 80px; left: 10px; z-index: 1000; 
                    background-color: rgba(255,255,255,0.95); padding: 15px; 
                    border: 2px solid #333; border-radius: 10px; font-family: Arial;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3); max-width: 280px;">
            
            <h3 style="margin-top: 0; color: #333; text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
                📊 Métricas de Rendimiento
            </h3>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #f8f9fa; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #495057;">📈 Rendimiento General</h4>
                <div style="font-size: 12px;">
                    • <b>Predicciones totales:</b> {total_preds}<br>
                    • <b>Tasa de acierto:</b> <span style="color: {'green' if accuracy_rate > 70 else 'orange' if accuracy_rate > 50 else 'red'}; font-weight: bold;">{accuracy_rate:.1f}%</span><br>
                    • <b>Error promedio:</b> {mean_pos_error:.1f} km<br>
                    • <b>Error intensidad:</b> {mean_int_error:.1f}%
                </div>
            </div>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #e8f4fd; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #0366d6;">🔮 Pronóstico Actual</h4>
                <div style="font-size: 12px;">
                    • <b>Predicciones activas:</b> {n_current_predictions}<br>
                    • <b>Confianza promedio:</b> <span style="color: {'green' if avg_confidence > 70 else 'orange' if avg_confidence > 50 else 'red'}; font-weight: bold;">{avg_confidence:.1f}%</span><br>
                    • <b>Métodos utilizados:</b><br>
                    <div style="margin-left: 10px; font-size: 11px;">{methods_str}</div>
                </div>
            </div>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #fff3cd; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #856404;">⏱️ Estado del Sistema</h4>
                <div style="font-size: 12px;">
                    • <b>Tracks activos:</b> {len(latest_data.get('cells_gdf', []))}<br>
                    • <b>Ventanas procesadas:</b> {len([d for d in [latest_data] if d])}<br>
                    • <b>Última actualización:</b><br>
                    <div style="margin-left: 10px; font-size: 11px;">{datetime.now().strftime('%H:%M:%S')}</div>
                </div>
            </div>
            
            <div style="text-align: center; font-size: 10px; color: #6c757d; border-top: 1px solid #ddd; padding-top: 5px;">
                Sistema GLM Nowcasting v2.0
            </div>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(panel_html))
    
    def _add_uncertainty_legend(self, folium_map):
        """Agrega leyenda para las zonas de incertidumbre."""
        legend_html = '''
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 1000; 
                    background-color: rgba(255,255,255,0.95); padding: 12px; 
                    border: 2px solid #333; border-radius: 10px; font-family: Arial;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <h4 style="margin-top: 0; text-align: center; color: #333;">🎯 Zonas de Confianza</h4>
            <div style="margin-bottom: 8px;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,255,0,0.35); border: 1px solid #FFFF00; margin-right: 8px;"></span>
                    <span style="font-size: 12px;">40% confianza</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,128,0,0.25); border: 1px solid #FF8000; margin-right: 8px;"></span>
                    <span style="font-size: 12px;">60% confianza</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,64,0,0.15); border: 1px solid #FF4000; margin-right: 8px;"></span>
                    <span style="font-size: 12px;">80% confianza</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span style="display: inline-block; width: 20px; height: 15px; background-color: rgba(255,0,0,0.1); border: 1px solid #FF0000; margin-right: 8px;"></span>
                    <span style="font-size: 12px;">90% confianza</span>
                </div>
            </div>
            <div style="font-size: 11px; color: #666; border-top: 1px solid #ddd; padding-top: 5px;">
                <b>🎯 Pronóstico:</b> +20 minutos<br>
                <b>📏 Área:</b> Error esperado en km
            </div>
        </div>
        '''
        folium_map.get_root().html.add_child(folium.Element(legend_html))
    
    def _create_uncertainty_ellipse(self, center_lat, center_lon, radius_km, num_points=36):
        """Crea coordenadas para una elipse de incertidumbre."""
        # Convertir km a grados
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
    
    def _add_arrow_marker(self, feature_group, start_lat, start_lon, end_lat, end_lon, color):
        """Agrega marcador con flecha direccional."""
        # Calcular punto medio y ángulo
        mid_lat = (start_lat + end_lat) / 2
        mid_lon = (start_lon + end_lon) / 2
        bearing = self._calculate_bearing(start_lat, start_lon, end_lat, end_lon)
        
        # Crear icono de flecha
        arrow_html = f'''
        <div style="transform: rotate({bearing}deg); font-size: 16px; color: {color}; 
                    text-shadow: 1px 1px 2px white;">
            ➤
        </div>
        '''
        
        folium.Marker(
            location=[mid_lat, mid_lon],
            icon=folium.DivIcon(
                icon_size=(20, 20),
                icon_anchor=(10, 10),
                html=arrow_html
            )
        ).add_to(feature_group)
    
    def _calculate_bearing(self, start_lat, start_lon, end_lat, end_lon):
        """Calcula el ángulo de orientación entre dos puntos."""
        start_lat = np.radians(start_lat)
        start_lon = np.radians(start_lon)
        end_lat = np.radians(end_lat)
        end_lon = np.radians(end_lon)
        
        d_lon = end_lon - start_lon
        y = np.sin(d_lon) * np.cos(end_lat)
        x = np.cos(start_lat) * np.sin(end_lat) - np.sin(start_lat) * np.cos(end_lat) * np.cos(d_lon)
        
        bearing = np.degrees(np.arctan2(y, x))
        return bearing
    
    def _calculate_track_distance(self, coords):
        """Calcula la distancia total de una trayectoria."""
        total_distance = 0
        for i in range(1, len(coords)):
            lat1, lon1 = coords[i-1]
            lat2, lon2 = coords[i]
            distance = self._haversine_distance(lat1, lon1, lat2, lon2)
            total_distance += distance
        return total_distance
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calcula distancia usando fórmula de Haversine."""
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
    
    def _get_intensity_color(self, n_flashes):
        """Devuelve color basado en intensidad de rayos."""
        if n_flashes < 10:
            return 'blue'
        elif n_flashes < 50:
            return 'green'
        elif n_flashes < 100:
            return 'yellow'
        elif n_flashes < 200:
            return 'orange'
        else:
            return 'red'
    
    def save_map(self, folium_map, filename=None):
        """Guarda el mapa en un archivo HTML."""
        if not self.output_dir:
            logger.warning("Output directory not specified, cannot save map")
            return None
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"consolidated_nowcast_{timestamp}.html"
        
        file_path = os.path.join(self.output_dir, filename)
        
        try:
            folium_map.save(file_path)
            logger.info(f"Mapa consolidado guardado en: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"Error guardando mapa: {e}")
            return None
    
    def create_performance_dashboard(self, historical_data, performance_metrics):
        """
        Crea un dashboard adicional con gráficos de rendimiento histórico.
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            import base64
            from io import BytesIO
            
            # Crear figura con subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Dashboard de Rendimiento del Sistema de Nowcasting', fontsize=16, fontweight='bold')
            
            # Recopilar datos históricos
            timestamps = [d['timestamp'] for d in historical_data]
            errors = []
            confidences = []
            n_predictions_per_window = []
            
            for window_data in historical_data:
                verifications = window_data.get('verification_results', [])
                predictions = window_data.get('predictions_df', pd.DataFrame())
                
                if verifications:
                    window_errors = [v['position_error_km'] for v in verifications]
                    errors.extend(list(zip([window_data['timestamp']] * len(window_errors), window_errors)))
                
                if not predictions.empty:
                    avg_conf = predictions['confidence_percentage'].mean()
                    confidences.append((window_data['timestamp'], avg_conf))
                    n_predictions_per_window.append((window_data['timestamp'], len(predictions)))
            
            # Gráfico 1: Error de posición vs tiempo
            if errors:
                error_times, error_values = zip(*errors)
                ax1.scatter(error_times, error_values, alpha=0.6, color='red')
                ax1.set_title('Error de Posición vs Tiempo')
                ax1.set_ylabel('Error (km)')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax1.text(0.5, 0.5, 'No hay datos de verificación', ha='center', va='center', transform=ax1.transAxes)
                ax1.set_title('Error de Posición vs Tiempo')
            
            # Gráfico 2: Confianza promedio vs tiempo
            if confidences:
                conf_times, conf_values = zip(*confidences)
                ax2.plot(conf_times, conf_values, marker='o', linewidth=2, color='blue')
                ax2.set_title('Confianza Promedio vs Tiempo')
                ax2.set_ylabel('Confianza (%)')
                ax2.grid(True, alpha=0.3)
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax2.text(0.5, 0.5, 'No hay datos de confianza', ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title('Confianza Promedio vs Tiempo')
            
            # Gráfico 3: Número de predicciones por ventana
            if n_predictions_per_window:
                pred_times, pred_counts = zip(*n_predictions_per_window)
                ax3.bar(pred_times, pred_counts, alpha=0.7, color='green')
                ax3.set_title('Predicciones por Ventana Temporal')
                ax3.set_ylabel('Número de Predicciones')
                ax3.grid(True, alpha=0.3)
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            else:
                ax3.text(0.5, 0.5, 'No hay datos de predicciones', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Predicciones por Ventana Temporal')
            
            # Gráfico 4: Distribución de errores (histograma)
            if errors:
                _, error_values = zip(*errors)
                ax4.hist(error_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
                ax4.set_title('Distribución de Errores de Posición')
                ax4.set_xlabel('Error (km)')
                ax4.set_ylabel('Frecuencia')
                ax4.grid(True, alpha=0.3)
                
                # Agregar estadísticas
                mean_error = np.mean(error_values)
                median_error = np.median(error_values)
                ax4.axvline(mean_error, color='red', linestyle='--', label=f'Media: {mean_error:.1f} km')
                ax4.axvline(median_error, color='blue', linestyle='--', label=f'Mediana: {median_error:.1f} km')
                ax4.legend()
            else:
                ax4.text(0.5, 0.5, 'No hay datos de errores', ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Distribución de Errores de Posición')
            
            # Ajustar layout
            plt.tight_layout()
            
            # Convertir a imagen base64 para embebber en HTML
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode()
            plt.close()
            
            # Crear HTML con el dashboard
            dashboard_html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Dashboard de Rendimiento - GLM Nowcasting</title>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px; }}
                    .metric-card {{ background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 4px solid #007bff; }}
                    .metric-value {{ font-size: 24px; font-weight: bold; color: #007bff; }}
                    .metric-label {{ font-size: 14px; color: #6c757d; }}
                    .chart-container {{ text-align: center; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>📊 Dashboard de Rendimiento del Sistema GLM Nowcasting</h1>
                        <p>Generado el {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
                    
                    <div class="metrics">
                        <div class="metric-card">
                            <div class="metric-value">{performance_metrics.get('total_predictions', 0)}</div>
                            <div class="metric-label">Total Predicciones</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{(performance_metrics.get('successful_verifications', 0) / max(performance_metrics.get('total_predictions', 1), 1) * 100):.1f}%</div>
                            <div class="metric-label">Tasa de Acierto</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{performance_metrics.get('mean_position_error_km', 0):.1f} km</div>
                            <div class="metric-label">Error Promedio Posición</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{performance_metrics.get('mean_intensity_error_pct', 0):.1f}%</div>
                            <div class="metric-label">Error Promedio Intensidad</div>
                        </div>
                    </div>
                    
                    <div class="chart-container">
                        <img src="data:image/png;base64,{image_base64}" alt="Dashboard Charts" style="max-width: 100%; height: auto;">
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Guardar dashboard
            if self.output_dir:
                dashboard_path = os.path.join(self.output_dir, f"performance_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
                with open(dashboard_path, 'w', encoding='utf-8') as f:
                    f.write(dashboard_html)
                
                logger.info(f"Dashboard de rendimiento guardado en: {dashboard_path}")
                return dashboard_path
            
        except ImportError:
            logger.warning("Matplotlib no disponible, no se puede generar dashboard de rendimiento")
            return None
        except Exception as e:
            logger.error(f"Error generando dashboard de rendimiento: {e}")
            return None