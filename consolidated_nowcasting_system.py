#!/usr/bin/env python3
"""
Sistema consolidado de nowcasting GLM - VERSIÓN REAL QUE USA ARGUMENTOS
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from collections import deque
import json

# Agregar el directorio actual y src al path para importaciones locales
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, '..'))  # Para acceder a src desde nowcasting/
sys.path.insert(0, os.path.join(current_dir, '../src'))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('consolidated_nowcasting')

def import_components():
    """Importa todos los componentes necesarios de manera robusta."""
    components = {}
    
    # 1. Importar componentes básicos
    logger.info("Importando componentes básicos...")
    try:
        from src.data.glm_processor import GLMProcessor
        components['GLMProcessor'] = GLMProcessor
        logger.info("✓ GLMProcessor importado")
    except ImportError as e:
        logger.error(f"✗ Error importando GLMProcessor: {e}")
        return None
    
    try:
        from src.models.flash_cell_identification import FlashCellIdentifier
        components['FlashCellIdentifier'] = FlashCellIdentifier
        logger.info("✓ FlashCellIdentifier importado")
    except ImportError as e:
        logger.error(f"✗ Error importando FlashCellIdentifier: {e}")
        return None
    
    # 2. Importar componentes mejorados
    logger.info("Importando componentes mejorados...")
    
    # ImprovedFlashCellTracker
    ImprovedFlashCellTracker = None
    import_locations = [
        ('src.models.improved_flash_cell_tracking', 'desde src/models/'),
        ('improved_flash_cell_tracking', 'desde directorio actual')
    ]
    
    for module_path, description in import_locations:
        try:
            module = __import__(module_path, fromlist=['ImprovedFlashCellTracker'])
            ImprovedFlashCellTracker = getattr(module, 'ImprovedFlashCellTracker')
            logger.info(f"✓ ImprovedFlashCellTracker importado {description}")
            break
        except (ImportError, AttributeError):
            continue
    
    if ImprovedFlashCellTracker is None:
        logger.error("✗ No se pudo importar ImprovedFlashCellTracker")
        return None
    
    components['ImprovedFlashCellTracker'] = ImprovedFlashCellTracker
    
    # ImprovedFlashCellNowcaster
    ImprovedFlashCellNowcaster = None
    for module_path, description in import_locations:
        module_path = module_path.replace('tracking', 'nowcasting')
        try:
            module = __import__(module_path, fromlist=['ImprovedFlashCellNowcaster'])
            ImprovedFlashCellNowcaster = getattr(module, 'ImprovedFlashCellNowcaster')
            logger.info(f"✓ ImprovedFlashCellNowcaster importado {description}")
            break
        except (ImportError, AttributeError):
            continue
    
    if ImprovedFlashCellNowcaster is None:
        logger.error("✗ No se pudo importar ImprovedFlashCellNowcaster")
        return None
    
    components['ImprovedFlashCellNowcaster'] = ImprovedFlashCellNowcaster
    
    # 3. Importar visualizador (con fallback)
    logger.info("Importando visualizador...")
    
    # Intentar visualizador mejorado primero
    EnhancedLightningVisualizer = None
    enhanced_viz_locations = [
        ('src.visualization.enhanced_lightning_visualizer', 'visualizador mejorado'),
        ('enhanced_lightning_visualizer', 'visualizador mejorado desde directorio actual')
    ]
    
    for module_path, description in enhanced_viz_locations:
        try:
            module = __import__(module_path, fromlist=['EnhancedLightningVisualizer'])
            EnhancedLightningVisualizer = getattr(module, 'EnhancedLightningVisualizer')
            logger.info(f"✓ {description} importado")
            break
        except (ImportError, AttributeError):
            continue
    
    # Fallback al visualizador original
    if EnhancedLightningVisualizer is None:
        logger.info("Usando visualizador original como fallback...")
        LightningVisualizer = None
        viz_locations = [
            ('src.visualization.maps', 'desde src/visualization/'),
            ('maps', 'desde directorio actual')
        ]
        
        for module_path, description in viz_locations:
            try:
                module = __import__(module_path, fromlist=['LightningVisualizer'])
                LightningVisualizer = getattr(module, 'LightningVisualizer')
                logger.info(f"✓ LightningVisualizer (original) importado {description}")
                break
            except (ImportError, AttributeError):
                continue
        
        components['LightningVisualizer'] = LightningVisualizer
        components['EnhancedLightningVisualizer'] = None
    else:
        components['LightningVisualizer'] = None
        components['EnhancedLightningVisualizer'] = EnhancedLightningVisualizer
    
    logger.info("Componentes importados correctamente")
    return components

class ConsolidatedNowcastingSystem:
    """Sistema consolidado que USA ARGUMENTOS REALES, no valores hardcodeados."""
    
    def __init__(self, components, output_dir, history_minutes=40, min_history_minutes=20):
        self.components = components
        self.output_dir = output_dir
        self.history_minutes = history_minutes
        self.min_history_minutes = min_history_minutes
        
        # Almacenamiento de datos históricos
        self.historical_data = deque(maxlen=int(history_minutes/10))
        self.performance_metrics = {
            'total_predictions': 0,
            'successful_verifications': 0,
            'mean_position_error_km': 0,
            'mean_intensity_error_pct': 0
        }
    
    def initialize_components(self, args):
        """Inicializa componentes usando argumentos recibidos - NO HARDCODED."""
        try:
            logger.info(f"🔧 Inicializando con data_dir: {args.data_dir}")
            logger.info(f"🔧 Inicializando con output_dir: {args.output_dir}")
            
            # GLM Processor - USAR args.data_dir REAL
            self.glm_processor = self.components['GLMProcessor'](
                data_dir=args.data_dir  # ← CRÍTICO: usar argumento real
            )
            logger.info(f"✅ GLMProcessor inicializado con: {args.data_dir}")
            
            # Cell Identifier
            self.cell_identifier = self.components['FlashCellIdentifier'](
                eps=getattr(args, 'eps', 0.01),
                min_samples=getattr(args, 'min_samples', 3)
            )
            
            # Tracker
            self.tracker = self.components['ImprovedFlashCellTracker'](
                max_distance_km=getattr(args, 'max_distance_km', 30),
                max_speed_kmh=getattr(args, 'max_speed_kmh', 100),
                intensity_weight=0.3,
                size_weight=0.2,
                prediction_weight=0.4
            )
            
            # Nowcaster
            self.nowcaster = self.components['ImprovedFlashCellNowcaster'](
                forecast_minutes=getattr(args, 'forecast_minutes', 20),
                ensemble_models=getattr(args, 'ensemble_models', True),
                uncertainty_quantification=getattr(args, 'uncertainty', True)
            )
            
            # Visualizador - USAR self.output_dir REAL
            if self.components['EnhancedLightningVisualizer']:
                self.visualizer = self.components['EnhancedLightningVisualizer'](
                    output_dir=self.output_dir  # ← CRÍTICO: usar directorio real
                )
                self.enhanced_visualization = True
                logger.info(f"✅ Visualizador mejorado con: {self.output_dir}")
            elif self.components['LightningVisualizer']:
                self.visualizer = self.components['LightningVisualizer'](
                    output_dir=self.output_dir  # ← CRÍTICO: usar directorio real
                )
                self.enhanced_visualization = False
                logger.info(f"✅ Visualizador original con: {self.output_dir}")
            else:
                self.visualizer = None
                self.enhanced_visualization = False
                logger.warning("⚠️ Ningún visualizador disponible")
            
            logger.info("✅ Todos los componentes inicializados correctamente")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error inicializando componentes: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def process_time_window(self, start_time, end_time, window_index):
        """Procesa una ventana temporal."""
        logger.info(f"Procesando ventana {window_index}: {start_time} - {end_time}")
        
        try:
            # 1. Procesar datos GLM
            flash_df = self.glm_processor.process_time_window(start_time, end_time)
            
            if flash_df.empty:
                logger.warning(f"No flash data for window {window_index}")
                return None
            
            logger.info(f"Procesados {len(flash_df)} flashes en ventana {window_index}")
            
            # 2. Identificar celdas
            flash_df_with_clusters, cell_polygons, cell_stats = self.cell_identifier.identify_cells(flash_df)
            cells_gdf = self.cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)
            
            if cells_gdf.empty:
                logger.warning(f"No cells identified for window {window_index}")
                return None
            
            logger.info(f"Identificadas {len(cells_gdf)} celdas en ventana {window_index}")
            
            # 3. Tracking
            tracked_cells = self.tracker.track_cells(cells_gdf, end_time)
            logger.info(f"Tracked {len(tracked_cells)} celdas")
            
            # 4. Nowcasting
            predictions_df = self.nowcaster.predict_cells(tracked_cells, self.tracker.tracked_cells)
            logger.info(f"Generadas {len(predictions_df)} predicciones")
            
            # 5. Almacenar datos
            window_data = {
                'timestamp': end_time,
                'window_index': window_index,
                'flash_df': flash_df_with_clusters,
                'cells_gdf': tracked_cells,
                'predictions_df': predictions_df,
                'verification_results': [],  # Placeholder
                'track_stats': self.tracker.get_track_statistics()
            }
            
            self.historical_data.append(window_data)
            
            # 6. Guardar resultados parciales
            timestamp_str = end_time.strftime('%Y%m%d_%H%M%S')
            
            # Guardar celdas tracked
            if not tracked_cells.empty:
                cells_file = os.path.join(self.output_dir, f'tracked_cells_{timestamp_str}.geojson')
                try:
                    tracked_cells.to_file(cells_file, driver='GeoJSON')
                    logger.info(f"Guardadas {len(tracked_cells)} celdas tracked en {cells_file}")
                except Exception as e:
                    logger.warning(f"Error guardando celdas tracked: {e}")
            
            # Guardar predicciones
            if not predictions_df.empty:
                pred_file = os.path.join(self.output_dir, f'predictions_{timestamp_str}.csv')
                try:
                    predictions_df.to_csv(pred_file, index=False)
                    logger.info(f"Guardadas {len(predictions_df)} predicciones en {pred_file}")
                except Exception as e:
                    logger.warning(f"Error guardando predicciones: {e}")
            
            return window_data
            
        except Exception as e:
            logger.error(f"Error procesando ventana {window_index}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_consolidated_visualization(self):
        """Genera visualización consolidada."""
        if len(self.historical_data) < 1:
            logger.warning("Datos históricos insuficientes para visualización")
            return None
        
        if not self.visualizer:
            logger.warning("Visualizador no disponible")
            return None
        
        try:
            if self.enhanced_visualization:
                logger.info("Generando visualización con visualizador mejorado...")
                consolidated_map = self.visualizer.create_consolidated_nowcast_map(
                    historical_data=list(self.historical_data),
                    performance_metrics=self.performance_metrics,
                    show_uncertainty=True,
                    show_verification=True
                )
            else:
                logger.info("Generando visualización con visualizador original...")
                # Preparar datos para visualizador original
                all_cells_gdf = []
                all_predictions_df = []
                timestamps = []
                
                for window_data in self.historical_data:
                    all_cells_gdf.append(window_data['cells_gdf'])
                    all_predictions_df.append(window_data['predictions_df'])
                    timestamps.append(window_data['timestamp'])
                
                consolidated_map = self.visualizer.create_track_visualization(
                    cells_gdf_list=all_cells_gdf,
                    timestamps=timestamps,
                    predictions_df_list=all_predictions_df
                )
            
            # Guardar mapa
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'consolidated_nowcast_{timestamp_str}.html'
            
            if self.enhanced_visualization:
                saved_path = self.visualizer.save_map(consolidated_map, filename)
            else:
                saved_path = self.visualizer.save_interactive_map(consolidated_map, filename)
            
            logger.info(f"Visualización consolidada guardada en: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error generando visualización: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # def run_consolidated_analysis(self, start_time, end_time, args):
    #     """Ejecuta análisis consolidado usando argumentos reales - NO HARDCODED."""
    #     logger.info("=== INICIANDO ANÁLISIS CONSOLIDADO ===")
    #     logger.info(f"📁 Directorio de datos: {args.data_dir}")
    #     logger.info(f"📁 Directorio de salida: {args.output_dir}")
    #     logger.info(f"⏰ Período: {start_time} a {end_time}")
    #     logger.info(f"⏱️ Ventana: {args.window_minutes} minutos")
        
    #     current_time = start_time
    #     window_minutes = args.window_minutes  # ← USA ARGUMENTO REAL, NO HARDCODED
    #     window_index = 0
        
    #     while current_time < end_time:
    #         window_end = min(current_time + timedelta(minutes=window_minutes), end_time)
            
    #         # Procesar ventana
    #         window_data = self.process_time_window(current_time, window_end, window_index)
            
    #         # Verificar si tenemos suficientes datos para generar visualización
    #         if (len(self.historical_data) * window_minutes >= self.min_history_minutes and 
    #             len(self.historical_data) >= 2):
                
    #             logger.info(f"Generando visualización consolidada en ventana {window_index}")
    #             self.generate_consolidated_visualization()
            
    #         current_time = window_end
    #         window_index += 1
        
    #     # Generar visualización final
    #     final_map_path = self.generate_consolidated_visualization()
        
    #     # Generar reporte de rendimiento
    #     self._generate_performance_report()
        
    #     logger.info("=== ANÁLISIS CONSOLIDADO COMPLETADO ===")
    #     logger.info(f"🗺️ Mapa final: {final_map_path}")
        
    #     return final_map_path
    
    def run_consolidated_analysis(self, start_time, end_time, args):
        """Ejecuta análisis consolidado y genera UN SOLO HTML final con animación."""
        logger.info("=== INICIANDO ANÁLISIS CONSOLIDADO ===")
        logger.info(f"📁 Directorio de datos: {args.data_dir}")
        logger.info(f"📁 Directorio de salida: {args.output_dir}")
        logger.info(f"⏰ Período: {start_time} a {end_time}")
        logger.info(f"⏱️ Ventana: {args.window_minutes} minutos")
        
        current_time = start_time
        window_minutes = args.window_minutes
        window_index = 0
        
        # ← CAMBIO: Solo generar visualización intermedia si se solicita debug
        debug_mode = getattr(args, 'debug_visualizations', False)
        
        while current_time < end_time:
            window_end = min(current_time + timedelta(minutes=window_minutes), end_time)
            
            # Procesar ventana
            window_data = self.process_time_window(current_time, window_end, window_index)
            
            # ← CAMBIO: Solo generar visualizaciones intermedias en modo debug
            if (debug_mode and 
                len(self.historical_data) * window_minutes >= self.min_history_minutes and 
                len(self.historical_data) >= 2):
                
                logger.info(f"Generando visualización debug en ventana {window_index}")
                debug_path = self.generate_debug_visualization(window_index)
            
            current_time = window_end
            window_index += 1
        
        # ← CAMBIO: Solo generar visualización final al terminar TODAS las ventanas
        logger.info("Procesamiento de ventanas completado. Generando visualización final...")
        final_map_path = self.generate_final_consolidated_visualization()
        
        # Generar reporte de rendimiento
        self._generate_performance_report()
        
        logger.info("=== ANÁLISIS CONSOLIDADO COMPLETADO ===")
        logger.info(f"🗺️ Mapa final: {final_map_path}")
        
        return final_map_path

    def generate_debug_visualization(self, window_index):
        """Genera visualización de debug para una ventana específica."""
        if len(self.historical_data) < 1 or not self.visualizer:
            return None
        
        try:
            # Usar solo datos actuales para debug
            latest_data = self.historical_data[-1]
            
            if self.enhanced_visualization:
                debug_map = self.visualizer.create_consolidated_nowcast_map(
                    historical_data=[latest_data],  # Solo datos actuales
                    performance_metrics=self.performance_metrics,
                    show_uncertainty=True,
                    show_verification=False
                )
            else:
                debug_map = self.visualizer.create_interactive_map(
                    flash_df=latest_data['flash_df'],
                    cells_gdf=latest_data['cells_gdf'],
                    predictions_gdf=None,
                    start_time=latest_data['timestamp'] - timedelta(minutes=10),
                    end_time=latest_data['timestamp']
                )
            
            # Guardar con nombre específico de debug
            timestamp_str = latest_data['timestamp'].strftime('%Y%m%d_%H%M%S')
            debug_filename = f'debug_window_{window_index:02d}_{timestamp_str}.html'
            
            if self.enhanced_visualization:
                saved_path = self.visualizer.save_map(debug_map, debug_filename)
            else:
                saved_path = self.visualizer.save_interactive_map(debug_map, debug_filename)
            
            logger.info(f"Visualización debug guardada: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.warning(f"Error generando visualización debug: {e}")
            return None

    def generate_final_consolidated_visualization(self):
        """Genera la visualización final consolidada con ANIMACIÓN TEMPORAL."""
        if len(self.historical_data) < 1:
            logger.warning("No hay datos históricos para visualización final")
            return None
        
        if not self.visualizer:
            logger.warning("Visualizador no disponible")
            return None
        
        try:
            logger.info(f"Generando visualización final con {len(self.historical_data)} ventanas de datos")
            
            if self.enhanced_visualization:
                # ← CAMBIO: Usar método específico para animación temporal
                final_map = self._create_animated_consolidated_map()
            else:
                # Usar visualizador original que SÍ tiene animación
                all_cells_gdf = []
                all_predictions_df = []
                timestamps = []
                
                for window_data in self.historical_data:
                    all_cells_gdf.append(window_data['cells_gdf'])
                    all_predictions_df.append(window_data['predictions_df'])
                    timestamps.append(window_data['timestamp'])
                
                final_map = self.visualizer.create_track_visualization(
                    cells_gdf_list=all_cells_gdf,
                    timestamps=timestamps,
                    predictions_df_list=all_predictions_df
                )
            
            # Guardar mapa final
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            final_filename = f'FINAL_consolidated_nowcast_{timestamp_str}.html'
            
            if self.enhanced_visualization:
                saved_path = self.visualizer.save_map(final_map, final_filename)
            else:
                saved_path = self.visualizer.save_interactive_map(final_map, final_filename)
            
            logger.info(f"🎉 Visualización final consolidada guardada en: {saved_path}")
            return saved_path
            
        except Exception as e:
            logger.error(f"Error generando visualización final: {e}")
            import traceback
            traceback.print_exc()
            return None


    # FUNCIÓN CORREGIDA - ELIMINAR DUPLICACIÓN

    def _create_animated_consolidated_map(self):
        """Crea mapa con lógica temporal correcta: pasado=líneas, presente=área, futuro=cono."""
        
        import folium
        from folium.plugins import TimestampedGeoJson, MeasureControl
        import branca.colormap as cm
        from shapely.geometry import Polygon
        import math
        
        # DEBUG: Verificar datos disponibles
        logger.info(f"🔍 DEBUG: Ventanas de datos disponibles: {len(self.historical_data)}")
        for i, window_data in enumerate(self.historical_data):
            cells_count = len(window_data.get('cells_gdf', []))
            preds_count = len(window_data.get('predictions_df', []))
            logger.info(f"   Ventana {i}: {cells_count} celdas, {preds_count} predicciones")
        
        if len(self.historical_data) == 0:
            logger.error("❌ No hay datos históricos para visualizar")
            return None
        
        # Determinar centro del mapa
        all_cells = []
        for window_data in self.historical_data:
            cells_gdf = window_data.get('cells_gdf', pd.DataFrame())
            if not cells_gdf.empty:
                all_cells.append(cells_gdf)
        
        if all_cells:
            combined_cells = pd.concat(all_cells, ignore_index=True)
            center_lat = combined_cells['centroid_lat'].mean()
            center_lon = combined_cells['centroid_lon'].mean()
            logger.info(f"📍 Centro del mapa: [{center_lat:.4f}, {center_lon:.4f}]")
        else:
            center_lat, center_lon = -34.0, -64.0
            logger.warning("⚠️ Usando coordenadas por defecto para el centro")
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='CartoDB positron'
        )
        
        m.add_child(MeasureControl())
        
        # Crear colormap para intensidad
        colormap = cm.LinearColormap(
            colors=['blue', 'green', 'yellow', 'orange', 'red'],
            vmin=0,
            vmax=500
        )
        colormap.caption = 'Intensidad de rayos'
        m.add_child(colormap)
        
        # AGREGAR TÍTULO
        start_time = self.historical_data[0]['timestamp']
        end_time = self.historical_data[-1]['timestamp']
        title_html = f'''
        <div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
                    z-index: 1000; background-color: rgba(255,255,255,0.95); 
                    padding: 10px 20px; border: 2px solid #333; border-radius: 10px; 
                    font-family: Arial; text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <h2 style="margin: 0; color: #333;">⚡ Sistema de Nowcasting GLM - Temporal ⚡</h2>
            <div style="margin-top: 5px; font-size: 14px; color: #666;">
                📅 {start_time.strftime('%Y-%m-%d %H:%M')} → {end_time.strftime('%H:%M UTC')} 
                | 📊 {len(self.historical_data)} ventanas | ⏱️ {len(self.historical_data) * 10} minutos
            </div>
            <div style="margin-top: 3px; font-size: 12px; color: #888;">
                🕐 Pasado: líneas | 🕑 Presente: áreas | 🕒 Futuro: conos de probabilidad
            </div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # AGREGAR PANELES
        latest_data = self.historical_data[-1]
        self._add_complete_performance_panel(m, latest_data)
        self._add_verification_panel(m)
        self._add_uncertainty_legend_to_map(m)
        
        # COLORES PARA TRACKS
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080",
            "#80FF00", "#00FF80", "#FF8080", "#80FF80", "#8080FF"
        ]
        
        # RECOPILAR TODOS LOS TRACKS Y SUS POSICIONES TEMPORALES
        logger.info("🔄 Organizando datos temporales por tracks...")
        track_history = {}  # track_id -> lista de posiciones ordenadas por tiempo
        
        for window_idx, window_data in enumerate(self.historical_data):
            timestamp = window_data['timestamp']
            cells_gdf = window_data.get('cells_gdf', pd.DataFrame())
            predictions_df = window_data.get('predictions_df', pd.DataFrame())
            
            if not cells_gdf.empty:
                for _, cell in cells_gdf.iterrows():
                    track_id = cell.get('track_id', -1)
                    if track_id != -1:
                        if track_id not in track_history:
                            track_history[track_id] = []
                        
                        track_history[track_id].append({
                            'window_idx': window_idx,
                            'timestamp': timestamp,
                            'cell_data': cell,
                            'predictions': predictions_df[predictions_df['track_id'] == track_id] if not predictions_df.empty else pd.DataFrame()
                        })
        
        # Ordenar por tiempo cada track
        for track_id in track_history:
            track_history[track_id].sort(key=lambda x: x['timestamp'])
        
        logger.info(f"📊 Tracks organizados: {len(track_history)} tracks únicos")
        
        # FUNCIÓN AUXILIAR: Interpolar puntos entre dos posiciones
        def interpolate_points(lat1, lon1, lat2, lon2, num_points=8):
            """Crea puntos intermedios entre dos posiciones para simular una línea."""
            points = []
            for i in range(num_points + 1):
                ratio = i / num_points if num_points > 0 else 0
                lat = lat1 + (lat2 - lat1) * ratio
                lon = lon1 + (lon2 - lon1) * ratio
                points.append([lat, lon])
            return points
        
        # INICIALIZAR FEATURES UNA SOLA VEZ
        features = []
        total_features_added = 0
        
        # PROCESAMIENTO ÚNICO: LÍNEAS SIMULADAS + ÁREAS + CONOS + PREDICCIONES
        for current_window_idx, window_data in enumerate(self.historical_data):
            current_timestamp = window_data['timestamp']
            time_str = current_timestamp.strftime('%Y-%m-%d %H:%M:%S')
            
            logger.info(f"🕐 Procesando momento temporal {current_window_idx}: {current_timestamp.strftime('%H:%M')}")
            
            # Para cada track, procesar TODAS sus features en un solo lugar
            for track_id, track_positions in track_history.items():
                color = colors[int(track_id) % len(colors)]
                
                # Encontrar todas las posiciones hasta el momento actual (inclusive)
                positions_until_now = [
                    pos for pos in track_positions 
                    if pos['timestamp'] <= current_timestamp
                ]
                
                # 1. LÍNEAS SIMULADAS CON PUNTOS (si hay múltiples posiciones)
                if len(positions_until_now) >= 2:
                    logger.info(f"   Track {track_id}: creando línea simulada con {len(positions_until_now)} segmentos")
                    
                    # Crear puntos interpolados entre cada par de posiciones consecutivas
                    all_line_points = []
                    for i in range(len(positions_until_now) - 1):
                        pos1 = positions_until_now[i]
                        pos2 = positions_until_now[i + 1]
                        
                        lat1 = pos1['cell_data']['centroid_lat']
                        lon1 = pos1['cell_data']['centroid_lon']
                        lat2 = pos2['cell_data']['centroid_lat']
                        lon2 = pos2['cell_data']['centroid_lon']
                        
                        # Interpolar 8 puntos entre cada par de posiciones (más denso)
                        interpolated = interpolate_points(lat1, lon1, lat2, lon2, 8)
                        all_line_points.extend(interpolated)
                    
                    # Crear un punto pequeño para cada posición interpolada
                    for point_idx, (lat, lon) in enumerate(all_line_points):
                        line_point_feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [lon, lat]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': '#006400',    # Verde oscuro
                                    'fillOpacity': 0.9,         # Más opaco
                                    'stroke': False,            # Sin borde
                                    'radius': 3                 # Un poco más grande
                                },
                                'popup': f'🛤️ Trayectoria Track {track_id} - Punto {point_idx+1}'
                            }
                        }
                        
                        features.append(line_point_feature)
                        total_features_added += 1
                
                # 2. PUNTOS PRINCIPALES DE POSICIÓN
                if positions_until_now:
                    for i, pos in enumerate(positions_until_now):
                        # Punto principal de posición (más grande y distintivo)
                        main_point_feature = {
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [pos['cell_data']['centroid_lon'], pos['cell_data']['centroid_lat']]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': '#00FF00' if i == len(positions_until_now)-1 else '#228B22',
                                    'fillOpacity': 1.0,
                                    'stroke': True,
                                    'color': 'white',
                                    'weight': 2,
                                    'radius': 8 if i == len(positions_until_now)-1 else 5
                                },
                                'popup': f'📍 Track {track_id} - Posición {i+1} ({pos["timestamp"].strftime("%H:%M")})'
                            }
                        }
                        
                        features.append(main_point_feature)
                        total_features_added += 1
                
                # 3. ÁREA DE TORMENTA ACTUAL (SOLO TIEMPO PRESENTE)
                current_position = next(
                    (pos for pos in positions_until_now if pos['timestamp'] == current_timestamp), 
                    None
                )
                
                if current_position:
                    cell_data = current_position['cell_data']
                    
                    # Área/polígono de la tormenta actual
                    if hasattr(cell_data, 'geometry') and cell_data.geometry is not None:
                        if isinstance(cell_data.geometry, Polygon):
                            
                            cell_popup = f"""
                            <div style="font-family: Arial; width: 260px;">
                                <h4 style="color: {color};">⛈️ Tormenta Actual - Track #{track_id}</h4>
                                <div style="background-color: #fff3cd; padding: 6px; border-radius: 4px; margin: 4px 0;">
                                    <b>⚡ Rayos:</b> {cell_data.get('n_flashes', 'N/A')}<br>
                                    <b>📏 Área:</b> {cell_data.get('area_km2', 0):.1f} km²<br>
                                    <b>🔋 Energía total:</b> {cell_data.get('total_energy', 0):.1e}<br>
                                    <b>👴 Edad:</b> {cell_data.get('age_minutes', 0):.1f} min
                                </div>
                                <div style="background-color: #f8f9fa; padding: 6px; border-radius: 4px; margin: 4px 0;">
                                    <b>🕐 Tiempo:</b> {current_timestamp.strftime('%H:%M:%S')}<br>
                                    <b>📍 Centro:</b> [{cell_data.get('centroid_lat', 0):.4f}, {cell_data.get('centroid_lon', 0):.4f}]<br>
                                    <b>🚗 Velocidad:</b> {cell_data.get('velocity_kmh', 0):.1f} km/h
                                </div>
                            </div>
                            """
                            
                            # Polígono de la tormenta (solo en tiempo presente) - COLOR AZUL
                            features.append({
                                'type': 'Feature',
                                'geometry': cell_data.geometry.__geo_interface__,
                                'properties': {
                                    'time': time_str,
                                    'style': {
                                        'color': '#0000FF',        # AZUL para perímetro
                                        'weight': 4,               # PERÍMETRO MÁS RESALTADO
                                        'fillColor': '#0000FF',    # AZUL para relleno
                                        'fillOpacity': 0.4         # ALPHA 0.4 como solicitado
                                    },
                                    'popup': cell_popup
                                }
                            })
                            total_features_added += 1
                    
                    # 4. CENTROIDE DE LA TORMENTA ACTUAL
                    if 'centroid_lat' in cell_data and 'centroid_lon' in cell_data:
                        features.append({
                            'type': 'Feature',
                            'geometry': {
                                'type': 'Point',
                                'coordinates': [cell_data['centroid_lon'], cell_data['centroid_lat']]
                            },
                            'properties': {
                                'time': time_str,
                                'icon': 'circle',
                                'iconstyle': {
                                    'fillColor': 'white',
                                    'fillOpacity': 1.0,
                                    'stroke': True,
                                    'color': color,
                                    'weight': 3,
                                    'radius': 8
                                },
                                'popup': cell_popup
                            }
                        })
                        total_features_added += 1
                    
                    # 5. CONO DE INCERTIDUMBRE FUTURO (PRONÓSTICO)
                    predictions = current_position['predictions']
                    if not predictions.empty:
                        prediction = predictions.iloc[0]  # Tomar primera predicción
                        
                        # Generar cono de incertidumbre con múltiples niveles de probabilidad
                        uncertainty_features = self._generate_uncertainty_cone(
                            current_lat=cell_data['centroid_lat'],
                            current_lon=cell_data['centroid_lon'], 
                            predicted_lat=prediction.get('pred_lat', cell_data['centroid_lat']),
                            predicted_lon=prediction.get('pred_lon', cell_data['centroid_lon']),
                            time_str=time_str,
                            track_id=track_id,
                            color=color,
                            prediction_data=prediction
                        )
                        
                        features.extend(uncertainty_features)
                        total_features_added += len(uncertainty_features)

        logger.info(f"✅ Total features procesados: {total_features_added}")
        
        if total_features_added == 0:
            logger.error("❌ No se pudieron agregar features al mapa")
            return m
        
        # CREAR ANIMACIÓN TEMPORAL
        try:
            timestamped_geojson = TimestampedGeoJson(
                {
                    'type': 'FeatureCollection',
                    'features': features
                },
                period='PT10M',          # Cada 10 minutos (ventanas reales)
                duration='PT8M',         # Transición de 8 minutos
                auto_play=True,
                loop=True,
                max_speed=1,
                loop_button=True,
                date_options='YYYY-MM-DD HH:mm:ss',
                time_slider_drag_update=True
            )
            
            timestamped_geojson.add_to(m)
            logger.info("✅ Animación temporal con lógica correcta agregada al mapa")
            
        except Exception as e:
            logger.error(f"❌ Error creando animación temporal: {e}")
            return m
        
        # AGREGAR CONTROLES
        folium.LayerControl().add_to(m)
        
        return m

    def _calculate_trajectory_distance(self, positions):
        """Calcula la distancia total de una trayectoria."""
        total_distance = 0
        for i in range(1, len(positions)):
            lat1 = positions[i-1]['cell_data']['centroid_lat']
            lon1 = positions[i-1]['cell_data']['centroid_lon']
            lat2 = positions[i]['cell_data']['centroid_lat']
            lon2 = positions[i]['cell_data']['centroid_lon']
            
            distance = self._calculate_distance_km(lat1, lon1, lat2, lon2)
            total_distance += distance
        
        return total_distance

    def _calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """Calcula distancia entre dos puntos en km."""
        lat_diff = lat2 - lat1
        lon_diff = lon2 - lon1
        lat_km = lat_diff * 111.0
        lon_km = lon_diff * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))
        return (lat_km**2 + lon_km**2)**0.5

    def _generate_uncertainty_cone(self, current_lat, current_lon, predicted_lat, predicted_lon, 
                                time_str, track_id, color, prediction_data):
        """Genera cono de incertidumbre con probabilidades 60%, 80%, 90% en ROJO."""
        features = []
        
        # Configuración de incertidumbre - COLORES ROJOS
        base_error_km = prediction_data.get('expected_error_km', 5.0)
        confidence_levels = [
            {'probability': 60, 'multiplier': 1.0, 'opacity': 0.6, 'color': '#FF0000'},  # Rojo más opaco
            {'probability': 80, 'multiplier': 1.5, 'opacity': 0.4, 'color': '#FF0000'}, # Rojo medio
            {'probability': 90, 'multiplier': 2.0, 'opacity': 0.25, 'color': '#FF0000'} # Rojo más transparente
        ]
        
        # Calcular dirección del movimiento
        direction_lat = predicted_lat - current_lat
        direction_lon = predicted_lon - current_lon
        prediction_distance_km = self._calculate_distance_km(current_lat, current_lon, predicted_lat, predicted_lon)
        
        # Si no hay movimiento significativo, usar círculos concéntricos ROJOS
        if prediction_distance_km < 1.0:
            for level in confidence_levels:
                radius_km = base_error_km * level['multiplier']
                circle_coords = self._create_uncertainty_circle(predicted_lat, predicted_lon, radius_km)
                
                uncertainty_popup = f"""
                <div style="font-family: Arial; width: 240px;">
                    <h4 style="color: {level['color']};">🎯 Incertidumbre {level['probability']}% - Track #{track_id}</h4>
                    <div style="background-color: #ffe0e0; padding: 6px; border-radius: 4px;">
                        <b>📊 Probabilidad:</b> {level['probability']}%<br>
                        <b>📏 Radio:</b> ±{radius_km:.1f} km<br>
                        <b>🎯 Tipo:</b> Zona estacionaria<br>
                        <b>🔮 Método:</b> {prediction_data.get('forecast_method', 'Linear')}
                    </div>
                </div>
                """
                
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [circle_coords]
                    },
                    'properties': {
                        'time': time_str,
                        'style': {
                            'color': level['color'],           # ROJO para perímetro
                            'weight': 2,
                            'fillColor': level['color'],       # ROJO para relleno
                            'fillOpacity': level['opacity']    # Transparencia variable
                        },
                        'popup': uncertainty_popup
                    }
                })
        
        else:
            # Crear cono de incertidumbre direccional ROJO
            for level in confidence_levels:
                cone_coords = self._create_uncertainty_cone_coords(
                    current_lat, current_lon,
                    predicted_lat, predicted_lon,
                    base_error_km * level['multiplier']
                )
                
                uncertainty_popup = f"""
                <div style="font-family: Arial; width: 260px;">
                    <h4 style="color: {level['color']};">🌪️ Cono de Incertidumbre {level['probability']}% - Track #{track_id}</h4>
                    <div style="background-color: #ffe0e0; padding: 6px; border-radius: 4px; margin: 4px 0;">
                        <b>📊 Probabilidad:</b> {level['probability']}%<br>
                        <b>📏 Error máximo:</b> ±{base_error_km * level['multiplier']:.1f} km<br>
                        <b>🚗 Distancia pronóstico:</b> {prediction_distance_km:.1f} km<br>
                        <b>⏱️ Lead time:</b> {prediction_data.get('lead_time_min', 20)} min
                    </div>
                    <div style="background-color: #fff0f0; padding: 6px; border-radius: 4px; margin: 4px 0;">
                        <b>🎯 Desde:</b> [{current_lat:.4f}, {current_lon:.4f}]<br>
                        <b>🎯 Hacia:</b> [{predicted_lat:.4f}, {predicted_lon:.4f}]<br>
                        <b>🔮 Método:</b> {prediction_data.get('forecast_method', 'Linear')}<br>
                        <b>🎲 Confianza:</b> {prediction_data.get('confidence_level', 0.75)*100:.1f}%
                    </div>
                </div>
                """
                
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [cone_coords]
                    },
                    'properties': {
                        'time': time_str,
                        'style': {
                            'color': level['color'],           # ROJO para perímetro
                            'weight': 2,
                            'fillColor': level['color'],       # ROJO para relleno  
                            'fillOpacity': level['opacity']    # Transparencia variable
                        },
                        'popup': uncertainty_popup
                    }
                })
        
        # Agregar punto de predicción central - MANTENER MAGENTA
        prediction_popup = f"""
        <div style="font-family: Arial; width: 220px;">
            <h4 style="color: #FF00FF;">🔮 Pronóstico Central - Track #{track_id}</h4>
            <div style="background-color: #f0f8ff; padding: 6px; border-radius: 4px;">
                <b>📍 Posición predicha:</b> [{predicted_lat:.4f}, {predicted_lon:.4f}]<br>
                <b>⏱️ Lead time:</b> {prediction_data.get('lead_time_min', 20)} min<br>
                <b>🎲 Confianza:</b> {prediction_data.get('confidence_level', 0.75)*100:.1f}%<br>
                <b>📏 Error esperado:</b> ±{base_error_km:.1f} km
            </div>
        </div>
        """
        
        features.append({
            'type': 'Feature',
            'geometry': {
                'type': 'Point',
                'coordinates': [predicted_lon, predicted_lat]
            },
            'properties': {
                'time': time_str,
                'icon': 'circle',
                'iconstyle': {
                    'fillColor': '#FF00FF',        # MANTENER MAGENTA para punto central
                    'fillOpacity': 1.0,
                    'stroke': True,
                    'color': 'white',
                    'weight': 3,
                    'radius': 10
                },
                'popup': prediction_popup
            }
        })
        
        return features

    def _create_uncertainty_circle(self, center_lat, center_lon, radius_km, num_points=24):
        """Crea coordenadas para un círculo de incertidumbre."""
        lat_radius = radius_km / 111.0
        lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        circle_points = []
        for angle in np.linspace(0, 2*np.pi, num_points):
            lat = center_lat + lat_radius * np.sin(angle)
            lon = center_lon + lon_radius * np.cos(angle)
            circle_points.append([lon, lat])
        
        circle_points.append(circle_points[0])  # Cerrar el polígono
        return circle_points

    def _create_uncertainty_cone_coords(self, current_lat, current_lon, predicted_lat, predicted_lon, max_error_km):
        """Crea coordenadas para un cono de incertidumbre direccional."""
        import math
        
        # Vector de dirección
        direction_lat = predicted_lat - current_lat
        direction_lon = predicted_lon - current_lon
        
        # Normalizar dirección
        distance = math.sqrt(direction_lat**2 + direction_lon**2)
        if distance == 0:
            return self._create_uncertainty_circle(predicted_lat, predicted_lon, max_error_km)
        
        unit_lat = direction_lat / distance
        unit_lon = direction_lon / distance
        
        # Vector perpendicular
        perp_lat = -unit_lon
        perp_lon = unit_lat
        
        # Conversión a km
        lat_to_km = 111.0
        lon_to_km = 111.0 * np.cos(np.radians((current_lat + predicted_lat) / 2))
        
        # Crear cono: estrecho en origen, ancho en destino
        origin_width_km = max_error_km * 0.2  # 20% del error en el origen
        dest_width_km = max_error_km          # 100% del error en destino
        
        # Puntos del cono
        cone_points = []
        
        # Lado izquierdo del cono
        left_origin_lat = current_lat + (perp_lat * origin_width_km / lat_to_km)
        left_origin_lon = current_lon + (perp_lon * origin_width_km / lon_to_km)
        left_dest_lat = predicted_lat + (perp_lat * dest_width_km / lat_to_km)
        left_dest_lon = predicted_lon + (perp_lon * dest_width_km / lon_to_km)
        
        # Lado derecho del cono
        right_origin_lat = current_lat - (perp_lat * origin_width_km / lat_to_km)
        right_origin_lon = current_lon - (perp_lon * origin_width_km / lon_to_km)
        right_dest_lat = predicted_lat - (perp_lat * dest_width_km / lat_to_km)
        right_dest_lon = predicted_lon - (perp_lon * dest_width_km / lon_to_km)
        
        # Construir polígono del cono
        cone_points = [
            [left_origin_lon, left_origin_lat],    # Inicio izquierdo
            [left_dest_lon, left_dest_lat],        # Destino izquierdo
            [right_dest_lon, right_dest_lat],      # Destino derecho
            [right_origin_lon, right_origin_lat],  # Inicio derecho
            [left_origin_lon, left_origin_lat]     # Cerrar polígono
        ]
        
        return cone_points


    def perform_forecast_verification(self, current_window_data, previous_predictions):
        """
        Verifica pronósticos del tiempo t-1 contra observaciones del tiempo t.
        
        Args:
            current_window_data: Datos actuales observados (tiempo t)
            previous_predictions: Predicciones hechas en tiempo t-1
        
        Returns:
            verification_results: Lista con resultados de verificación
        """
        verification_results = []
        
        if not previous_predictions or len(previous_predictions) == 0:
            return verification_results
        
        current_cells = current_window_data.get('cells_gdf', pd.DataFrame())
        if current_cells.empty:
            return verification_results
        
        logger.info(f"🔍 Verificando {len(previous_predictions)} pronósticos contra {len(current_cells)} observaciones")
        
        for pred_idx, prediction in enumerate(previous_predictions):
            try:
                track_id = prediction.get('track_id', -1)
                pred_lat = prediction.get('pred_lat', None)
                pred_lon = prediction.get('pred_lon', None)
                pred_time = prediction.get('pred_time', current_window_data['timestamp'])
                confidence_level = prediction.get('confidence_level', 0.75)
                forecast_method = prediction.get('forecast_method', 'unknown')
                expected_error_km = prediction.get('expected_error_km', 10.0)
                
                if pred_lat is None or pred_lon is None:
                    continue
                
                # Buscar la celda observada correspondiente al mismo track_id
                observed_cells = current_cells[current_cells['track_id'] == track_id]
                
                if observed_cells.empty:
                    # El track desapareció - predicción fallida
                    verification_result = {
                        'track_id': track_id,
                        'prediction_time': pred_time - timedelta(minutes=20),  # Tiempo cuando se hizo la predicción
                        'observation_time': current_window_data['timestamp'],
                        'predicted_lat': pred_lat,
                        'predicted_lon': pred_lon,
                        'observed_lat': None,
                        'observed_lon': None,
                        'position_error_km': float('inf'),
                        'intensity_error_pct': float('inf'),
                        'was_within_uncertainty': False,
                        'track_disappeared': True,
                        'confidence_level': confidence_level,
                        'forecast_method': forecast_method,
                        'expected_error_km': expected_error_km
                    }
                    verification_results.append(verification_result)
                    continue
                
                # Tomar la celda observada (debería ser única por track_id)
                observed_cell = observed_cells.iloc[0]
                obs_lat = observed_cell['centroid_lat']
                obs_lon = observed_cell['centroid_lon']
                
                # Calcular error de posición en km
                position_error_km = self._calculate_distance_km(pred_lat, pred_lon, obs_lat, obs_lon)
                
                # Calcular error de intensidad
                pred_intensity = prediction.get('pred_n_flashes', 0)
                obs_intensity = observed_cell.get('n_flashes', 0)
                
                if obs_intensity > 0:
                    intensity_error_pct = abs(pred_intensity - obs_intensity) / obs_intensity * 100
                else:
                    intensity_error_pct = 100.0 if pred_intensity > 0 else 0.0
                
                # Determinar si estuvo dentro de la incertidumbre esperada
                was_within_uncertainty = position_error_km <= expected_error_km
                
                verification_result = {
                    'track_id': track_id,
                    'prediction_time': pred_time - timedelta(minutes=20),
                    'observation_time': current_window_data['timestamp'],
                    'predicted_lat': pred_lat,
                    'predicted_lon': pred_lon,
                    'observed_lat': obs_lat,
                    'observed_lon': obs_lon,
                    'position_error_km': position_error_km,
                    'intensity_error_pct': intensity_error_pct,
                    'was_within_uncertainty': was_within_uncertainty,
                    'track_disappeared': False,
                    'confidence_level': confidence_level,
                    'forecast_method': forecast_method,
                    'expected_error_km': expected_error_km,
                    'predicted_intensity': pred_intensity,
                    'observed_intensity': obs_intensity
                }
                
                verification_results.append(verification_result)
                
                # Actualizar métricas globales
                self.performance_metrics['total_predictions'] += 1
                if was_within_uncertainty:
                    self.performance_metrics['successful_verifications'] += 1
                
                # Actualizar errores promedio
                total_preds = self.performance_metrics['total_predictions']
                current_mean_pos = self.performance_metrics['mean_position_error_km']
                current_mean_int = self.performance_metrics['mean_intensity_error_pct']
                
                self.performance_metrics['mean_position_error_km'] = (
                    (current_mean_pos * (total_preds - 1) + position_error_km) / total_preds
                )
                self.performance_metrics['mean_intensity_error_pct'] = (
                    (current_mean_int * (total_preds - 1) + intensity_error_pct) / total_preds
                )
                
                logger.info(f"✅ Verificación Track {track_id}: Error={position_error_km:.1f}km, Dentro_incertidumbre={was_within_uncertainty}")
                
            except Exception as e:
                logger.warning(f"⚠️ Error verificando predicción {pred_idx}: {e}")
                continue
        
        return verification_results

    def _calculate_distance_km(self, lat1, lon1, lat2, lon2):
        """Calcula distancia entre dos puntos en km usando aproximación simple."""
        # Aproximación para distancias cortas
        lat_diff = lat2 - lat1
        lon_diff = lon2 - lon1
        
        # Convertir a km (aproximadamente)
        lat_km = lat_diff * 111.0  # 1 grado lat ≈ 111 km
        lon_km = lon_diff * 111.0 * np.cos(np.radians((lat1 + lat2) / 2))  # Corrección por latitud
        
        distance_km = (lat_km**2 + lon_km**2)**0.5
        return distance_km

    # MODIFICAR EL MÉTODO process_time_window PARA INCLUIR VERIFICACIÓN
    def process_time_window(self, start_time, end_time, window_index):
        """Procesa una ventana temporal con verificación de pronósticos."""
        logger.info(f"Procesando ventana {window_index}: {start_time} - {end_time}")
            
        try:
            # 1. Procesar datos GLM
            flash_df = self.glm_processor.process_time_window(start_time, end_time)
            
            if flash_df.empty:
                logger.warning(f"No flash data for window {window_index}")
                return None
                
            logger.info(f"Procesados {len(flash_df)} flashes en ventana {window_index}")
                
            # 2. Identificar celdas
            flash_df_with_clusters, cell_polygons, cell_stats = self.cell_identifier.identify_cells(flash_df)
            cells_gdf = self.cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)
                
            if cells_gdf.empty:
                logger.warning(f"No cells identified for window {window_index}")
                return None
                
            logger.info(f"Identificadas {len(cells_gdf)} celdas en ventana {window_index}")
                
            # 3. Tracking
            tracked_cells = self.tracker.track_cells(cells_gdf, end_time)
            logger.info(f"Tracked {len(tracked_cells)} celdas")
                
            # 4. VERIFICACIÓN: Verificar pronósticos anteriores contra observaciones actuales
            verification_results = []
            if len(self.historical_data) > 0:
                previous_window = self.historical_data[-1]  # Ventana anterior
                previous_predictions = previous_window.get('predictions_df', pd.DataFrame())
                    
                if not previous_predictions.empty:
                    current_window_data = {
                        'timestamp': end_time,
                        'cells_gdf': tracked_cells
                    }
                    verification_results = self.perform_forecast_verification(
                        current_window_data, 
                        previous_predictions.to_dict('records')
                    )
                
            # 5. Nowcasting (generar nuevos pronósticos)
            predictions_df = self.nowcaster.predict_cells(tracked_cells, self.tracker.tracked_cells)
            logger.info(f"Generadas {len(predictions_df)} predicciones")
                
            # 6. Almacenar datos CON verificación
            window_data = {
                'timestamp': end_time,
                'window_index': window_index,
                'flash_df': flash_df_with_clusters,
                'cells_gdf': tracked_cells,
                'predictions_df': predictions_df,
                'verification_results': verification_results,
                'track_stats': self.tracker.get_track_statistics()
            }
                
            self.historical_data.append(window_data)
                
            # 7. Guardar resultados con verificación
            timestamp_str = end_time.strftime('%Y%m%d_%H%M%S')
                
            # Guardar verificaciones
            if verification_results:
                verification_file = os.path.join(self.output_dir, f'verification_{timestamp_str}.csv')
                verification_df = pd.DataFrame(verification_results)
                verification_df.to_csv(verification_file, index=False)
                logger.info(f"Guardadas {len(verification_results)} verificaciones en {verification_file}")
            
            # Guardar celdas tracked
            if not tracked_cells.empty:
                cells_file = os.path.join(self.output_dir, f'tracked_cells_{timestamp_str}.geojson')
                try:
                    tracked_cells.to_file(cells_file, driver='GeoJSON')
                    logger.info(f"Guardadas {len(tracked_cells)} celdas tracked en {cells_file}")
                except Exception as e:
                    logger.warning(f"Error guardando celdas tracked: {e}")
            
            # Guardar predicciones
            if not predictions_df.empty:
                pred_file = os.path.join(self.output_dir, f'predictions_{timestamp_str}.csv')
                try:
                    predictions_df.to_csv(pred_file, index=False)
                    logger.info(f"Guardadas {len(predictions_df)} predicciones en {pred_file}")
                except Exception as e:
                    logger.warning(f"Error guardando predicciones: {e}")
            
            # Guardar datos de flashes con clusters (opcional)
            if not flash_df_with_clusters.empty:
                flash_file = os.path.join(self.output_dir, f'flashes_clustered_{timestamp_str}.csv')
                try:
                    flash_df_with_clusters.to_csv(flash_file, index=False)
                    logger.info(f"Guardados {len(flash_df_with_clusters)} flashes clustered en {flash_file}")
                except Exception as e:
                    logger.warning(f"Error guardando flashes clustered: {e}")
            
            # Guardar estadísticas de tracking
            track_stats = self.tracker.get_track_statistics()
            if track_stats:
                stats_file = os.path.join(self.output_dir, f'track_stats_{timestamp_str}.json')
                try:
                    with open(stats_file, 'w') as f:
                        json.dump(track_stats, f, indent=2, default=str)
                    logger.info(f"Guardadas estadísticas de tracking en {stats_file}")
                except Exception as e:
                    logger.warning(f"Error guardando estadísticas de tracking: {e}")
            
            # Log de resumen de la ventana
            logger.info(f"✅ Ventana {window_index} procesada exitosamente:")
            logger.info(f"   📊 {len(flash_df)} flashes → {len(cells_gdf)} celdas → {len(tracked_cells)} tracked")
            logger.info(f"   🔮 {len(predictions_df)} predicciones generadas")
            logger.info(f"   ✅ {len(verification_results)} verificaciones realizadas")
            
            # Mostrar información de tracks activos
            if not tracked_cells.empty and 'track_id' in tracked_cells.columns:
                unique_tracks = tracked_cells['track_id'].unique()
                logger.info(f"   🎯 Tracks activos: {list(unique_tracks)}")
                
                # Mostrar estadísticas de cada track
                for track_id in unique_tracks:
                    track_cells = tracked_cells[tracked_cells['track_id'] == track_id]
                    if not track_cells.empty:
                        track_cell = track_cells.iloc[0]
                        age = track_cell.get('age_minutes', 0)
                        n_flashes = track_cell.get('n_flashes', 0)
                        area = track_cell.get('area_km2', 0)
                        logger.info(f"      Track {track_id}: {age:.1f}min, {n_flashes}⚡, {area:.1f}km²")
            
            return window_data
                
        except Exception as e:
            logger.error(f"❌ Error procesando ventana {window_index}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _add_complete_performance_panel(self, folium_map, latest_data):
        
        import folium
        """Agrega panel completo de métricas de rendimiento."""
        
        # Calcular estadísticas actuales
        total_preds = self.performance_metrics.get('total_predictions', 0)
        successful_verifs = self.performance_metrics.get('successful_verifications', 0)
        accuracy_rate = (successful_verifs / total_preds * 100) if total_preds > 0 else 0
        
        mean_pos_error = self.performance_metrics.get('mean_position_error_km', 0)
        mean_int_error = self.performance_metrics.get('mean_intensity_error_pct', 0)
        
        # Información del último pronóstico
        latest_predictions = latest_data.get('predictions_df', pd.DataFrame())
        if not latest_predictions.empty:
            if 'confidence_percentage' in latest_predictions.columns:
                avg_confidence = latest_predictions['confidence_percentage'].mean()
            elif 'confidence_level' in latest_predictions.columns:
                avg_confidence = latest_predictions['confidence_level'].mean() * 100
            else:
                avg_confidence = 75.0
            
            n_current_predictions = len(latest_predictions)
            
            if 'forecast_method' in latest_predictions.columns:
                methods_used = latest_predictions['forecast_method'].value_counts().to_dict()
                methods_str = ', '.join([f"{k}({v})" for k, v in methods_used.items()])
            else:
                methods_str = "Linear+Physics"
        else:
            avg_confidence = 0
            n_current_predictions = 0
            methods_str = "Ninguno"
        
        # HTML del panel completo
        panel_html = f'''
        <div style="position: fixed; top: 80px; left: 10px; z-index: 1000; 
                    background-color: rgba(255,255,255,0.95); padding: 15px; 
                    border: 2px solid #333; border-radius: 10px; font-family: Arial;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3); max-width: 300px;">
            
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
                    • <b>Ventanas procesadas:</b> {len(self.historical_data)}<br>
                    • <b>Tiempo total:</b> {len(self.historical_data) * 10} min<br>
                    • <b>Última actualización:</b><br>
                    <div style="margin-left: 10px; font-size: 11px;">{datetime.now().strftime('%H:%M:%S')}</div>
                </div>
            </div>
            
            <div style="text-align: center; font-size: 10px; color: #6c757d; border-top: 1px solid #ddd; padding-top: 5px;">
                Sistema GLM Nowcasting v2.0 - Animado
            </div>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(panel_html))

    def _add_verification_panel(self, folium_map):
        
        import folium
        """Agrega panel con información de verificación de errores."""
        
        # Recopilar todas las verificaciones
        all_verifications = []
        for window_data in self.historical_data:
            all_verifications.extend(window_data.get('verification_results', []))
        
        if not all_verifications:
            return  # No hay verificaciones que mostrar
        
        # Calcular estadísticas de verificación
        position_errors = [v['position_error_km'] for v in all_verifications]
        intensity_errors = [v.get('intensity_error_pct', 0) for v in all_verifications]
        successful_predictions = sum(1 for v in all_verifications if v['was_within_uncertainty'])
        
        verification_html = f'''
        <div style="position: fixed; top: 80px; right: 10px; z-index: 1000; 
                    background-color: rgba(255,255,255,0.95); padding: 15px; 
                    border: 2px solid #333; border-radius: 10px; font-family: Arial;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3); max-width: 280px;">
            
            <h3 style="margin-top: 0; color: #333; text-align: center; border-bottom: 2px solid #ddd; padding-bottom: 5px;">
                ✅ Verificación de Errores
            </h3>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #e8f5e8; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #2e7d32;">📊 Estadísticas de Error</h4>
                <div style="font-size: 12px;">
                    • <b>Verificaciones:</b> {len(all_verifications)}<br>
                    • <b>Predicciones exitosas:</b> {successful_predictions}<br>
                    • <b>Tasa de éxito:</b> <span style="color: {'green' if (successful_predictions/len(all_verifications)*100) > 70 else 'orange'}; font-weight: bold;">{successful_predictions/len(all_verifications)*100:.1f}%</span>
                </div>
            </div>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #fff3e0; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #f57c00;">📏 Errores de Posición</h4>
                <div style="font-size: 12px;">
                    • <b>Error promedio:</b> {np.mean(position_errors):.1f} km<br>
                    • <b>Error mínimo:</b> {np.min(position_errors):.1f} km<br>
                    • <b>Error máximo:</b> {np.max(position_errors):.1f} km<br>
                    • <b>Mediana:</b> {np.median(position_errors):.1f} km
                </div>
            </div>
            
            <div style="margin-bottom: 12px; padding: 8px; background-color: #f3e5f5; border-radius: 5px;">
                <h4 style="margin: 0 0 5px 0; color: #7b1fa2;">⚡ Errores de Intensidad</h4>
                <div style="font-size: 12px;">
                    • <b>Error promedio:</b> {np.mean(intensity_errors):.1f}%<br>
                    • <b>Error mediano:</b> {np.median(intensity_errors):.1f}%<br>
                    • <b>Rango:</b> {np.min(intensity_errors):.1f}%-{np.max(intensity_errors):.1f}%
                </div>
            </div>
            
            <div style="text-align: center; font-size: 10px; color: #6c757d; border-top: 1px solid #ddd; padding-top: 5px;">
                Última verificación: {all_verifications[-1]['observation_time'].strftime('%H:%M') if all_verifications else 'N/A'}
            </div>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(verification_html))

    def _add_uncertainty_legend_to_map(self, folium_map):
        """Agrega leyenda actualizada con líneas verdes."""
        
        import folium
        
        legend_html = '''
        <div style="position: fixed; bottom: 10px; right: 10px; z-index: 1000; 
                    background-color: rgba(255,255,255,0.95); padding: 12px; 
                    border: 2px solid #333; border-radius: 10px; font-family: Arial;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.3);">
            <h4 style="margin-top: 0; text-align: center; color: #333;">🎯 Leyenda</h4>
            
            <div style="margin-bottom: 8px;">
                <h5 style="margin: 5px 0; color: #666;">Tormentas Actuales:</h5>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #0000FF; opacity: 0.4; border: 2px solid #0000FF; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Área de tormenta (azul)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 3px; background-color: #006400; margin-right: 5px; border-style: dashed; border-width: 2px;"></span>
                    <span style="font-size: 11px;">Trayectoria histórica (verde)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: white; border: 2px solid #000000; border-radius: 50%; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Centro actual</span>
                </div>
            </div>
            
            <div style="margin-bottom: 8px;">
                <h5 style="margin: 5px 0; color: #666;">Pronósticos:</h5>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #FF0000; opacity: 0.6; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Probabilidad 60% (rojo)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #FF0000; opacity: 0.4; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Probabilidad 80% (rojo)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #FF0000; opacity: 0.25; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Probabilidad 90% (rojo)</span>
                </div>
                <div style="display: flex; align-items: center; margin-bottom: 3px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background-color: #FF00FF; margin-right: 5px;"></span>
                    <span style="font-size: 11px;">Centro predicción</span>
                </div>
            </div>
            
            <div style="font-size: 10px; color: #666; border-top: 1px solid #ddd; padding-top: 5px;">
                <b>🎮 Controles:</b> Play/Pause, Loop, Velocidad<br>
                <b>🔮 Pronóstico:</b> +20 minutos<br>
                <b>⏱️ Animación:</b> 10 min/frame<br>
                <b>🛤️ Trayectorias:</b> Verde punteado
            </div>
        </div>
        '''
        
        folium_map.get_root().html.add_child(folium.Element(legend_html))

    def _create_uncertainty_circle(self, center_lat, center_lon, radius_km, num_points=36):
        """Crea coordenadas para un círculo de incertidumbre."""
        # Convertir km a grados
        lat_radius = radius_km / 111.0
        lon_radius = radius_km / (111.0 * np.cos(np.radians(center_lat)))
        
        # Generar puntos del círculo
        circle_points = []
        for angle in np.linspace(0, 2*np.pi, num_points):
            lat = center_lat + lat_radius * np.sin(angle)
            lon = center_lon + lon_radius * np.cos(angle)
            circle_points.append([lat, lon])
        
        # Cerrar el polígono
        circle_points.append(circle_points[0])
        
        return circle_points

    def _generate_performance_report(self):
        """Genera un reporte de rendimiento."""
        report_file = os.path.join(self.output_dir, 'consolidated_performance_report.json')
        
        # Recopilar estadísticas
        stats = {
            'total_windows': len(self.historical_data),
            'performance_metrics': self.performance_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        # Guardar reporte
        with open(report_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Reporte de rendimiento guardado en: {report_file}")


    def debug_trajectory_generation(self):
        """Debug específico para ver por qué no aparecen las líneas."""
        logger.info("🔍 DEBUG DETALLADO: Generación de trayectorias")
        
        # Verificar datos básicos
        logger.info(f"Total ventanas históricas: {len(self.historical_data)}")
        
        # Recopilar tracks como lo hace el código principal
        track_history = {}
        for window_idx, window_data in enumerate(self.historical_data):
            timestamp = window_data['timestamp']
            cells_gdf = window_data.get('cells_gdf', pd.DataFrame())
            
            logger.info(f"Ventana {window_idx} ({timestamp.strftime('%H:%M')}): {len(cells_gdf)} celdas")
            
            if not cells_gdf.empty:
                for _, cell in cells_gdf.iterrows():
                    track_id = cell.get('track_id', -1)
                    if track_id != -1:
                        if track_id not in track_history:
                            track_history[track_id] = []
                        
                        track_history[track_id].append({
                            'window_idx': window_idx,
                            'timestamp': timestamp,
                            'lat': cell.get('centroid_lat', 0),
                            'lon': cell.get('centroid_lon', 0),
                            'cell_data': cell
                        })
                        logger.info(f"   Track {track_id}: [{cell.get('centroid_lat', 0):.4f}, {cell.get('centroid_lon', 0):.4f}]")
        
        # Verificar tracks con múltiples posiciones
        logger.info("📊 Análisis de tracks para trayectorias:")
        for track_id, positions in track_history.items():
            positions.sort(key=lambda x: x['timestamp'])
            logger.info(f"   Track {track_id}: {len(positions)} posiciones")
            
            if len(positions) >= 2:
                logger.info(f"      ✅ PUEDE GENERAR TRAYECTORIA")
                for i, pos in enumerate(positions):
                    logger.info(f"         {i+1}. {pos['timestamp'].strftime('%H:%M')} -> [{pos['lat']:.4f}, {pos['lon']:.4f}]")
                
                # Probar crear coordenadas de línea
                test_coords = [[pos['lon'], pos['lat']] for pos in positions]
                logger.info(f"      🧪 Coordenadas de línea: {test_coords}")
            else:
                logger.info(f"      ❌ INSUFICIENTE PARA TRAYECTORIA (necesita ≥2)")
        
        return track_history

def parse_arguments():
    """Parsea argumentos de línea de comandos - ACEPTA ARGUMENTOS REALES."""
    parser = argparse.ArgumentParser(description='Consolidated GLM Nowcasting System - REAL VERSION')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with GLM data')
    parser.add_argument('--start_time', type=str, required=True,
                        help='Start time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_time', type=str, required=True,
                        help='End time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
    # Parámetros del sistema consolidado
    parser.add_argument('--history_minutes', type=int, default=40,
                        help='Minutes of history to maintain (default: 40)')
    parser.add_argument('--min_history_minutes', type=int, default=20,
                        help='Minimum minutes before generating visualizations (default: 20)')
    
    # Parámetros de identificación
    parser.add_argument('--eps', type=float, default=0.01,
                        help='DBSCAN eps parameter')
    parser.add_argument('--min_samples', type=int, default=3,
                        help='DBSCAN min_samples parameter')
    
    # Parámetros de tracking
    parser.add_argument('--max_distance_km', type=float, default=30,
                        help='Maximum tracking distance (km)')
    parser.add_argument('--max_speed_kmh', type=float, default=100,
                        help='Maximum realistic storm speed (km/h)')
    
    # Parámetros de nowcasting
    parser.add_argument('--forecast_minutes', type=int, default=20,
                        help='Forecast lead time (minutes)')
    parser.add_argument('--ensemble_models', action='store_true',
                        help='Use ensemble of models')
    parser.add_argument('--uncertainty', action='store_true',
                        help='Calculate uncertainty estimates')
    
    # Parámetros generales
    parser.add_argument('--window_minutes', type=int, default=10,
                        help='Time window size (minutes)')
    
    parser.add_argument('--debug_visualizations', action='store_true',
                        help='Generate debug visualizations for each window (creates multiple HTMLs)')
    parser.add_argument('--no_intermediate_vis', action='store_true', default=True,
                        help='Do not generate intermediate visualizations (default: True)')


    return parser.parse_args()

def main():
    """Función principal que USA ARGUMENTOS REALES - NO HARDCODED."""
    logger.info("=== SISTEMA CONSOLIDADO DE NOWCASTING GLM - VERSIÓN REAL ===")
    
    args = parse_arguments()
    
    # 🔧 DEBUG: Mostrar argumentos recibidos
    print("=== ARGUMENTOS RECIBIDOS (CONFIRMACIÓN) ===")
    print(f"📁 data_dir: {args.data_dir}")
    print(f"⏰ start_time: {args.start_time}")
    print(f"⏰ end_time: {args.end_time}")
    print(f"📁 output_dir: {args.output_dir}")
    print(f"⏱️ window_minutes: {args.window_minutes}")
    print(f"🔮 forecast_minutes: {args.forecast_minutes}")
    print(f"🎯 uncertainty: {args.uncertainty}")
    print(f"🤖 ensemble_models: {args.ensemble_models}")
    print("=" * 60)
    
    # 1. Importar componentes
    logger.info("Paso 1: Importando componentes...")
    components = import_components()
    
    if components is None:
        logger.error("No se pudieron importar todos los componentes necesarios. Abortando.")
        sys.exit(1)
    
    # 2. Validar argumentos de tiempo - USAR ARGUMENTOS REALES
    logger.info("Paso 2: Validando argumentos...")
    try:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')  # ← USA args.start_time
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M')      # ← USA args.end_time
        
        if end_time <= start_time:
            logger.error("El tiempo de fin debe ser posterior al tiempo de inicio")
            sys.exit(1)
            
        logger.info(f"✅ Período validado: {start_time} a {end_time}")
            
    except ValueError as e:
        logger.error(f"Error parseando tiempos: {e}")
        logger.error("Usa formato: YYYY-MM-DD HH:MM")
        sys.exit(1)
    
    # 3. Crear directorio de salida - USAR ARGUMENTO REAL
    os.makedirs(args.output_dir, exist_ok=True)  # ← USA args.output_dir
    logger.info(f"✅ Directorio de salida: {args.output_dir}")
    
    # 4. Verificar que existen datos GLM
    if not os.path.exists(args.data_dir):
        logger.error(f"❌ Directorio de datos no existe: {args.data_dir}")
        sys.exit(1)
    
    glm_files = [f for f in os.listdir(args.data_dir) if f.endswith('.nc')]
    if not glm_files:
        logger.error(f"❌ No se encontraron archivos GLM (.nc) en: {args.data_dir}")
        sys.exit(1)
    
    logger.info(f"✅ Encontrados {len(glm_files)} archivos GLM en {args.data_dir}")
    
    # 5. Inicializar sistema consolidado - USAR ARGUMENTOS REALES
    logger.info("Paso 3: Inicializando sistema consolidado...")
    consolidated_system = ConsolidatedNowcastingSystem(
        components=components,
        output_dir=args.output_dir,                    # ← USA args.output_dir
        history_minutes=args.history_minutes,          # ← USA args.history_minutes
        min_history_minutes=args.min_history_minutes   # ← USA args.min_history_minutes
    )
    
    if not consolidated_system.initialize_components(args):
        logger.error("Error inicializando componentes del sistema consolidado")
        sys.exit(1)
    
    # 6. Ejecutar análisis consolidado - PASAR ARGUMENTOS REALES
    logger.info("Paso 4: Ejecutando análisis consolidado...")
    final_map_path = consolidated_system.run_consolidated_analysis(start_time, end_time, args)
    
    if final_map_path:
        logger.info(f"=== PROCESO COMPLETADO EXITOSAMENTE ===")
        logger.info(f"🗺️ Mapa consolidado: {final_map_path}")
        logger.info(f"📁 Todos los archivos en: {args.output_dir}")
        
        # Listar archivos generados
        if os.path.exists(args.output_dir):
            files = os.listdir(args.output_dir)
            logger.info(f"📄 Archivos generados: {len(files)}")
            for file in sorted(files)[:5]:  # Mostrar primeros 5
                logger.info(f"   - {file}")
            if len(files) > 5:
                logger.info(f"   ... y {len(files)-5} más")
    else:
        logger.error("Error generando visualización final")
        sys.exit(1)

if __name__ == "__main__":
    main()