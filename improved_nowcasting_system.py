#!/usr/bin/env python3
"""
Script de integración para usar los componentes mejorados de tracking y nowcasting.
VERSIÓN COMPLETAMENTE CORREGIDA - Maneja todas las importaciones robustamente.
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Agregar el directorio actual y src al path para importaciones locales
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('improved_nowcasting')

def import_components():
    """
    Importa todos los componentes necesarios de manera robusta.
    """
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
    
    # 2. Importar componentes mejorados (con múltiples intentos)
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
        except ImportError:
            continue
        except AttributeError:
            continue
    
    if ImprovedFlashCellTracker is None:
        logger.error("✗ No se pudo importar ImprovedFlashCellTracker")
        logger.error("Verifica que improved_flash_cell_tracking.py esté disponible en:")
        logger.error("- ./improved_flash_cell_tracking.py")
        logger.error("- ./src/models/improved_flash_cell_tracking.py")
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
        except ImportError:
            continue
        except AttributeError:
            continue
    
    if ImprovedFlashCellNowcaster is None:
        logger.error("✗ No se pudo importar ImprovedFlashCellNowcaster")
        logger.error("Verifica que improved_flash_cell_nowcasting.py esté disponible en:")
        logger.error("- ./improved_flash_cell_nowcasting.py")
        logger.error("- ./src/models/improved_flash_cell_nowcasting.py")
        return None
    
    components['ImprovedFlashCellNowcaster'] = ImprovedFlashCellNowcaster
    
    # 3. Importar visualizador (opcional)
    logger.info("Importando visualizador (opcional)...")
    LightningVisualizer = None
    viz_locations = [
        ('src.visualization.maps', 'desde src/visualization/'),
        ('maps', 'desde directorio actual')
    ]
    
    for module_path, description in viz_locations:
        try:
            module = __import__(module_path, fromlist=['LightningVisualizer'])
            LightningVisualizer = getattr(module, 'LightningVisualizer')
            logger.info(f"✓ LightningVisualizer importado {description}")
            break
        except ImportError:
            continue
        except AttributeError:
            continue
    
    if LightningVisualizer is None:
        logger.warning("⚠ LightningVisualizer no disponible - visualizaciones deshabilitadas")
        logger.warning("Para habilitar visualizaciones, asegúrate de tener maps.py con clase LightningVisualizer")
    
    components['LightningVisualizer'] = LightningVisualizer
    
    logger.info("Todos los componentes necesarios importados correctamente")
    return components

def run_improved_analysis(components, data_dir, start_time, end_time, output_dir, args):
    """
    Ejecuta análisis con componentes mejorados.
    """
    logger.info("Iniciando análisis con componentes mejorados")
    
    # Verificar que el directorio de datos existe
    if not os.path.exists(data_dir):
        logger.error(f"Directorio de datos no existe: {data_dir}")
        return
    
    # Crear directorio de salida
    os.makedirs(output_dir, exist_ok=True)
    
    # Inicializar componentes
    try:
        glm_processor = components['GLMProcessor'](data_dir=data_dir)
        logger.info("GLMProcessor inicializado")
    except Exception as e:
        logger.error(f"Error inicializando GLMProcessor: {e}")
        return
    
    cell_identifier = components['FlashCellIdentifier'](
        eps=getattr(args, 'eps', 0.01),
        min_samples=getattr(args, 'min_samples', 3)
    )
    logger.info("FlashCellIdentifier inicializado")
    
    # Usar tracker mejorado
    improved_tracker = components['ImprovedFlashCellTracker'](
        max_distance_km=getattr(args, 'max_distance_km', 30),
        max_speed_kmh=getattr(args, 'max_speed_kmh', 100),
        intensity_weight=0.3,
        size_weight=0.2,
        prediction_weight=0.4
    )
    logger.info("ImprovedFlashCellTracker inicializado")
    
    # Usar nowcaster mejorado
    improved_nowcaster = components['ImprovedFlashCellNowcaster'](
        forecast_minutes=getattr(args, 'forecast_minutes', 20),
        ensemble_models=getattr(args, 'ensemble_models', True),
        uncertainty_quantification=getattr(args, 'uncertainty', True)
    )
    logger.info("ImprovedFlashCellNowcaster inicializado")
    
    # Inicializar visualizador
    visualizer = None
    if getattr(args, 'visualize', False) and components['LightningVisualizer']:
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        try:
            visualizer = components['LightningVisualizer'](output_dir=images_dir)
            logger.info("LightningVisualizer inicializado")
        except Exception as e:
            logger.warning(f"No se pudo inicializar visualizador: {e}")
            logger.warning("Continuando sin visualizaciones")
    
    # Procesar ventanas temporales
    current_time = start_time
    window_minutes = getattr(args, 'window_minutes', 10)
    
    all_results = []
    all_predictions = []
    window_index = 0
    
    logger.info(f"Procesando desde {start_time} hasta {end_time}")
    logger.info(f"Ventana temporal: {window_minutes} minutos")
    
    while current_time < end_time:
        window_end = min(current_time + timedelta(minutes=window_minutes), end_time)
        
        logger.info(f"Procesando ventana {window_index}: {current_time} - {window_end}")
        
        try:
            # 1. Procesar datos GLM
            flash_df = glm_processor.process_time_window(current_time, window_end)
            
            if flash_df.empty:
                logger.warning(f"No flash data for window {window_index}")
                current_time = window_end
                window_index += 1
                continue
            
            logger.info(f"Procesados {len(flash_df)} flashes")
            
            # 2. Identificar celdas
            flash_df_with_clusters, cell_polygons, cell_stats = cell_identifier.identify_cells(flash_df)
            cells_gdf = cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)
            
            if cells_gdf.empty:
                logger.warning(f"No cells identified for window {window_index}")
                current_time = window_end
                window_index += 1
                continue
            
            logger.info(f"Identificadas {len(cells_gdf)} celdas")
            
            # 3. Tracking mejorado
            tracked_cells = improved_tracker.track_cells(cells_gdf, window_end)
            logger.info(f"Tracked {len(tracked_cells)} celdas")
            
            # 4. Nowcasting mejorado
            predictions_df = improved_nowcaster.predict_cells(tracked_cells, improved_tracker.tracked_cells)
            logger.info(f"Generadas {len(predictions_df)} predicciones")
            
            # 5. Crear geometrías de predicción
            predictions_gdf = None
            if not predictions_df.empty:
                try:
                    predictions_gdf = improved_nowcaster.create_prediction_geometries(predictions_df)
                except Exception as e:
                    logger.warning(f"Error creando geometrías de predicción: {e}")
            
            # 6. Guardar resultados
            timestamp_str = window_end.strftime('%Y%m%d_%H%M%S')
            
            # Guardar celdas tracked
            if not tracked_cells.empty:
                cells_file = os.path.join(output_dir, f'tracked_cells_{timestamp_str}.geojson')
                try:
                    tracked_cells.to_file(cells_file, driver='GeoJSON')
                    logger.info(f"Guardadas {len(tracked_cells)} celdas tracked en {cells_file}")
                except Exception as e:
                    logger.warning(f"Error guardando celdas tracked: {e}")
            
            # Guardar predicciones
            if not predictions_df.empty:
                pred_file = os.path.join(output_dir, f'predictions_{timestamp_str}.csv')
                try:
                    predictions_df.to_csv(pred_file, index=False)
                    logger.info(f"Guardadas {len(predictions_df)} predicciones en {pred_file}")
                except Exception as e:
                    logger.warning(f"Error guardando predicciones: {e}")
                
                # Guardar geometrías de predicción
                if predictions_gdf is not None and not predictions_gdf.empty:
                    pred_geom_file = os.path.join(output_dir, f'prediction_geometries_{timestamp_str}.geojson')
                    try:
                        predictions_gdf.to_file(pred_geom_file, driver='GeoJSON')
                    except Exception as e:
                        logger.warning(f"Error guardando geometrías de predicción: {e}")
            
            # 7. Crear visualizaciones si se solicita
            if visualizer and predictions_gdf is not None:
                try:
                    # Mapa principal con tracking y predicciones
                    m = visualizer.create_interactive_map(
                        flash_df=flash_df_with_clusters,
                        cells_gdf=tracked_cells,
                        predictions_gdf=predictions_gdf,
                        start_time=current_time,
                        end_time=window_end,
                        show_uncertainty=getattr(args, 'uncertainty', False)
                    )
                    
                    map_file = f'improved_map_{timestamp_str}.html'
                    visualizer.save_interactive_map(m, filename=map_file)
                    logger.info(f"Mapa guardado: {map_file}")
                except Exception as e:
                    logger.warning(f"Error creando visualización: {e}")
            
            # 8. Recopilar estadísticas
            window_stats = {
                'window_index': window_index,
                'timestamp': window_end,
                'n_flashes': len(flash_df),
                'n_cells': len(cells_gdf),
                'n_tracked': len(tracked_cells),
                'n_predictions': len(predictions_df),
                'active_tracks': len(improved_tracker.tracked_cells)
            }
            
            # Agregar estadísticas de tracking
            try:
                track_stats = improved_tracker.get_track_statistics()
                window_stats.update(track_stats)
            except Exception as e:
                logger.warning(f"Error obteniendo estadísticas de tracking: {e}")
            
            all_results.append(window_stats)
            all_predictions.append(predictions_df)
            
            # Imprimir progreso
            logger.info(f"Ventana {window_index}: {len(flash_df)} rayos, {len(cells_gdf)} celdas, "
                       f"{len(tracked_cells)} tracked, {len(predictions_df)} predicciones")
            
        except Exception as e:
            logger.error(f"Error procesando ventana {window_index}: {e}")
            import traceback
            traceback.print_exc()
        
        # Avanzar ventana
        current_time = window_end
        window_index += 1
    
    # Generar reporte final
    if all_results:
        generate_performance_report(all_results, all_predictions, output_dir)
    
    logger.info("Análisis completado")

def generate_performance_report(all_results, all_predictions, output_dir):
    """
    Genera un reporte de rendimiento detallado.
    """
    logger.info("Generando reporte de rendimiento")
    
    # Crear DataFrame con estadísticas
    stats_df = pd.DataFrame(all_results)
    
    # Calcular métricas generales
    total_flashes = stats_df['n_flashes'].sum()
    total_cells = stats_df['n_cells'].sum()
    total_predictions = stats_df['n_predictions'].sum()
    avg_track_length = stats_df['avg_track_length'].mean() if 'avg_track_length' in stats_df.columns else 0
    
    # Análisis de predicciones
    all_pred_dfs = [df for df in all_predictions if not df.empty]
    prediction_analysis = {}
    
    if all_pred_dfs:
        combined_preds = pd.concat(all_pred_dfs, ignore_index=True)
        
        prediction_analysis = {
            'total_predictions': len(combined_preds),
            'avg_forecast_distance': 0,
            'methods_used': combined_preds['forecast_method'].value_counts().to_dict() if 'forecast_method' in combined_preds.columns else {},
            'uncertainty_available': 'uncertainty_lat' in combined_preds.columns
        }
        
        # Estadísticas de incertidumbre si están disponibles
        if 'confidence_level' in combined_preds.columns:
            prediction_analysis['avg_confidence'] = combined_preds['confidence_level'].mean()
            prediction_analysis['min_confidence'] = combined_preds['confidence_level'].min()
    
    # Escribir reporte
    report_file = os.path.join(output_dir, 'performance_report.txt')
    
    with open(report_file, 'w') as f:
        f.write("REPORTE DE RENDIMIENTO - SISTEMA MEJORADO\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("ESTADÍSTICAS GENERALES:\n")
        f.write(f"  Total de rayos procesados: {total_flashes:,}\n")
        f.write(f"  Total de celdas identificadas: {total_cells:,}\n")
        f.write(f"  Total de predicciones generadas: {total_predictions:,}\n")
        f.write(f"  Longitud promedio de tracks: {avg_track_length:.2f}\n")
        f.write(f"  Ventanas procesadas: {len(all_results)}\n\n")
        
        f.write("ANÁLISIS DE TRACKING:\n")
        if not stats_df.empty:
            f.write(f"  Tracks totales creados: {stats_df['total_tracks'].max() if 'total_tracks' in stats_df.columns else 'N/A'}\n")
            f.write(f"  Tracks activos promedio: {stats_df['active_tracks'].mean():.1f}\n")
            f.write(f"  Tracks con predicciones: {stats_df['tracks_with_predictions'].sum() if 'tracks_with_predictions' in stats_df.columns else 'N/A'}\n")
        
        f.write("\nANÁLISIS DE PREDICCIONES:\n")
        for key, value in prediction_analysis.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\nESTADÍSTICAS POR VENTANA:\n")
        f.write(stats_df.to_string(index=False))
    
    # También guardar como CSV
    csv_file = os.path.join(output_dir, 'performance_report.csv')
    stats_df.to_csv(csv_file, index=False)
    
    logger.info(f"Reporte guardado en: {report_file}")
    logger.info(f"Datos CSV guardados en: {csv_file}")

def parse_arguments():
    """
    Parsea argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description='Improved GLM Nowcasting System')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory with GLM data')
    parser.add_argument('--start_time', type=str, required=True,
                        help='Start time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--end_time', type=str, required=True,
                        help='End time (YYYY-MM-DD HH:MM)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory')
    
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
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualizations')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with original methods')
    parser.add_argument('--monitoring', action='store_true',
                        help='Enable monitoring mode')
    
    return parser.parse_args()

def main():
    """
    Función principal.
    """
    logger.info("=== SISTEMA MEJORADO DE NOWCASTING GLM ===")
    
    args = parse_arguments()
    
    # 1. Importar todos los componentes
    logger.info("Paso 1: Importando componentes...")
    components = import_components()
    
    if components is None:
        logger.error("No se pudieron importar todos los componentes necesarios. Abortando.")
        sys.exit(1)
    
    # 2. Validar argumentos de tiempo
    logger.info("Paso 2: Validando argumentos...")
    try:
        start_time = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M')
        
        if end_time <= start_time:
            logger.error("El tiempo de fin debe ser posterior al tiempo de inicio")
            sys.exit(1)
            
    except ValueError as e:
        logger.error(f"Error parseando tiempos: {e}")
        logger.error("Usa formato: YYYY-MM-DD HH:MM")
        sys.exit(1)
    
    # 3. Crear directorio de salida
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 4. Ejecutar análisis
    logger.info("Paso 3: Ejecutando análisis...")
    run_improved_analysis(
        components, args.data_dir, start_time, end_time, args.output_dir, args
    )
    
    logger.info("=== PROCESO COMPLETADO EXITOSAMENTE ===")

if __name__ == "__main__":
    main()