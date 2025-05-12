# scripts/run_nowcasting.py

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Asegurar que el paquete src esté en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Importar componentes
from src.data.glm_processor import GLMProcessor
from src.models.flash_cell_identification import FlashCellIdentifier
from src.models.flash_cell_tracking import FlashCellTracker
from src.models.flash_cell_nowcasting import FlashCellNowcaster
from src.visualization.maps import LightningVisualizer

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'nowcasting.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('nowcasting')

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(description='Run lightning nowcasting system')
    
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')),
                        help='Directory with GLM data files')
    
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs')),
                        help='Directory to save outputs')
    
    parser.add_argument('--start_time', type=str, default=None,
                        help='Start time in YYYY-MM-DD HH:MM format (default: 30 minutes ago)')
    
    parser.add_argument('--end_time', type=str, default=None,
                        help='End time in YYYY-MM-DD HH:MM format (default: now)')
    
    parser.add_argument('--window_minutes', type=int, default=10,
                        help='Size of time window in minutes (default: 10)')
    
    parser.add_argument('--forecast_minutes', type=int, default=20,
                        help='Minutes into the future to forecast (default: 20)')
    
    parser.add_argument('--eps', type=float, default=0.05,
                        help='DBSCAN eps parameter for clustering (default: 0.05)')
    
    parser.add_argument('--min_samples', type=int, default=5,
                        help='DBSCAN min_samples parameter for clustering (default: 5)')
    
    parser.add_argument('--max_distance_km', type=float, default=30,
                        help='Maximum distance (km) for cell tracking (default: 30)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization maps')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def get_time_range(args):
    """
    Determina el rango de tiempo para el procesamiento.
    
    Args:
        args: Argumentos parseados
        
    Returns:
        tuple: (start_dt, end_dt) - Objetos datetime que representan el inicio y fin del rango
    """
    now = datetime.utcnow()
    
    if args.start_time:
        try:
            start_dt = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')
        except ValueError:
            logger.error(f"Invalid start time format: {args.start_time}")
            start_dt = now - timedelta(minutes=30)
    else:
        start_dt = now - timedelta(minutes=30)
    
    if args.end_time:
        try:
            end_dt = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M')
        except ValueError:
            logger.error(f"Invalid end time format: {args.end_time}")
            end_dt = now
    else:
        end_dt = now
    
    # Asegurar que end_dt es posterior a start_dt
    if end_dt <= start_dt:
        logger.error(f"End time ({end_dt}) must be after start time ({start_dt})")
        end_dt = start_dt + timedelta(minutes=30)
    
    return start_dt, end_dt

def main():
    """
    Función principal del sistema de nowcasting.
    """
    args = parse_arguments()
    
    # Configurar nivel de logging
    if args.debug:
        logger.setLevel(logging.DEBUG)
        # También enviar logs a la consola
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Obtener rango de tiempo
    start_dt, end_dt = get_time_range(args)
    
    logger.info(f"Running nowcasting from {start_dt} to {end_dt}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Inicializar componentes
    glm_processor = GLMProcessor(data_dir=args.data_dir)
    cell_identifier = FlashCellIdentifier(eps=args.eps, min_samples=args.min_samples)
    cell_tracker = FlashCellTracker(max_distance_km=args.max_distance_km)
    cell_nowcaster = FlashCellNowcaster(forecast_minutes=args.forecast_minutes)
    
    # Inicializar visualizador si se requiere
    visualizer = None
    if args.visualize:
        visualizer = LightningVisualizer(output_dir=os.path.join(args.output_dir, 'images'))
    
    # Dividir el rango de tiempo en ventanas
    current_time = start_dt
    window_minutes = args.window_minutes
    
    while current_time < end_dt:
        window_end = min(current_time + timedelta(minutes=window_minutes), end_dt)
        
        logger.info(f"Processing time window: {current_time} to {window_end}")
        
        # Procesar ventana de tiempo
        flash_df = glm_processor.process_time_window(current_time, window_end)
        
        if flash_df.empty:
            logger.warning(f"No flash data found for window {current_time} to {window_end}")
            current_time = window_end
            continue
        
        # Identificar celdas
        flash_df_with_clusters, cell_polygons, cell_stats = cell_identifier.identify_cells(flash_df)
        cells_gdf = cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)
        
        if cells_gdf.empty:
            logger.warning(f"No cells identified for window {current_time} to {window_end}")
            current_time = window_end
            continue
        
        # Realizar seguimiento de celdas
        tracked_cells = cell_tracker.track_cells(cells_gdf, window_end)
        
        # Realizar predicción
        predictions_df = cell_nowcaster.predict_cells(tracked_cells, cell_tracker.tracked_cells)
        predictions_gdf = cell_nowcaster.create_prediction_geometries(predictions_df)
        
        # Guardar resultados
        output_subdir = os.path.join(args.output_dir, 'predictions')
        os.makedirs(output_subdir, exist_ok=True)
        
        timestamp_str = window_end.strftime('%Y%m%d_%H%M%S')
        
        if not cells_gdf.empty:
            cells_gdf.to_file(os.path.join(output_subdir, f'cells_{timestamp_str}.geojson'), driver='GeoJSON')
        
        if not predictions_df.empty:
            predictions_df.to_csv(os.path.join(output_subdir, f'predictions_{timestamp_str}.csv'), index=False)
        
        # Crear visualizaciones
        if args.visualize and visualizer:
            # Mapa interactivo
            m = visualizer.create_interactive_map(
                flash_df=flash_df_with_clusters,
                cells_gdf=tracked_cells,
                predictions_gdf=predictions_gdf,
                start_time=current_time,
                end_time=window_end
            )
            
            map_file = os.path.join('images', f'map_{timestamp_str}.html')
            visualizer.save_interactive_map(m, filename=map_file)
            
            # Mapa de densidad
            density_map = visualizer.create_density_map(flash_df)
            density_file = os.path.join('images', f'density_{timestamp_str}.html')
            visualizer.save_interactive_map(density_map, filename=density_file)
        
        # Avanzar a la siguiente ventana
        current_time = window_end
    
    logger.info("Nowcasting process completed")

if __name__ == "__main__":
    main()