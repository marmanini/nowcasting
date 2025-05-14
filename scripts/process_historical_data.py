#!/usr/bin/env python3
# scripts/process_historical_data.py
#
# Script para procesar datos históricos del GLM y realizar nowcasting
# Autor: Matias
# Fecha: Mayo 2025

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
    filename=os.path.join(log_dir, 'historical_nowcasting.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('historical_nowcasting')

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(description='Process historical GLM data and perform nowcasting')
    
    parser.add_argument('--data_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')),
                        help='Directory with GLM data files')
    
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'historical')),
                        help='Directory to save outputs')
    
    parser.add_argument('--start_time', type=str, required=True,
                        help='Start time in YYYY-MM-DD HH:MM format')
    
    parser.add_argument('--end_time', type=str, required=True,
                        help='End time in YYYY-MM-DD HH:MM format')
    
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
    
    parser.add_argument('--validation', action='store_true',
                        help='Enable validation mode (compare predictions with actual observations)')
    
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization maps')
    
    parser.add_argument('--animation', action='store_true',
                        help='Create animation of the event evolution')
    
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
    try:
        start_dt = datetime.strptime(args.start_time, '%Y-%m-%d %H:%M')
    except ValueError:
        logger.error(f"Invalid start time format: {args.start_time}")
        sys.exit(1)
    
    try:
        end_dt = datetime.strptime(args.end_time, '%Y-%m-%d %H:%M')
    except ValueError:
        logger.error(f"Invalid end time format: {args.end_time}")
        sys.exit(1)
    
    # Asegurar que end_dt es posterior a start_dt
    if end_dt <= start_dt:
        logger.error(f"End time ({end_dt}) must be after start time ({start_dt})")
        sys.exit(1)
    
    return start_dt, end_dt

def create_animation(output_dir, file_pattern='map_*.html', output_name='event_animation'):
    """
    Crea una animación a partir de mapas HTML.
    
    Args:
        output_dir: Directorio donde se encuentran los mapas
        file_pattern: Patrón para los archivos HTML
        output_name: Nombre del archivo de salida
    
    Returns:
        str: Ruta al archivo de animación
    """
    try:
        import imageio
        import glob
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time
        
        # Configurar opciones de Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        # Iniciar navegador
        browser = webdriver.Chrome(options=chrome_options)
        
        # Obtener archivos HTML ordenados por tiempo
        html_files = sorted(glob.glob(os.path.join(output_dir, file_pattern)))
        
        if not html_files:
            logger.warning("No HTML files found for animation")
            return None
        
        logger.info(f"Creating animation from {len(html_files)} HTML files")
        
        # Crear directorio temporal para imágenes
        temp_dir = os.path.join(output_dir, 'temp_frames')
        os.makedirs(temp_dir, exist_ok=True)
        
        image_files = []
        
        # Capturar imagen de cada HTML
        for i, html_file in enumerate(html_files):
            file_url = 'file://' + os.path.abspath(html_file)
            browser.get(file_url)
            
            # Esperar a que la página cargue
            time.sleep(2)
            
            # Capturar imagen
            image_file = os.path.join(temp_dir, f'frame_{i:04d}.png')
            browser.save_screenshot(image_file)
            image_files.append(image_file)
            
            logger.info(f"Captured frame {i+1}/{len(html_files)}")
        
        # Cerrar navegador
        browser.quit()
        
        # Crear GIF
        output_gif = os.path.join(output_dir, f'{output_name}.gif')
        
        with imageio.get_writer(output_gif, mode='I', duration=0.5) as writer:
            for image_file in image_files:
                image = imageio.imread(image_file)
                writer.append_data(image)
        
        logger.info(f"Animation created: {output_gif}")
        
        return output_gif
        
    except ImportError:
        logger.error("Animation creation requires additional packages: imageio, selenium")
        logger.error("Install with: pip install imageio selenium")
        return None
    except Exception as e:
        logger.error(f"Error creating animation: {e}")
        return None

def main():
    """
    Función principal para procesar datos históricos del GLM.
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
    
    # Crear directorio de salida específico para este evento
    event_name = f"event_{start_dt.strftime('%Y%m%d_%H%M')}_{end_dt.strftime('%H%M')}"
    output_dir = os.path.join(args.output_dir, event_name)
    os.makedirs(output_dir, exist_ok=True)
    
    if args.visualize:
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
    
    predictions_dir = os.path.join(output_dir, 'predictions')
    os.makedirs(predictions_dir, exist_ok=True)
    
    logger.info(f"Processing historical data from {start_dt} to {end_dt}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Inicializar componentes
    glm_processor = GLMProcessor(data_dir=args.data_dir)
    cell_identifier = FlashCellIdentifier(eps=args.eps, min_samples=args.min_samples)
    cell_tracker = FlashCellTracker(max_distance_km=args.max_distance_km)
    cell_nowcaster = FlashCellNowcaster(forecast_minutes=args.forecast_minutes)
    
    # Inicializar visualizador si se requiere
    visualizer = None
    if args.visualize:
        visualizer = LightningVisualizer(output_dir=images_dir)
    
    # Dividir el rango de tiempo en ventanas
    current_time = start_dt
    window_minutes = args.window_minutes
    
    # Almacenar resultados
    all_windows = []
    all_flash_dfs = []
    all_cell_gdfs = []
    all_tracked_gdfs = []
    all_predictions = []
    
    # Procesar cada ventana temporal
    window_index = 0
    
    while current_time < end_dt:
        window_end = min(current_time + timedelta(minutes=window_minutes), end_dt)
        
        logger.info(f"Processing time window {window_index}: {current_time} to {window_end}")
        
        # Procesar ventana de tiempo
        flash_df = glm_processor.process_time_window(current_time, window_end)
        
        if flash_df.empty:
            logger.warning(f"No flash data found for window {current_time} to {window_end}")
            current_time = window_end
            window_index += 1
            continue
        
        # Identificar celdas
        flash_df_with_clusters, cell_polygons, cell_stats = cell_identifier.identify_cells(flash_df)
        cells_gdf = cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)
        
        if cells_gdf.empty:
            logger.warning(f"No cells identified for window {current_time} to {window_end}")
            current_time = window_end
            window_index += 1
            continue
        
        # Realizar seguimiento de celdas
        tracked_cells = cell_tracker.track_cells(cells_gdf, window_end)
        
        # Realizar predicción
        predictions_df = cell_nowcaster.predict_cells(tracked_cells, cell_tracker.tracked_cells)
        predictions_gdf = cell_nowcaster.create_prediction_geometries(predictions_df)
        
        # Guardar resultados
        timestamp_str = window_end.strftime('%Y%m%d_%H%M%S')
        
        if not cells_gdf.empty:
            cells_file = os.path.join(predictions_dir, f'cells_{timestamp_str}.geojson')
            cells_gdf.to_file(cells_file, driver='GeoJSON')
        
        if not predictions_df.empty:
            predictions_file = os.path.join(predictions_dir, f'predictions_{timestamp_str}.csv')
            predictions_df.to_csv(predictions_file, index=False)
        
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
            
            map_file = f'map_{timestamp_str}.html'
            visualizer.save_interactive_map(m, filename=map_file)
            
            # Mapa de densidad
            density_map = visualizer.create_density_map(flash_df)
            density_file = f'density_{timestamp_str}.html'
            visualizer.save_interactive_map(density_map, filename=density_file)
        
        # Almacenar datos para análisis posterior
        all_windows.append((current_time, window_end))
        all_flash_dfs.append(flash_df)
        all_cell_gdfs.append(cells_gdf)
        all_tracked_gdfs.append(tracked_cells)
        all_predictions.append(predictions_df)
        
        # Avanzar a la siguiente ventana
        current_time = window_end
        window_index += 1
    
    # Realizar validación si se solicita
    if args.validation and all_predictions and len(all_predictions) > 1:
        logger.info("Performing validation of predictions...")
        
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        
        # Comparar predicciones con observaciones reales
        validation_results = []
        
        for i in range(len(all_predictions) - 1):
            pred_df = all_predictions[i]
            
            if pred_df.empty:
                continue
                
            # Ventana de tiempo para validación
            val_window_start = all_windows[i+1][0]
            val_window_end = all_windows[i+1][1]
            
            # Obtener observaciones reales (próxima ventana)
            next_cells_gdf = all_tracked_gdfs[i+1]
            
            for _, pred in pred_df.iterrows():
                track_id = pred['track_id']
                pred_time = pred['pred_time']
                pred_lon = pred['pred_lon']
                pred_lat = pred['pred_lat']
                
                # Buscar el mismo track en la siguiente ventana
                if track_id in next_cells_gdf['track_id'].values:
                    actual_cell = next_cells_gdf[next_cells_gdf['track_id'] == track_id].iloc[0]
                    actual_lon = actual_cell['centroid_lon']
                    actual_lat = actual_cell['centroid_lat']
                    
                    # Calcular error
                    from math import sqrt
                    distance_error = sqrt((pred_lon - actual_lon)**2 + (pred_lat - actual_lat)**2) * 111  # km aproximados
                    
                    validation_results.append({
                        'track_id': track_id,
                        'prediction_window': i,
                        'pred_time': pred_time,
                        'pred_lon': pred_lon,
                        'pred_lat': pred_lat,
                        'actual_lon': actual_lon,
                        'actual_lat': actual_lat,
                        'distance_error_km': distance_error
                    })
        
        if validation_results:
            # Crear DataFrame con resultados
            validation_df = pd.DataFrame(validation_results)
            
            # Guardar resultados
            validation_file = os.path.join(output_dir, 'validation_results.csv')
            validation_df.to_csv(validation_file, index=False)
            
            # Calcular estadísticas
            mean_error = validation_df['distance_error_km'].mean()
            median_error = validation_df['distance_error_km'].median()
            max_error = validation_df['distance_error_km'].max()
            
            logger.info(f"Validation results:")
            logger.info(f"  Number of validated predictions: {len(validation_df)}")
            logger.info(f"  Mean distance error: {mean_error:.2f} km")
            logger.info(f"  Median distance error: {median_error:.2f} km")
            logger.info(f"  Maximum distance error: {max_error:.2f} km")
            
            # Graficar distribución de errores si se visualiza
            if args.visualize:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 6))
                plt.hist(validation_df['distance_error_km'], bins=20, alpha=0.7)
                plt.axvline(mean_error, color='r', linestyle='--', label=f'Media: {mean_error:.2f} km')
                plt.axvline(median_error, color='g', linestyle='-.', label=f'Mediana: {median_error:.2f} km')
                plt.xlabel('Error de distancia (km)')
                plt.ylabel('Frecuencia')
                plt.title('Distribución de errores de predicción')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                error_plot_file = os.path.join(images_dir, 'error_distribution.png')
                plt.savefig(error_plot_file)
                plt.close()
                
                # Graficar error vs tiempo
                plt.figure(figsize=(12, 6))
                for track_id in validation_df['track_id'].unique():
                    track_data = validation_df[validation_df['track_id'] == track_id]
                    if len(track_data) > 1:
                        plt.plot(track_data['prediction_window'], track_data['distance_error_km'], 'o-', label=f'Track {track_id}')
                
                plt.xlabel('Ventana de predicción')
                plt.ylabel('Error de distancia (km)')
                plt.title('Evolución del error de predicción por track')
                if len(validation_df['track_id'].unique()) <= 10:
                    plt.legend()
                plt.grid(True, alpha=0.3)
                
                error_evolution_file = os.path.join(images_dir, 'error_evolution.png')
                plt.savefig(error_evolution_file)
                plt.close()
        else:
            logger.warning("No validation data available")
    
    # Crear animación si se solicita
    if args.animation and args.visualize:
        logger.info("Creating animation of event evolution...")
        animation_file = create_animation(images_dir)
        if animation_file:
            logger.info(f"Animation created: {animation_file}")
    
    logger.info("Historical data processing completed")
    
    # Generar informe resumen
    report_file = os.path.join(output_dir, 'event_summary.txt')
    
    with open(report_file, 'w') as f:
        f.write(f"GLM Lightning Nowcasting - Event Summary\n")
        f.write(f"=======================================\n\n")
        f.write(f"Event Period: {start_dt} to {end_dt}\n")
        f.write(f"Processing Parameters:\n")
        f.write(f"  - Time Window: {window_minutes} minutes\n")
        f.write(f"  - Forecast Time: {args.forecast_minutes} minutes\n")
        f.write(f"  - Clustering: eps={args.eps}, min_samples={args.min_samples}\n")
        f.write(f"  - Max Tracking Distance: {args.max_distance_km} km\n\n")
        
        f.write(f"Summary Statistics:\n")
        f.write(f"  - Windows Processed: {window_index}\n")
        
        total_flashes = sum(len(df) for df in all_flash_dfs if not df.empty)
        f.write(f"  - Total Flashes: {total_flashes}\n")
        
        unique_cells = set()
        for gdf in all_cell_gdfs:
            if not gdf.empty:
                unique_cells.update(gdf['cell_id'].unique())
        f.write(f"  - Unique Cells: {len(unique_cells)}\n")
        
        unique_tracks = set()
        for gdf in all_tracked_gdfs:
            if not gdf.empty:
                unique_tracks.update(gdf['track_id'].unique())
        f.write(f"  - Unique Tracks: {len(unique_tracks)}\n\n")
        
        # Incluir estadísticas de validación si disponibles
        if args.validation and 'validation_df' in locals() and not validation_df.empty:
            f.write(f"Validation Results:\n")
            f.write(f"  - Validated Predictions: {len(validation_df)}\n")
            f.write(f"  - Mean Distance Error: {mean_error:.2f} km\n")
            f.write(f"  - Median Distance Error: {median_error:.2f} km\n")
            f.write(f"  - Maximum Distance Error: {max_error:.2f} km\n")
    
    logger.info(f"Event summary report created: {report_file}")
    print(f"\nProcessing completed. Results saved to: {output_dir}")

    def save_clusters_csv(flash_df, output_dir, window_end):
        """
        Guarda un CSV con la información de clusters para diagnóstico.
        
        Args:
            flash_df (pandas.DataFrame): DataFrame con datos de flashes incluyendo columna 'cluster'
            output_dir (str): Directorio de salida
            window_end (datetime): Tiempo de fin de la ventana
        """
        if 'cluster' not in flash_df.columns:
            logger.warning("No cluster column in flash DataFrame, cannot save clusters CSV")
            return
        
        # Crear directorio si no existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Nombre de archivo
        timestamp_str = window_end.strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(output_dir, f'clusters_{timestamp_str}.csv')
        
        # Contar flashes por cluster
        cluster_counts = flash_df.groupby('cluster').size().reset_index(name='count')
        
        # Agregar coordenadas promedio
        cluster_coords = flash_df.groupby('cluster').agg({
            'flash_lon': 'mean',
            'flash_lat': 'mean',
            'flash_energy': 'sum'
        }).reset_index()
        
        # Combinar datos
        cluster_info = pd.merge(cluster_counts, cluster_coords, on='cluster')
        
        # Guardar CSV
        cluster_info.to_csv(file_path, index=False)
        logger.info(f"Saved clusters CSV to {file_path}")
        
        # También imprimir para diagnóstico inmediato
        print(f"Clusters identified ({len(cluster_info)} total):")
        print(cluster_info.sort_values('count', ascending=False).head(10))
        
        return file_path

    # Luego, dentro de la función main() de process_historical_data.py,
    # Después de identificar celdas, añadir:

    # Identificar celdas
    flash_df_with_clusters, cell_polygons, cell_stats = cell_identifier.identify_cells(flash_df)
    cells_gdf = cell_identifier.create_cell_geodataframe(cell_polygons, cell_stats)

    # Diagnóstico: guardar información de clusters
    save_clusters_csv(flash_df_with_clusters, os.path.join(output_dir, 'diagnostics'), window_end)

if __name__ == "__main__":
    main()