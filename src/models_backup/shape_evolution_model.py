import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
from shapely.geometry import Polygon
from shapely.affinity import scale, rotate
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StormCellEvolutionModel:
    """
    Modelo para predecir la evolución de la forma y características de las celdas de tormenta.
    """
    
    def __init__(self):
        """Inicializa el modelo de evolución."""
        self.area_model = LinearRegression()
        self.intensity_model = LinearRegression()
        self.eccentricity_model = LinearRegression()
        self.rotation_model = LinearRegression()
        self.trained = False
    
    def extract_cell_features(self, cell_gdf):
        """
        Extrae características relevantes de las celdas para el modelado.
        
        Args:
            cell_gdf: GeoDataFrame con datos de celdas
            
        Returns:
            DataFrame con características extraídas
        """
        features = []
        
        for _, cell in cell_gdf.iterrows():
            if isinstance(cell.geometry, Polygon):
                # Características básicas
                area = cell['area_km2']
                n_flashes = cell['n_flashes']
                
                # Calcular excentricidad aproximada (relación entre ejes)
                bounds = cell.geometry.bounds  # (minx, miny, maxx, maxy)
                width = bounds[2] - bounds[0]
                height = bounds[3] - bounds[1]
                eccentricity = max(width, height) / min(width, height) if min(width, height) > 0 else 1
                
                # Calcular orientación principal (ángulo)
                # Esto es una aproximación; para mayor precisión se podría usar PCA
                orientation = np.arctan2(height, width) * 180 / np.pi
                
                # Edad de la celda (si está disponible)
                age_minutes = cell.get('age_minutes', 0)
                
                # Densidad de rayos
                flash_density = n_flashes / area if area > 0 else 0
                
                # Agregar al conjunto de características
                features.append({
                    'cell_id': cell['cell_id'],
                    'track_id': cell.get('track_id', -1),
                    'timestamp': cell.get('timestamp', None),
                    'area_km2': area,
                    'n_flashes': n_flashes,
                    'eccentricity': eccentricity,
                    'orientation_degrees': orientation,
                    'age_minutes': age_minutes,
                    'flash_density': flash_density
                })
        
        return pd.DataFrame(features)
    
    def fit(self, cell_features_by_track):
        """
        Entrena el modelo usando datos históricos de celdas agrupados por track.
        
        Args:
            cell_features_by_track: Dict con DataFrames de características por track_id
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        # Preparar datos para el entrenamiento
        X_area = []
        y_area = []
        X_intensity = []
        y_intensity = []
        X_eccentricity = []
        y_eccentricity = []
        X_rotation = []
        y_rotation = []
        
        for track_id, features_df in cell_features_by_track.items():
            if len(features_df) < 2:
                continue  # Necesitamos al menos 2 puntos para tendencia
                
            # Ordenar por timestamp
            features_df = features_df.sort_values('timestamp')
            
            # Para cada par de registros consecutivos
            for i in range(len(features_df) - 1):
                current = features_df.iloc[i]
                next_state = features_df.iloc[i+1]
                
                # Variables predictoras comunes
                X_common = [
                    current['area_km2'], 
                    current['n_flashes'],
                    current['eccentricity'],
                    current['age_minutes'],
                    current['flash_density']
                ]
                
                # Modelar cambio de área
                X_area.append(X_common)
                y_area.append(next_state['area_km2'])
                
                # Modelar cambio de intensidad
                X_intensity.append(X_common)
                y_intensity.append(next_state['n_flashes'])
                
                # Modelar cambio de excentricidad
                X_eccentricity.append(X_common)
                y_eccentricity.append(next_state['eccentricity'])
                
                # Modelar cambio de orientación
                X_rotation.append(X_common)
                y_rotation.append(next_state['orientation_degrees'])
        
        # Entrenar modelos
        if X_area:
            self.area_model.fit(X_area, y_area)
            self.intensity_model.fit(X_intensity, y_intensity)
            self.eccentricity_model.fit(X_eccentricity, y_eccentricity)
            self.rotation_model.fit(X_rotation, y_rotation)
            self.trained = True
            
            logger.info("Modelos de evolución entrenados")
            return True
        else:
            logger.warning("No hay datos suficientes para entrenar modelos de evolución")
            return False
        
    def predict_evolution(self, current_cell, timesteps=1):
        """
        Predice la evolución de una celda en uno o más pasos de tiempo.
        
        Args:
            current_cell: Diccionario con características actuales de la celda
            timesteps: Número de pasos de tiempo a predecir
            
        Returns:
            Lista de diccionarios con características predichas
        """
        if not self.trained:
            logger.error("El modelo no ha sido entrenado")
            return []
        
        predictions = []
        current_state = current_cell.copy()
        
        for step in range(timesteps):
            # Preparar features para la predicción
            X_features = [
                current_state['area_km2'],
                current_state['n_flashes'],
                current_state['eccentricity'],
                current_state['age_minutes'] + step * 10,  # Asumiendo pasos de 10 minutos
                current_state['flash_density']
            ]
            
            # Predecir próximo estado
            next_area = max(0.1, self.area_model.predict([X_features])[0])
            next_intensity = max(1, round(self.intensity_model.predict([X_features])[0]))
            next_eccentricity = max(1.0, self.eccentricity_model.predict([X_features])[0])
            next_orientation = self.rotation_model.predict([X_features])[0]
            
            # Actualizar densidad para próxima predicción
            next_density = next_intensity / next_area if next_area > 0 else 0
            
            # Crear nuevo estado
            next_state = {
                'step': step + 1,
                'area_km2': next_area,
                'n_flashes': next_intensity,
                'eccentricity': next_eccentricity,
                'orientation_degrees': next_orientation,
                'age_minutes': current_state['age_minutes'] + (step + 1) * 10,
                'flash_density': next_density
            }
            
            predictions.append(next_state)
            current_state = next_state
        
        return predictions
    
    def transform_polygon(self, polygon, pred_evolution):
        """
        Transforma un polígono basado en la evolución predicha.
        
        Args:
            polygon: Geometría Shapely de la celda actual
            pred_evolution: Diccionario con predicción de evolución
            
        Returns:
            Polygon: Geometría transformada
        """
        # Extraer parámetros de transformación
        area_ratio = np.sqrt(pred_evolution['area_km2'] / polygon.area)
        eccentricity_ratio = pred_evolution['eccentricity']
        orientation = pred_evolution['orientation_degrees']
        
        # Aplicar transformaciones
        # 1. Escalar uniformemente basado en la relación de áreas
        scaled_poly = scale(polygon, xfact=area_ratio, yfact=area_ratio)
        
        # 2. Ajustar excentricidad (esto requiere conocer la orientación actual)
        # Por simplicidad, asumimos que el polígono ya tiene alguna orientación
        # y solo ajustamos la forma relativa
        current_bounds = polygon.bounds
        current_width = current_bounds[2] - current_bounds[0]
        current_height = current_bounds[3] - current_bounds[1]
        
        x_factor = np.sqrt(eccentricity_ratio)
        y_factor = 1.0 / np.sqrt(eccentricity_ratio)
        
        # Aplicar transformación de forma
        reshaped_poly = scale(scaled_poly, xfact=x_factor, yfact=y_factor)
        
        # 3. Rotar para ajustar orientación
        # Para esto necesitaríamos la orientación original
        # Esta es una simplificación
        rotated_poly = rotate(reshaped_poly, orientation)
        
        return rotated_poly
    
    def plot_shape_evolution(self, original_polygon, evolution_predictions, save_path=None):
        """
        Visualiza la evolución predicha de la forma de una celda.
        
        Args:
            original_polygon: Geometría Shapely de la celda original
            evolution_predictions: Lista de predicciones de evolución
            save_path: Ruta opcional para guardar la figura
            
        Returns:
            matplotlib.figure.Figure: Figura generada
        """
        # Crear figura
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Graficar polígono original
        x, y = original_polygon.exterior.xy
        ax.fill(x, y, alpha=0.5, fc='blue', ec='black', label='Original')
        
        # Colores para evolución temporal
        colors = ['green', 'orange', 'red', 'purple', 'brown']
        
        # Graficar evoluciones predichas
        for i, pred in enumerate(evolution_predictions):
            # Transformar polígono según predicción
            evolved_poly = self.transform_polygon(original_polygon, pred)
            
            # Graficar
            x, y = evolved_poly.exterior.xy
            color = colors[i % len(colors)]
            label = f"T+{(i+1)*10}min"
            ax.fill(x, y, alpha=0.3, fc=color, ec=color, label=label)
        
        # Configurar gráfico
        ax.set_aspect('equal')
        ax.legend(loc='upper right')
        ax.set_title('Evolución Predicha de Forma de Celda')
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Guardar si se especificó ruta
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figura guardada en {save_path}")
        
        return fig
    
    def prepare_cell_features_by_track(geojson_files):
        """
        Prepara características de celdas agrupadas por track_id para entrenamiento.
        
        Args:
            geojson_files: Lista de rutas a archivos GeoJSON con celdas
            
        Returns:
            dict: Diccionario con DataFrames de características por track_id
        """
        import geopandas as gpd
        from datetime import datetime
        
        # Ordenar archivos
        geojson_files.sort()
        
        # Cargar todos los datos
        all_cells = []
        
        for geojson_file in geojson_files:
            try:
                # Extraer timestamp del nombre del archivo
                filename = os.path.basename(geojson_file)
                time_str = filename.split('_')[-1].split('.')[0]
                
                # Parse timestamp
                year = int(time_str[:4])
                month = int(time_str[4:6])
                day = int(time_str[6:8])
                hour = int(time_str[8:10])
                minute = int(time_str[10:12])
                
                timestamp = datetime(year, month, day, hour, minute)
                
                # Cargar GeoJSON
                gdf = gpd.read_file(geojson_file)
                
                # Añadir timestamp
                gdf['timestamp'] = timestamp
                
                all_cells.append(gdf)
            except Exception as e:
                logger.error(f"Error procesando archivo {geojson_file}: {e}")
        
        if not all_cells:
            logger.error("No se pudieron cargar datos de celdas")
            return {}
        
        # Combinar todos los GeoDataFrames
        combined_cells = pd.concat(all_cells)
        
        # Verificar que existe la columna track_id
        if 'track_id' not in combined_cells.columns:
            logger.error("No se encontró columna track_id en los datos")
            return {}
        
        # Inicializar modelo
        model = StormCellEvolutionModel()
        
        # Extraer características de todas las celdas
        all_features = model.extract_cell_features(combined_cells)
        
        # Agrupar por track_id
        features_by_track = {}
        
        for track_id, group in all_features.groupby('track_id'):
            if len(group) >= 2:  # Solo tracks con al menos 2 puntos
                features_by_track[track_id] = group
        
        logger.info(f"Datos preparados para {len(features_by_track)} tracks")
        
        return features_by_track


    def predict_shape_evolution(geojson_files, output_dir='./shape_evolution'):
        """
        Predice y visualiza la evolución de forma para celdas de tormenta.
        
        Args:
            geojson_files: Lista de rutas a archivos GeoJSON con celdas
            output_dir: Directorio para guardar resultados
            
        Returns:
            dict: Resultados con rutas a visualizaciones generadas
        """
        import geopandas as gpd
        from datetime import datetime
        
        # Asegurar que el directorio de salida existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Preparar características por track
        features_by_track = prepare_cell_features_by_track(geojson_files)
        
        if not features_by_track:
            logger.error("No hay datos suficientes para entrenar el modelo")
            return {}
        
        # Crear y entrenar modelo
        model = StormCellEvolutionModel()
        success = model.fit(features_by_track)
        
        if not success:
            logger.error("No se pudo entrenar el modelo de evolución")
            return {}
        
        # Cargar último conjunto de datos para predicciones
        latest_file = sorted(geojson_files)[-1]
        latest_cells = gpd.read_file(latest_file)
        
        # Verificar datos
        if latest_cells.empty or 'track_id' not in latest_cells.columns:
            logger.error("No hay datos adecuados en el último archivo para generar predicciones")
            return {}
        
        # Extraer características del último conjunto
        latest_features = model.extract_cell_features(latest_cells)
        
        # Generar predicciones y visualizaciones por track
        results = {
            'predictions': {},
            'visualizations': {}
        }
        
        for track_id, track_features in latest_features.groupby('track_id'):
            if len(track_features) == 0:
                continue
                
            # Tomar la última celda del track
            current_cell = track_features.iloc[-1]
            
            # Encontrar el polígono correspondiente
            cell_id = current_cell['cell_id']
            cell_polygon = latest_cells[latest_cells['cell_id'] == cell_id].geometry.iloc[0]
            
            # Predecir evolución (3 pasos de tiempo = 30 minutos)
            evolution_predictions = model.predict_evolution(current_cell, timesteps=3)
            
            # Guardar predicciones
            results['predictions'][track_id] = evolution_predictions
            
            # Generar visualización
            vis_file = os.path.join(output_dir, f"shape_evolution_track{track_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            model.plot_shape_evolution(cell_polygon, evolution_predictions, save_path=vis_file)
            
            # Guardar ruta de visualización
            results['visualizations'][track_id] = vis_file
        
        logger.info(f"Predicciones de evolución generadas para {len(results['predictions'])} tracks")
        
        return results


if __name__ == "__main__":
    import glob
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Predecir evolución de forma de celdas de tormenta.')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--output-dir', type=str, default='shape_evolution', 
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Buscar archivos GeoJSON
    geojson_files = sorted(glob.glob(f"{args.geojson_dir}/cells_*.geojson"))
    
    if not geojson_files:
        print(f"No se encontraron archivos GeoJSON en {args.geojson_dir}")
        exit(1)
        
    print(f"Se encontraron {len(geojson_files)} archivos GeoJSON")
    
    # Generar predicciones de evolución
    results = predict_shape_evolution(geojson_files, output_dir=args.output_dir)
    
    # Mostrar resultados
    if results and 'visualizations' in results:
        print(f"\nPredicciones de evolución generadas para {len(results['visualizations'])} tracks")
        print("Visualizaciones guardadas en:")
        for track_id, vis_file in results['visualizations'].items():
            print(f"- Track #{track_id}: {vis_file}")
    else:
        print("No se pudieron generar predicciones de evolución")