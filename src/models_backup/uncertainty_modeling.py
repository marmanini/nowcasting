import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import norm
import folium
from folium.plugins import TimestampedGeoJson
from sklearn.linear_model import LinearRegression
from datetime import datetime, timedelta
import os
import logging

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UncertaintyModeling:
    """
    Clase para modelar y visualizar la incertidumbre en predicciones de nowcasting.
    """
    
    def __init__(self):
        """Inicializa el modelador de incertidumbre."""
        # Parámetros de modelo de incertidumbre - pueden ser calibrados con datos
        self.pos_error_growth_rate = 2.0  # km por minuto (incremento en error por tiempo)
        self.min_pos_error = 5.0  # km (error mínimo base)
        
        # Factores de penalización para otros aspectos
        self.intensity_penalty = 0.5  # Factor para error adicional por intensidad
        self.speed_penalty = 0.3  # Factor para error adicional por velocidad
        self.complex_shape_penalty = 0.2  # Factor para error adicional por forma compleja
        
        # Coeficientes calibrados para modelo de incertidumbre (a entrenar)
        self.coefficients = {
            'intercept': self.min_pos_error,
            'lead_time': self.pos_error_growth_rate,
            'intensity': 0.05,
            'speed': 0.1,
            'area': 0.02,
            'eccentricity': 1.0
        }
        
        # Modelo entrenado (inicialmente None)
        self.error_model = None
    
    def train_uncertainty_model(self, metrics_df, predictions_df):
        """
        Entrena un modelo de regresión para estimar la incertidumbre en predicciones.
        
        Args:
            metrics_df: DataFrame con métricas de error de predicciones históricas
            predictions_df: DataFrame con datos de predicciones
            
        Returns:
            bool: True si el entrenamiento fue exitoso
        """
        # Fusionar DataFrames para obtener todos los features necesarios
        combined_df = pd.merge(
            metrics_df,
            predictions_df,
            on=['track_id', 'pred_time'],
            how='inner'
        )
        
        # Filtrar para usar solo datos con errores disponibles
        valid_data = combined_df.dropna(subset=['position_error_km'])
        
        if len(valid_data) < 10:
            print("Datos insuficientes para entrenar modelo de incertidumbre")
            return False
        
        # Preparar features para el modelo
        X_features = []
        y_target = []
        
        for _, row in valid_data.iterrows():
            # Tiempo de anticipación (en minutos)
            lead_time = row['lead_time_min']
            
            # Características del sistema de tormentas
            # (Ajustar estos nombres según los disponibles en tu DataFrame)
            intensity = row.get('last_n_flashes', 0)
            area = row.get('last_area', 0)
            
            # Calcular velocidad a partir de componentes si está disponible
            if 'velocity_lon' in row and 'velocity_lat' in row:
                speed = np.sqrt(row['velocity_lon']**2 + row['velocity_lat']**2)
            else:
                speed = 0
            
            # Calcular excentricidad o usar valor por defecto
            eccentricity = row.get('eccentricity', 1.0)
            
            # Crear vector de características
            features = [
                1.0,  # Intercepto
                lead_time,  # Tiempo de anticipación
                intensity,  # Intensidad (número de rayos)
                speed,  # Velocidad de movimiento
                area,  # Área de la celda
                eccentricity  # Forma de la celda
            ]
            
            X_features.append(features)
            y_target.append(row['position_error_km'])
        
        # Entrenar modelo de regresión lineal
        self.error_model = LinearRegression()
        self.error_model.fit(X_features, y_target)
        
        # Actualizar coeficientes
        coeffs = self.error_model.coef_
        self.coefficients = {
            'intercept': self.error_model.intercept_,
            'lead_time': coeffs[1],
            'intensity': coeffs[2],
            'speed': coeffs[3],
            'area': coeffs[4],
            'eccentricity': coeffs[5]
        }
        
        print("Modelo de incertidumbre entrenado. Coeficientes:")
        for name, value in self.coefficients.items():
            print(f"  {name}: {value:.4f}")
        
        return True
    
    def estimate_position_uncertainty(self, prediction, lead_time_min=None):
        """
        Estima la incertidumbre posicional para una predicción.
        
        Args:
            prediction: Diccionario o Series con datos de predicción
            lead_time_min: Tiempo de anticipación en minutos (opcional)
            
        Returns:
            dict: Valores de incertidumbre estimados
        """
        # Si tenemos un modelo entrenado, usarlo
        if self.error_model is not None:
            # Extraer características
            if lead_time_min is None:
                # Calcular a partir de las fechas si están disponibles
                if 'last_time' in prediction and 'pred_time' in prediction:
                    lead_time_min = (prediction['pred_time'] - prediction['last_time']).total_seconds() / 60
                else:
                    lead_time_min = prediction.get('lead_time_min', 30)  # Valor por defecto
            
            # Extraer otras características
            intensity = prediction.get('last_n_flashes', 0)
            area = prediction.get('last_area', 0)
            
            # Calcular velocidad
            if 'velocity_lon' in prediction and 'velocity_lat' in prediction:
                speed = np.sqrt(prediction['velocity_lon']**2 + prediction['velocity_lat']**2)
            else:
                speed = 0
                
            # Excentricidad (forma)
            eccentricity = prediction.get('eccentricity', 1.0)
            
            # Crear vector de características
            features = np.array([
                [1.0, lead_time_min, intensity, speed, area, eccentricity]
            ])
            
            # Predecir error medio esperado
            expected_error = self.error_model.predict(features)[0]
            
            # Estimar otros aspectos de incertidumbre basados en este error base
            position_std = expected_error / 2  # Desviación estándar estimada
            area_uncertainty_pct = 20 + lead_time_min  # % de incertidumbre en área
            intensity_uncertainty_pct = 25 + lead_time_min * 1.5  # % de incertidumbre en intensidad
        else:
            # Modelo simple basado en reglas si no hay modelo entrenado
            if lead_time_min is None:
                # Calcular a partir de las fechas si están disponibles
                if 'last_time' in prediction and 'pred_time' in prediction:
                    lead_time_min = (prediction['pred_time'] - prediction['last_time']).total_seconds() / 60
                else:
                    lead_time_min = prediction.get('lead_time_min', 30)  # Valor por defecto
            
            # Incertidumbre base que crece con el tiempo de anticipación
            expected_error = self.min_pos_error + (self.pos_error_growth_rate * lead_time_min)
            
            # Ajustar por intensidad de la tormenta
            intensity = prediction.get('last_n_flashes', 0)
            if intensity > 100:
                expected_error *= (1 + self.intensity_penalty)
                
            # Ajustar por velocidad de movimiento
            if 'velocity_lon' in prediction and 'velocity_lat' in prediction:
                speed = np.sqrt(prediction['velocity_lon']**2 + prediction['velocity_lat']**2)
                if speed > 25:  # Velocidad alta
                    expected_error *= (1 + self.speed_penalty)
            
            # Valores derivados
            position_std = expected_error / 2
            area_uncertainty_pct = 20 + lead_time_min
            intensity_uncertainty_pct = 25 + lead_time_min * 1.5
        
        # Crear diccionario de resultados
        uncertainty = {
            'expected_position_error_km': expected_error,
            'position_std_km': position_std,
            'position_40ci_km': expected_error * 0.52,  # Intervalo de confianza del 40%
            'position_60ci_km': expected_error * 0.84,  # Intervalo de confianza del 60% 
            'position_80ci_km': expected_error * 1.28,  # Intervalo de confianza del 80%
            'position_90ci_km': expected_error * 1.645, # Intervalo de confianza del 90%
            'area_uncertainty_pct': area_uncertainty_pct,
            'intensity_uncertainty_pct': intensity_uncertainty_pct,
            'lead_time_min': lead_time_min
        }
        
        return uncertainty
    
    def add_uncertainty_to_predictions(self, predictions_df):
        """
        Añade estimaciones de incertidumbre a un DataFrame de predicciones.
        
        Args:
            predictions_df: DataFrame con predicciones
            
        Returns:
            DataFrame: DataFrame original con columnas de incertidumbre añadidas
        """
        # Crear copia para no modificar el original
        result_df = predictions_df.copy()
        
        # Añadir columnas para incertidumbre
        result_df['expected_error_km'] = np.nan
        result_df['error_40ci_km'] = np.nan
        result_df['error_60ci_km'] = np.nan
        result_df['error_80ci_km'] = np.nan
        result_df['error_90ci_km'] = np.nan
        result_df['area_uncertainty_pct'] = np.nan
        result_df['intensity_uncertainty_pct'] = np.nan
        
        # Procesar cada predicción
        for idx, row in result_df.iterrows():
            uncertainty = self.estimate_position_uncertainty(row)
            
            # Añadir valores calculados
            result_df.at[idx, 'expected_error_km'] = uncertainty['expected_position_error_km']
            result_df.at[idx, 'error_40ci_km'] = uncertainty['position_40ci_km']
            result_df.at[idx, 'error_60ci_km'] = uncertainty['position_60ci_km']
            result_df.at[idx, 'error_80ci_km'] = uncertainty['position_80ci_km']
            result_df.at[idx, 'error_90ci_km'] = uncertainty['position_90ci_km']
            result_df.at[idx, 'area_uncertainty_pct'] = uncertainty['area_uncertainty_pct']
            result_df.at[idx, 'intensity_uncertainty_pct'] = uncertainty['intensity_uncertainty_pct']
        
        return result_df
    
    def create_uncertainty_ellipse(self, pred_lat, pred_lon, uncertainty_km, num_points=36):
        """
        Crea coordenadas para una elipse de incertidumbre alrededor de un punto.
        
        Args:
            pred_lat, pred_lon: Coordenadas del centro de la elipse
            uncertainty_km: Radio de incertidumbre en kilómetros
            num_points: Número de puntos para la elipse
            
        Returns:
            list: Lista de coordenadas [lat, lon] para la elipse
        """
        # Convertir km a grados (aproximación)
        # 1 grado de latitud ≈ 111 km
        # 1 grado de longitud ≈ 111 * cos(latitud) km
        lat_radius = uncertainty_km / 111.0
        lon_radius = uncertainty_km / (111.0 * np.cos(np.radians(pred_lat)))
        
        # Generar puntos de la elipse
        ellipse_points = []
        for angle in np.linspace(0, 2*np.pi, num_points):
            lat = pred_lat + lat_radius * np.sin(angle)
            lon = pred_lon + lon_radius * np.cos(angle)
            ellipse_points.append([lat, lon])
        
        # Cerrar el polígono
        ellipse_points.append(ellipse_points[0])
        
        return ellipse_points
    
    def visualize_predictions_with_uncertainty(self, predictions_df, map_center=None, output_file=None):
        """
        Crea un mapa interactivo mostrando predicciones con elipses de incertidumbre.
        
        Args:
            predictions_df: DataFrame con predicciones e incertidumbre
            map_center: [lat, lon] opcionales para centrar el mapa
            output_file: Ruta para guardar el mapa HTML
            
        Returns:
            folium.Map: Mapa interactivo
        """
        # Añadir incertidumbre si no existe en el DataFrame
        if 'expected_error_km' not in predictions_df.columns:
            predictions_df = self.add_uncertainty_to_predictions(predictions_df)
        
        # Determinar centro del mapa
        if map_center is None:
            center_lat = predictions_df['pred_lat'].mean()
            center_lon = predictions_df['pred_lon'].mean()
        else:
            center_lat, center_lon = map_center
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='CartoDB positron'
        )
        
        # Lista de colores para track_ids
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080"
        ]
        
        # Crear FeatureGroups para organizar capas
        pred_group = folium.FeatureGroup(name='Predicciones')
        uncertainty_group = folium.FeatureGroup(name='Zonas de Incertidumbre')
        
        # Añadir datos de predicciones e incertidumbre
        for _, pred in predictions_df.iterrows():
            # Obtener información básica
            track_id = pred['track_id']
            pred_lat = pred['pred_lat']
            pred_lon = pred['pred_lon']
            
            # Información de incertidumbre
            error_km = pred['expected_error_km']
            error_90ci = pred['error_90ci_km']
            
            # Color por track_id
            color = colors[track_id % len(colors)]
            
            # Crear popup con información detallada
            popup_html = f"""
            <div style="font-family: Arial; width: 250px;">
                <h4>Predicción Track #{track_id}</h4>
                <hr style="margin: 5px 0;">
                <b>Posición:</b> [{pred_lat:.4f}, {pred_lon:.4f}]<br>
                <b>Tiempo:</b> {pred['pred_time']}<br>
                <b>Anticipación:</b> {pred.get('lead_time_min', '-')} min<br>
                <hr style="margin: 5px 0;">
                <b>Error esperado:</b> {error_km:.1f} km<br>
                <b>Int. Confianza 90%:</b> {error_90ci:.1f} km<br>
                <b>Incertidumbre área:</b> {pred['area_uncertainty_pct']:.1f}%<br>
                <b>Incertidumbre intensidad:</b> {pred['intensity_uncertainty_pct']:.1f}%
            </div>
            """
            
            # Añadir marcador para posición predicha
            folium.CircleMarker(
                location=[pred_lat, pred_lon],
                radius=6,
                color='white',
                fill=True,
                fill_color=color,
                fill_opacity=0.8,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(pred_group)
            
            # Crear y añadir elipse de incertidumbre (90% IC)
            ellipse_points = self.create_uncertainty_ellipse(
                pred_lat, pred_lon, error_90ci
            )
            
            folium.Polygon(
                locations=ellipse_points,
                color=color,
                weight=1,
                fill=True,
                fill_opacity=0.2,
                fill_color=color,
                popup=f"Zona de 90% confianza - Track #{track_id}"
            ).add_to(uncertainty_group)
        
        # Añadir grupos al mapa
        pred_group.add_to(m)
        uncertainty_group.add_to(m)
        
        # Añadir control de capas
        folium.LayerControl().add_to(m)
        
        # Añadir título
        title_html = f'''
            <h3 align="center" style="font-size:16px">
                <b>Predicciones con Zonas de Incertidumbre</b><br>
                Total: {len(predictions_df)} predicciones
            </h3>
        '''
        folium.Element(title_html).add_to(m)
        
        # Añadir leyenda
        legend_html = '''
        <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; padding: 10px; border: 1px solid grey; border-radius: 5px;">
            <h4 style="margin-top: 0;">Leyenda</h4>
            <div><span style="display: inline-block; width: 12px; height: 12px; border-radius: 50%; background-color: white; border: 2px solid black;"></span> Posición predicha</div>
            <div><span style="display: inline-block; width: 20px; height: 10px; background-color: rgba(255,0,0,0.2); border: 1px solid red;"></span> Zona de incertidumbre (90% IC)</div>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Guardar mapa si se especifica ruta
        if output_file:
            m.save(output_file)
            print(f"Mapa con incertidumbre guardado en {output_file}")
        
        return m
    
    def create_animated_uncertainty_map(self, predictions_by_time, output_file=None):
        """
        Crea un mapa animado con evolución temporal de incertidumbre.
        
        Args:
            predictions_by_time: Dict con DataFrame de predicciones por timestamp
            output_file: Ruta para guardar el mapa HTML
            
        Returns:
            folium.Map: Mapa interactivo
        """
        # Verificar datos
        if not predictions_by_time:
            print("No hay datos para crear mapa animado")
            return None
        
        # Extraer primer conjunto de datos para centrar mapa
        first_preds = list(predictions_by_time.values())[0]
        center_lat = first_preds['pred_lat'].mean()
        center_lon = first_preds['pred_lon'].mean()
        
        # Crear mapa base
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=8,
            tiles='CartoDB positron'
        )
        
        # Lista de colores para track_ids
        colors = [
            "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", 
            "#00FFFF", "#FF8000", "#8000FF", "#0080FF", "#FF0080"
        ]
        
        # Crear features para TimestampedGeoJson
        features = []
        
        # Procesar cada conjunto de predicciones por tiempo
        for timestamp, preds_df in predictions_by_time.items():
            # Calcular incertidumbre si no existe
            if 'expected_error_km' not in preds_df.columns:
                preds_df = self.add_uncertainty_to_predictions(preds_df)
            
            # Formatear timestamp para GeoJSON
            time_str = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
            
            # Procesar cada predicción
            for _, pred in preds_df.iterrows():
                track_id = pred['track_id']
                color = colors[track_id % len(colors)]
                error_90ci = pred['error_90ci_km']
                
                # Crear popup para la predicción
                popup_html = f"""
                <div style="font-family: Arial; width: 230px;">
                    <h4>Predicción Track #{track_id}</h4>
                    <b>Tiempo:</b> {timestamp}<br>
                    <b>Error esperado:</b> {pred['expected_error_km']:.1f} km<br>
                    <b>IC 90%:</b> {error_90ci:.1f} km
                </div>
                """
                
                # Añadir feature para el punto de predicción
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Point',
                        'coordinates': [pred['pred_lon'], pred['pred_lat']]
                    },
                    'properties': {
                        'time': time_str,
                        'icon': 'circle',
                        'popup': popup_html,
                        'iconstyle': {
                            'fillColor': color,
                            'fillOpacity': 0.8,
                            'stroke': True,
                            'color': 'white',
                            'weight': 1,
                            'radius': 6
                        }
                    }
                })
                
                # Crear elipse de incertidumbre
                ellipse_points = self.create_uncertainty_ellipse(
                    pred['pred_lat'], pred['pred_lon'], error_90ci
                )
                
                # Convertir a formato GeoJSON
                ellipse_coords = [[lon, lat] for lat, lon in ellipse_points]
                
                # Añadir feature para la elipse de incertidumbre
                features.append({
                    'type': 'Feature',
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [ellipse_coords]
                    },
                    'properties': {
                        'time': time_str,
                        'style': {
                            'color': color,
                            'weight': 1,
                            'fillColor': color,
                            'fillOpacity': 0.2
                        },
                        'popup': f"Incertidumbre Track #{track_id} - 90% IC"
                    }
                })
        
        # Crear capa TimestampedGeoJson
        time_layer = TimestampedGeoJson(
            {
                'type': 'FeatureCollection',
                'features': features
            },
            period='PT10M',  # Intervalo de tiempo (10 minutos)
            duration='PT1M',  # Duración de la transición (1 minuto)
            auto_play=False,
            loop=False
        )
        
        # Añadir capa al mapa
        time_layer.add_to(m)
        
        # Añadir título
        title_html = f'''
            <h3 align="center" style="font-size:16px">
                <b>Evolución Temporal de Predicciones con Incertidumbre</b>
            </h3>
        '''
        folium.Element(title_html).add_to(m)
        
        # Añadir control de capas
        folium.LayerControl().add_to(m)
        
        # Guardar mapa si se especifica ruta
        if output_file:
            m.save(output_file)
            print(f"Mapa animado guardado en {output_file}")
        
        return m
    
    def plot_uncertainty_calibration(self, calibration_results, save_path=None):
        """
        Visualiza la calibración del modelo de incertidumbre.
        
        Args:
            calibration_results: Diccionario con resultados de calibración
            save_path: Ruta para guardar la figura generada
        """
        if not calibration_results or 'leadtime_calibration' not in calibration_results:
            print("No hay datos suficientes para visualizar calibración")
            return
        
        # Extraer datos de calibración por tiempo
        leadtime_data = calibration_results['leadtime_calibration']
        
        # Preparar datos para graficar
        lead_times = sorted(leadtime_data.keys())
        within_90ci = [leadtime_data[lt]['within_90ci_pct'] for lt in lead_times]
        expected_errors = [leadtime_data[lt]['mean_expected_error'] for lt in lead_times]
        actual_errors = [leadtime_data[lt]['mean_actual_error'] for lt in lead_times]
        
        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # 1. Gráfico de porcentaje dentro del IC 90% vs tiempo de anticipación
        ax1.plot(lead_times, within_90ci, 'o-', color='blue', linewidth=2)
        ax1.axhline(y=90, color='red', linestyle='--', label='Ideal (90%)')
        
        ax1.set_title('Calibración del Intervalo de Confianza del 90%')
        ax1.set_xlabel('Tiempo de Anticipación (min)')
        ax1.set_ylabel('% de Casos Dentro del IC 90%')
        ax1.grid(True, linestyle='--', alpha=0.7)
        ax1.legend()
        
        # 2. Gráfico de error esperado vs error observado
        ax2.plot(lead_times, expected_errors, 'o-', color='green', label='Error Esperado')
        ax2.plot(lead_times, actual_errors, 'o-', color='orange', label='Error Observado')
        
        ax2.set_title('Comparación de Error Esperado vs Observado')
        ax2.set_xlabel('Tiempo de Anticipación (min)')
        ax2.set_ylabel('Error (km)')
        ax2.grid(True, linestyle='--', alpha=0.7)
        ax2.legend()
        
        # Añadir texto con métricas generales
        fig.text(
            0.5, 0.01,
            f"Muestra: {calibration_results['sample_size']} predicciones | "
            f"Global dentro de IC 90%: {calibration_results['within_90ci_pct']:.1f}% | "
            f"RMSE: {calibration_results['uncertainty_rmse_km']:.2f} km | "
            f"Sesgo: {calibration_results['bias_km']:.2f} km",
            ha='center', fontsize=10
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Gráfico de calibración guardado en {save_path}")
        
        plt.show()


def generate_uncertainty_maps(geojson_files, prediction_files, output_dir='./uncertainty_maps'):
    """
    Genera mapas interactivos que muestran la incertidumbre en predicciones.
    
    Args:
        geojson_files: Lista de rutas a archivos GeoJSON con celdas reales
        prediction_files: Lista de rutas a archivos CSV con predicciones
        output_dir: Directorio para guardar mapas generados
        
    Returns:
        dict: Rutas a los mapas generados
    """
    import os
    import pandas as pd
    import geopandas as gpd
    from datetime import datetime
    
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Crear modelador de incertidumbre
    uncertainty_model = UncertaintyModeling()
    
    # Cargar datos
    actual_cells_list = []
    predictions_list = []
    timestamps = []
    
    # Ordenar archivos
    geojson_files.sort()
    prediction_files.sort()
    
    # Cargar celdas reales
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
            timestamps.append(timestamp)
            
            # Cargar GeoJSON
            gdf = gpd.read_file(geojson_file)
            
            # Añadir timestamp
            gdf['timestamp'] = timestamp
            
            actual_cells_list.append(gdf)
        except Exception as e:
            print(f"Error procesando archivo {geojson_file}: {e}")
    
    # Cargar predicciones
    predictions_by_time = {}
    
    for pred_file in prediction_files:
        try:
            # Cargar CSV
            pred_df = pd.read_csv(pred_file)
            
            # Convertir columnas de tiempo a datetime
            if 'last_time' in pred_df.columns:
                pred_df['last_time'] = pd.to_datetime(pred_df['last_time'])
            if 'pred_time' in pred_df.columns:
                pred_df['pred_time'] = pd.to_datetime(pred_df['pred_time'])
            
            # Añadir a la lista
            predictions_list.append(pred_df)
            
            # Agrupar por tiempo de predicción
            for pred_time, group in pred_df.groupby('pred_time'):
                predictions_by_time[pred_time] = group
        except Exception as e:
            print(f"Error procesando archivo {pred_file}: {e}")
    
    # Combinar en un solo DataFrame
    if actual_cells_list:
        all_actual_cells = pd.concat(actual_cells_list)
    else:
        all_actual_cells = pd.DataFrame()
    
    if predictions_list:
        all_predictions = pd.concat(predictions_list)
    else:
        all_predictions = pd.DataFrame()
    
    # Si no hay datos suficientes, salir
    if all_predictions.empty:
        print("No hay datos de predicciones para generar mapas")
        return {}
    
    # Evaluar predicciones para obtener errores reales
    from src.evaluation.nowcasting_evaluator import NowcastingEvaluator
    evaluator = NowcastingEvaluator()
    
    metrics_df = None
    if not all_actual_cells.empty:
        metrics_df, _ = evaluator.calculate_prediction_metrics(
            all_actual_cells, all_predictions
        )
    
    # Entrenar modelo de incertidumbre si hay métricas disponibles
    if metrics_df is not None and not metrics_df.empty:
        uncertainty_model.train_uncertainty_model(metrics_df, all_predictions)
    
    # Añadir estimaciones de incertidumbre a las predicciones
    predictions_with_uncertainty = uncertainty_model.add_uncertainty_to_predictions(all_predictions)
    
    # Generar mapa estático con todas las predicciones
    static_map_file = os.path.join(output_dir, f"uncertainty_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    static_map = uncertainty_model.visualize_predictions_with_uncertainty(
        predictions_with_uncertainty,
        output_file=static_map_file
    )
    
    # Generar mapa animado con evolución temporal
    animated_map_file = os.path.join(output_dir, f"uncertainty_animated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    animated_map = uncertainty_model.create_animated_uncertainty_map(
        predictions_by_time,
        output_file=animated_map_file
    )
    
    # Evaluar calibración del modelo si hay métricas disponibles
    calibration_results = None
    if metrics_df is not None and not metrics_df.empty:
        calibration_results = uncertainty_model.evaluate_uncertainty_calibration(
            metrics_df, predictions_with_uncertainty
        )
        
        # Visualizar calibración
        calibration_plot_file = os.path.join(output_dir, f"uncertainty_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
        uncertainty_model.plot_uncertainty_calibration(
            calibration_results,
            save_path=calibration_plot_file
        )
    
    # Devolver resultados
    results = {
        'static_map_file': static_map_file,
        'animated_map_file': animated_map_file,
        'calibration_results': calibration_results
    }
    
    print(f"Mapas de incertidumbre generados en el directorio {output_dir}")
    return results


if __name__ == "__main__":
    import glob
    import argparse
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Modelar incertidumbre en predicciones de nowcasting.')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--predictions-dir', type=str, default='predictions', 
                        help='Directorio con archivos CSV de predicciones')
    parser.add_argument('--output-dir', type=str, default='uncertainty_maps', 
                        help='Directorio para guardar mapas de incertidumbre')
    
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
    
    # Generar mapas de incertidumbre
    if geojson_files and prediction_files:
        results = generate_uncertainty_maps(
            geojson_files,
            prediction_files,
            output_dir=args.output_dir
        )
        
        # Mostrar resultados
        print("\nMapas generados:")
        for key, value in results.items():
            if isinstance(value, str):
                print(f"- {key}: {value}")
        
        if results.get('calibration_results'):
            print("\nResumen de calibración de incertidumbre:")
            calibration = results['calibration_results']
            print(f"Proporción dentro de IC 90%: {calibration['within_90ci_pct']:.1f}%")
            print(f"Sesgo medio: {calibration['bias_km']:.2f} km")
            print(f"RMSE: {calibration['uncertainty_rmse_km']:.2f} km")
    else:
        print("No hay suficientes datos para generar mapas de incertidumbre")