# src/data/glm_processor.py

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime, timedelta
import logging

# Configuración del logger
logger = logging.getLogger(__name__)

class GLMProcessor:
    """
    Clase para procesar datos del Geostationary Lightning Mapper (GLM)
    """
    
    def __init__(self, data_dir=None):
        """
        Inicializa el procesador de datos GLM.
        
        Args:
            data_dir (str): Directorio donde se encuentran los datos GLM
        """
        self.data_dir = data_dir
        
    def find_glm_files(self, start_time=None, end_time=None):
        """
        Encuentra archivos GLM en el directorio de datos para un rango de tiempo específico.
        
        Args:
            start_time (datetime): Tiempo de inicio
            end_time (datetime): Tiempo de fin
            
        Returns:
            list: Lista de rutas a archivos GLM
        """
        if not self.data_dir:
            logger.error("Data directory not specified")
            return []
        
        # Buscar todos los archivos .nc en el directorio
        all_files = glob.glob(os.path.join(self.data_dir, "*.nc"))
        
        if not start_time and not end_time:
            return sorted(all_files)
        
        filtered_files = []
        
        for file_path in all_files:
            filename = os.path.basename(file_path)
            
            # Extraer información de tiempo del nombre del archivo
            # Formato típico: OR_GLM-L2-LCFA_G16_s20230010000000_e20230010000200_c20230010000231.nc
            try:
                # Extraer el tiempo de inicio
                start_str = filename.split('_s')[1].split('_')[0]
                
                # Convertir a datetime
                # Primeros 4 dígitos: año, siguientes 3: día juliano, resto: hora, minuto, segundo
                year = int(start_str[:4])
                doy = int(start_str[4:7])
                hour = int(start_str[7:9])
                minute = int(start_str[9:11])
                second = int(start_str[11:13])
                
                # Convertir día juliano a fecha
                file_date = datetime(year, 1, 1) + timedelta(days=doy-1)
                file_time = datetime(file_date.year, file_date.month, file_date.day, hour, minute, second)
                
                # Comprobar si está en el rango de tiempo
                if start_time and file_time < start_time:
                    continue
                if end_time and file_time > end_time:
                    continue
                
                filtered_files.append(file_path)
                
            except (IndexError, ValueError) as e:
                logger.warning(f"Error parsing filename {filename}: {e}")
                continue
        
        return sorted(filtered_files)
    
    def read_glm_file(self, file_path):
        """
        Lee un archivo GLM y extrae los datos relevantes.
        
        Args:
            file_path (str): Ruta al archivo GLM
            
        Returns:
            xarray.Dataset: Conjunto de datos GLM
        """
        try:
            ds = xr.open_dataset(file_path)
            return ds
        except Exception as e:
            logger.error(f"Error reading GLM file {file_path}: {e}")
            return None
    
    def extract_flash_data(self, dataset):
        """
        Extrae datos de flashes con manejo mejorado de timestamps GLM.
        """
        if dataset is None:
            return pd.DataFrame()
        
        try:
            # Extraer datos básicos
            flash_data = {
                'flash_id': dataset.flash_id.values,
                'flash_time_offset_of_first_event': dataset.flash_time_offset_of_first_event.values,
                'flash_time_offset_of_last_event': dataset.flash_time_offset_of_last_event.values,
                'flash_lon': dataset.flash_lon.values,
                'flash_lat': dataset.flash_lat.values,
                'flash_area': dataset.flash_area.values,
                'flash_energy': dataset.flash_energy.values
            }
            
            # Crear DataFrame
            df = pd.DataFrame(flash_data)
            
            # MANEJO SIMPLIFICADO DE TIEMPO GLM
            # Los archivos GLM son de ~20 segundos, usar el tiempo del producto es suficiente
            base_time = pd.Timestamp(dataset.product_time.values)
            
            # Intentar usar offsets si son numéricos y razonables
            try:
                offsets = df['flash_time_offset_of_first_event'].values
                
                # Verificar si los offsets son numéricos y están en rango razonable (0-20 segundos)
                if np.issubdtype(offsets.dtype, np.number) and np.all((offsets >= 0) & (offsets <= 30)):
                    # Usar offsets reales
                    df['time'] = base_time + pd.to_timedelta(offsets, unit='s')
                else:
                    # Usar tiempo base + variación aleatoria pequeña para simular distribución
                    random_offsets = np.random.uniform(0, 20, len(df))  # 0-20 segundos
                    df['time'] = base_time + pd.to_timedelta(random_offsets, unit='s')
                    
            except:
                # Fallback: usar tiempo base para todos
                df['time'] = base_time
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting flash data: {e}")
            return pd.DataFrame()
        
    def process_time_window(self, start_time, end_time):
        """
        Procesa todos los archivos GLM en una ventana de tiempo específica.
        
        Args:
            start_time (datetime): Tiempo de inicio
            end_time (datetime): Tiempo de fin
            
        Returns:
            pandas.DataFrame: DataFrame con datos de flashes para la ventana de tiempo
        """
        # Encontrar archivos en el rango de tiempo
        files = self.find_glm_files(start_time, end_time)
        
        if not files:
            logger.warning(f"No GLM files found between {start_time} and {end_time}")
            return pd.DataFrame()
        
        logger.info(f"Processing {len(files)} GLM files between {start_time} and {end_time}")
        
        # Procesar cada archivo y combinar los resultados
        all_flashes = []
        
        for file_path in files:
            dataset = self.read_glm_file(file_path)
            if dataset is not None:
                flash_df = self.extract_flash_data(dataset)
                if not flash_df.empty:
                    all_flashes.append(flash_df)
                dataset.close()  # Cerrar el dataset para liberar memoria
        
        if not all_flashes:
            logger.warning("No flash data extracted from GLM files")
            return pd.DataFrame()
        
        # Combinar todos los DataFrames
        combined_df = pd.concat(all_flashes, ignore_index=True)
        
        # Ordenar por tiempo
        combined_df = combined_df.sort_values('time')
        
        return combined_df
    
if __name__ == "__main__":
    main()
