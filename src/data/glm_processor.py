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
        Extrae datos de flashes de un conjunto de datos GLM.
        
        Args:
            dataset (xarray.Dataset): Conjunto de datos GLM
            
        Returns:
            pandas.DataFrame: DataFrame con datos de flashes
        """
        if dataset is None:
            return pd.DataFrame()
        
        try:
            # Extraer coordenadas y atributos de flashes
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
            
            # Agregar información de tiempo absoluto
            # CORRECCIÓN: Verificar si los offsets son timedelta o datetime64
            # Si son datetime64, calcular el offset manualmente
            try:
                if isinstance(df['flash_time_offset_of_first_event'].iloc[0], np.datetime64):
                    # Si es datetime64, ya es tiempo absoluto
                    df['time'] = pd.to_datetime(df['flash_time_offset_of_first_event'])
                else:
                    # Es un offset como esperábamos
                    base_time = pd.Timestamp(dataset.product_time.values)
                    df['time'] = base_time + pd.to_timedelta(df['flash_time_offset_of_first_event'], unit='s')
            except (TypeError, ValueError):
                # Plan alternativo: calcular offset manualmente
                # Convertir a segundos (o microsegundos) y luego crear timedeltas
                try:
                    base_time = pd.Timestamp(dataset.product_time.values)
                    
                    # Intenta diferentes enfoques para extraer los segundos
                    if hasattr(dataset, 'product_time') and hasattr(dataset.product_time, 'units'):
                        # Si el atributo tiene unidades definidas, úsalas
                        time_units = dataset.product_time.units
                        
                        # Si los offsets ya son datetimes, calcular diferencia
                        if isinstance(df['flash_time_offset_of_first_event'].iloc[0], (np.datetime64, pd.Timestamp)):
                            df['time'] = pd.to_datetime(df['flash_time_offset_of_first_event'])
                        else:
                            # Convertir offsets numéricos a timedeltas
                            df['time'] = base_time + pd.to_timedelta(df['flash_time_offset_of_first_event'].astype(float), unit='s')
                    else:
                        # Último recurso: usar la hora actual más los offsets si son numéricos
                        if pd.api.types.is_numeric_dtype(df['flash_time_offset_of_first_event']):
                            df['time'] = base_time + pd.to_timedelta(df['flash_time_offset_of_first_event'].astype(float), unit='s')
                        else:
                            # Si todo falla, usar la hora del producto para todos los flashes
                            df['time'] = base_time
                            
                            # Registrar advertencia
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.warning("No se pudo calcular tiempos precisos, usando tiempo de producto para todos los flashes")
                except Exception as e:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.error(f"Error calculando tiempos de flash: {e}")
                    # Último recurso - usar el tiempo del producto para todos
                    df['time'] = pd.Timestamp(dataset.product_time.values)
            
            return df
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
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