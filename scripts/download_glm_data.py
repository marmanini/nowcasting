# scripts/download_glm_data.py

import os
import sys
import argparse
import subprocess
import logging
from datetime import datetime, timedelta
import time
import re

# Asegurar que el paquete src esté en el path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuración de logging
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'logs'))
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(log_dir, 'glm_downloader.log'),
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('glm_downloader')

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos.
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(description='Download GLM data from AWS')
    
    parser.add_argument('--date', type=str, default=None,
                        help='Date in YYYYMMDD format (default: today)')
    
    parser.add_argument('--hour', type=str, default=None,
                        help='Hour in HH format (default: current hour)')
    
    parser.add_argument('--minute', type=str, default=None,
                        help='Minute or minute range in MM format or MM-MM format (default: last 10 minutes)')
    
    parser.add_argument('--bucket', type=str, default='noaa-goes16',
                        help='AWS S3 bucket name (default: noaa-goes16)')
    
    parser.add_argument('--prefix', type=str, default='GLM-L2-LCFA',
                        help='S3 prefix for GLM data (default: GLM-L2-LCFA)')
    
    parser.add_argument('--output_dir', type=str, 
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')),
                        help='Directory to save downloaded files')
    
    parser.add_argument('--aws_profile', type=str, default=None,
                        help='AWS profile to use (optional)')
    
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    
    return parser.parse_args()

def get_time_range(args):
    """
    Determina el rango de tiempo para la descarga basado en los argumentos.
    
    Args:
        args: Argumentos parseados
        
    Returns:
        tuple: (start_dt, end_dt) - Objetos datetime que representan el inicio y fin del rango
    """
    now = datetime.utcnow()
    
    # Determinar fecha
    if args.date:
        try:
            year = int(args.date[:4])
            month = int(args.date[4:6])
            day = int(args.date[6:8])
            date = datetime(year, month, day)
        except (ValueError, IndexError):
            logger.error(f"Invalid date format: {args.date}. Using current date.")
            date = datetime(now.year, now.month, now.day)
    else:
        date = datetime(now.year, now.month, now.day)
    
    # Determinar hora
    if args.hour is not None:
        try:
            hour = int(args.hour)
            if hour < 0 or hour > 23:
                raise ValueError("Hour must be between 0 and 23")
        except ValueError:
            logger.error(f"Invalid hour format: {args.hour}. Using current hour.")
            hour = now.hour
    else:
        hour = now.hour
    
    # Determinar minutos
    if args.minute is not None:
        try:
            # Verificar si es un rango (formato MM-MM)
            if '-' in args.minute:
                start_min, end_min = map(int, args.minute.split('-'))
                if start_min < 0 or start_min > 59 or end_min < 0 or end_min > 59:
                    raise ValueError("Minutes must be between 0 and 59")
            else:
                # Minuto único
                start_min = int(args.minute)
                end_min = start_min
                if start_min < 0 or start_min > 59:
                    raise ValueError("Minute must be between 0 and 59")
        except ValueError as e:
            logger.error(f"Invalid minute format: {args.minute}. Using last 10 minutes. Error: {e}")
            end_min = (now.minute // 10) * 10  # Redondear al decaminuto anterior
            start_min = end_min - 10
            if start_min < 0:
                start_min = 0
    else:
        # Por defecto, usar los últimos 10 minutos redondeados
        end_min = (now.minute // 10) * 10  # Redondear al decaminuto anterior
        start_min = end_min - 10
        if start_min < 0:
            start_min = 50  # Ir a la hora anterior
            hour = hour - 1
            if hour < 0:
                hour = 23
                date = date - timedelta(days=1)
    
    # Crear objetos datetime para inicio y fin
    try:
        start_dt = datetime(date.year, date.month, date.day, hour, start_min)
        end_dt = datetime(date.year, date.month, date.day, hour, end_min)
        
        # Si el minuto final es menor que el inicial, asumir que cruza la hora
        if end_min < start_min:
            end_dt = end_dt + timedelta(hours=1)
        
        # Asegurar que no estamos solicitando datos del futuro
        if end_dt > now:
            end_dt = now
            logger.warning(f"Adjusted end time to current time: {end_dt}")
        
    except ValueError as e:
        logger.error(f"Error creating datetime objects: {e}")
        # Fallback a los últimos 10 minutos desde ahora
        end_dt = now
        start_dt = now - timedelta(minutes=10)
    
    return start_dt, end_dt

def get_s3_keys(bucket, prefix, start_dt, end_dt, aws_profile=None):
    """
    Obtiene las claves S3 para los archivos GLM en el rango de tiempo especificado.
    
    Args:
        bucket (str): Nombre del bucket S3
        prefix (str): Prefijo para los archivos GLM
        start_dt (datetime): Tiempo de inicio
        end_dt (datetime): Tiempo de fin
        aws_profile (str, optional): Perfil AWS a usar
        
    Returns:
        list: Lista de claves S3 para los archivos GLM
    """
    # Construir el comando AWS CLI base
    cmd_base = ['aws', 's3', 'ls', f's3://{bucket}/{prefix}/']
    
    # Agregar perfil AWS si se especifica
    if aws_profile:
        cmd_base.extend(['--profile', aws_profile])
    
    # Convertir fechas a formato de día juliano para construir prefijos
    date_prefixes = []
    
    current_dt = start_dt
    while current_dt <= end_dt:
        year = current_dt.year
        doy = current_dt.strftime('%j')  # Día del año, formato julian (001-366)
        hour = current_dt.hour
        
        # Formato: {año}/{día_juliano}/{hora}
        date_prefix = f'{year}/{doy}/{hour:02d}/'
        date_prefixes.append(date_prefix)
        
        # Avanzar una hora
        current_dt += timedelta(hours=1)
    
    # Eliminar duplicados
    date_prefixes = list(set(date_prefixes))
    
    all_keys = []
    
    for date_prefix in date_prefixes:
        full_prefix = f'{prefix}/{date_prefix}'
        cmd = cmd_base.copy()
        cmd[-1] = f's3://{bucket}/{full_prefix}'
        
        try:
            logger.info(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Extraer nombres de archivos
            lines = output.strip().split('\n')
            for line in lines:
                if line.strip():
                    # Formato típico: 2023-01-01 12:34:56 1234 GLM-L2-LCFA/...
                    parts = line.split()
                    if len(parts) >= 4:
                        file_key = full_prefix + ' '.join(parts[3:])
                        
                        # Verificar si el archivo está en el rango de tiempo
                        # Formato típico: OR_GLM-L2-LCFA_G16_s20230010000000_e20230010000200_c20230010000231.nc
                        match = re.search(r'_s(\d{14})_e(\d{14})_', file_key)
                        if match:
                            start_str, end_str = match.groups()
                            
                            # Convertir a datetime
                            # Formato: YYYYDDDHHMMSS (año, día juliano, hora, minuto, segundo)
                            file_start_year = int(start_str[:4])
                            file_start_doy = int(start_str[4:7])
                            file_start_hour = int(start_str[7:9])
                            file_start_min = int(start_str[9:11])
                            
                            # Convertir día juliano a fecha
                            file_start_date = datetime(file_start_year, 1, 1) + timedelta(days=int(file_start_doy) - 1)
                            file_start = datetime(
                                file_start_date.year, 
                                file_start_date.month, 
                                file_start_date.day,
                                file_start_hour,
                                file_start_min
                            )
                            
                            # Solo agregar si el archivo está en el rango
                            if start_dt <= file_start <= end_dt:
                                all_keys.append(file_key)
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error listing S3 objects: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
    
    return all_keys

def download_s3_files(bucket, keys, output_dir, aws_profile=None):
    """
    Descarga archivos desde S3.
    
    Args:
        bucket (str): Nombre del bucket S3
        keys (list): Lista de claves S3 a descargar
        output_dir (str): Directorio de destino
        aws_profile (str, optional): Perfil AWS a usar
        
    Returns:
        int: Número de archivos descargados
    """
    if not keys:
        logger.info("No files to download")
        return 0
    
    # Asegurar que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    downloaded = 0
    
    for key in keys:
        # Extraer nombre de archivo
        filename = os.path.basename(key)
        output_path = os.path.join(output_dir, filename)
        
        # Verificar si el archivo ya existe
        if os.path.exists(output_path):
            logger.info(f"File already exists: {output_path}")
            continue
        
        # Construir el comando AWS CLI
        cmd = ['aws', 's3', 'cp', f's3://{bucket}/{key}', output_path]
        
        # Agregar perfil AWS si se especifica
        if aws_profile:
            cmd.extend(['--profile', aws_profile])
        
        try:
            logger.info(f"Downloading: {key} to {output_path}")
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"Download successful: {output_path}")
            downloaded += 1
        
        except subprocess.CalledProcessError as e:
            logger.error(f"Error downloading file {key}: {e}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
    
    return downloaded

def main():
    """
    Función principal para descargar datos GLM de AWS.
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
    
    logger.info(f"Downloading GLM data from {start_dt} to {end_dt}")
    logger.info(f"Bucket: {args.bucket}")
    logger.info(f"Prefix: {args.prefix}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Obtener claves S3
    keys = get_s3_keys(args.bucket, args.prefix, start_dt, end_dt, args.aws_profile)
    
    logger.info(f"Found {len(keys)} GLM files to download")
    
    # Descargar archivos
    downloaded = download_s3_files(args.bucket, keys, args.output_dir, args.aws_profile)
    
    logger.info(f"Downloaded {downloaded} GLM files")
    
    # También imprimir en la consola
    print(f"Downloaded {downloaded} GLM files from {start_dt} to {end_dt}")

if __name__ == "__main__":
    main()