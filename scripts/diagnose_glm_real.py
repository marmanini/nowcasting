#!/usr/bin/env python3
"""
Diagnóstico específico para GLMProcessor real con archivos NetCDF GLM
"""

import os
import sys
import glob
import xarray as xr
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def diagnose_glm_files(data_dir, start_time, end_time):
    """
    Diagnóstico específico para archivos GLM NetCDF
    """
    print("🛰️  DIAGNÓSTICO ESPECÍFICO GLM NOWCASTING")
    print("=" * 60)
    
    # 1. Verificar directorio
    print(f"\n1. VERIFICANDO DIRECTORIO GLM")
    print(f"   📁 Directorio: {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"   ❌ PROBLEMA CRÍTICO: Directorio no existe")
        return False
    
    # 2. Buscar archivos NetCDF
    print(f"\n2. BUSCANDO ARCHIVOS GLM (.nc)")
    nc_files = glob.glob(os.path.join(data_dir, "*.nc"))
    
    # También buscar en subdirectorios
    nc_files_recursive = glob.glob(os.path.join(data_dir, "**", "*.nc"), recursive=True)
    all_nc_files = list(set(nc_files + nc_files_recursive))
    
    print(f"   📊 Total archivos .nc encontrados: {len(all_nc_files)}")
    
    if len(all_nc_files) == 0:
        print(f"   ❌ PROBLEMA CRÍTICO: No hay archivos .nc")
        print(f"   💡 Los datos GLM deben estar en formato NetCDF (.nc)")
        print(f"   💡 Verificar que la descarga se completó correctamente")
        return False
    
    # 3. Analizar nombres de archivos GLM
    print(f"\n3. ANALIZANDO NOMENCLATURA GLM")
    
    glm_pattern = r'OR_GLM-L2-LCFA_G\d+_s(\d{14})_e(\d{14})_c(\d{14})\.nc'
    valid_glm_files = []
    invalid_files = []
    
    for file_path in all_nc_files[:10]:  # Analizar primeros 10
        filename = os.path.basename(file_path)
        match = re.match(glm_pattern, filename)
        
        if match:
            start_str, end_str, creation_str = match.groups()
            valid_glm_files.append((file_path, filename, start_str, end_str))
            print(f"   ✅ {filename}")
            print(f"      🕒 Inicio: {parse_glm_time(start_str)}")
            print(f"      🕒 Fin: {parse_glm_time(end_str)}")
        else:
            invalid_files.append((file_path, filename))
            print(f"   ⚠️  {filename} (formato no estándar)")
    
    if len(all_nc_files) > 10:
        remaining = len(all_nc_files) - 10
        remaining_valid = sum(1 for f in all_nc_files[10:] if re.match(glm_pattern, os.path.basename(f)))
        print(f"   📄 ... y {remaining} archivos más ({remaining_valid} válidos estimados)")
    
    # 4. Verificar rango temporal
    print(f"\n4. VERIFICANDO COBERTURA TEMPORAL")
    print(f"   🎯 Rango solicitado: {start_time} a {end_time}")
    
    files_in_range = []
    all_glm_files = [f for f in all_nc_files if re.match(glm_pattern, os.path.basename(f))]
    
    for file_path in all_glm_files:
        filename = os.path.basename(file_path)
        match = re.match(glm_pattern, filename)
        if match:
            file_start = parse_glm_time(match.group(1))
            file_end = parse_glm_time(match.group(2))
            
            if file_start <= end_time and file_end >= start_time:
                files_in_range.append((file_path, file_start, file_end))
    
    print(f"   📊 Archivos en rango temporal: {len(files_in_range)}")
    
    if len(files_in_range) == 0:
        print(f"   ❌ PROBLEMA: No hay archivos GLM en el rango temporal")
        if all_glm_files:
            # Mostrar el rango de fechas disponible
            first_file = min(all_glm_files, key=lambda f: parse_glm_time(re.match(glm_pattern, os.path.basename(f)).group(1)))
            last_file = max(all_glm_files, key=lambda f: parse_glm_time(re.match(glm_pattern, os.path.basename(f)).group(2)))
            
            first_time = parse_glm_time(re.match(glm_pattern, os.path.basename(first_file)).group(1))
            last_time = parse_glm_time(re.match(glm_pattern, os.path.basename(last_file)).group(2))
            
            print(f"   📅 Rango disponible: {first_time} a {last_time}")
            print(f"   💡 SOLUCIÓN: Ajustar fechas de procesamiento")
        return False
    
    # 5. Probar lectura de archivos
    print(f"\n5. PROBANDO LECTURA DE ARCHIVOS GLM")
    
    test_files = files_in_range[:3] if files_in_range else all_glm_files[:3]
    successful_reads = 0
    total_flashes = 0
    
    for i, file_info in enumerate(test_files):
        if isinstance(file_info, tuple):
            file_path = file_info[0]
        else:
            file_path = file_info
            
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path) / (1024*1024)
        
        print(f"   🔍 Probando: {filename} ({file_size:.1f} MB)")
        
        try:
            # Intentar abrir con xarray
            ds = xr.open_dataset(file_path)
            
            print(f"      ✅ Archivo abierto exitosamente")
            print(f"      📊 Variables: {len(ds.data_vars)} encontradas")
            
            # Verificar variables críticas GLM
            critical_vars = ['flash_id', 'flash_lon', 'flash_lat', 'flash_energy']
            missing_vars = []
            
            for var in critical_vars:
                if var in ds.variables:
                    n_flashes = len(ds[var])
                    print(f"      ✅ {var}: {n_flashes} registros")
                    if var == 'flash_id':
                        total_flashes += n_flashes
                else:
                    missing_vars.append(var)
                    print(f"      ❌ {var}: NO ENCONTRADA")
            
            if missing_vars:
                print(f"      ⚠️  Variables faltantes: {missing_vars}")
                print(f"      📋 Variables disponibles: {list(ds.variables.keys())}")
            
            # Verificar dimensiones
            print(f"      📐 Dimensiones: {dict(ds.dims)}")
            
            # Verificar tiempo del producto
            if 'product_time' in ds.variables:
                product_time = pd.Timestamp(ds.product_time.values)
                print(f"      🕒 Tiempo del producto: {product_time}")
            
            ds.close()
            successful_reads += 1
            
        except Exception as e:
            print(f"      ❌ Error: {str(e)[:100]}...")
    
    print(f"\n   📊 RESUMEN DE LECTURA:")
    print(f"   ✅ Archivos legibles: {successful_reads}/{len(test_files)}")
    print(f"   ⚡ Total flashes encontrados: {total_flashes}")
    
    # 6. Simular GLMProcessor
    print(f"\n6. SIMULANDO GLMProcessor.process_time_window()")
    
    if files_in_range:
        print(f"   🔄 Simulando procesamiento de ventana temporal...")
        
        try:
            # Importar y usar el GLMProcessor real
            sys.path.append('src/data')
            from glm_processor import GLMProcessor
            
            processor = GLMProcessor(data_dir=data_dir)
            flash_df = processor.process_time_window(start_time, end_time)
            
            print(f"   📊 RESULTADO GLMProcessor:")
            print(f"   🎯 DataFrame resultante: {len(flash_df)} registros")
            
            if not flash_df.empty:
                print(f"   ✅ ÉXITO: GLMProcessor devolvió datos")
                print(f"   📋 Columnas: {list(flash_df.columns)}")
                
                if 'flash_lat' in flash_df.columns and 'flash_lon' in flash_df.columns:
                    print(f"   📍 Rango lat: {flash_df['flash_lat'].min():.3f} a {flash_df['flash_lat'].max():.3f}")
                    print(f"   📍 Rango lon: {flash_df['flash_lon'].min():.3f} a {flash_df['flash_lon'].max():.3f}")
                
                if 'time' in flash_df.columns:
                    print(f"   🕒 Rango temporal: {flash_df['time'].min()} a {flash_df['time'].max()}")
                
                return True
            else:
                print(f"   ❌ PROBLEMA: GLMProcessor devolvió DataFrame vacío")
                print(f"   💡 Posibles causas:")
                print(f"      - Archivos no contienen datos para la ventana solicitada")
                print(f"      - Error en extracción de datos")
                print(f"      - Problema con cálculo de timestamps")
                return False
                
        except ImportError as e:
            print(f"   ❌ Error importando GLMProcessor: {e}")
            return False
        except Exception as e:
            print(f"   ❌ Error ejecutando GLMProcessor: {e}")
            import traceback
            print(f"   📋 Traceback: {traceback.format_exc()}")
            return False
    else:
        print(f"   ⚠️  No hay archivos en rango para probar")
        return False

def parse_glm_time(time_str):
    """
    Convierte string de tiempo GLM a datetime
    Formato: YYYYDDDHHMISS (donde DDD es día juliano)
    """
    try:
        year = int(time_str[:4])
        doy = int(time_str[4:7])  # Día del año (juliano)
        hour = int(time_str[7:9])
        minute = int(time_str[9:11])
        second = int(time_str[11:13])
        
        # Convertir día juliano a fecha
        base_date = datetime(year, 1, 1) + timedelta(days=doy-1)
        return datetime(base_date.year, base_date.month, base_date.day, hour, minute, second)
    except:
        return None

def suggest_fixes(data_dir, start_time, end_time):
    """
    Sugiere soluciones para problemas encontrados
    """
    print(f"\n🔧 SUGERENCIAS DE SOLUCIÓN")
    print("=" * 60)
    
    # Buscar archivos y analizar problemas
    nc_files = glob.glob(os.path.join(data_dir, "**", "*.nc"), recursive=True)
    
    if not nc_files:
        print(f"1. ❌ NO HAY ARCHIVOS .nc")
        print(f"   💡 Verificar descarga de datos GLM")
        print(f"   💡 Los datos deben estar en formato NetCDF")
        print(f"   💡 Comando de descarga: verificar si se completó")
        return
    
    glm_pattern = r'OR_GLM-L2-LCFA_G\d+_s(\d{14})_e(\d{14})_c(\d{14})\.nc'
    valid_files = [f for f in nc_files if re.match(glm_pattern, os.path.basename(f))]
    
    if not valid_files:
        print(f"2. ❌ ARCHIVOS NO SIGUEN FORMATO GLM ESTÁNDAR")
        print(f"   📄 Archivos encontrados:")
        for f in nc_files[:5]:
            print(f"      - {os.path.basename(f)}")
        print(f"   💡 Formato esperado: OR_GLM-L2-LCFA_G16_s20230010000000_e20230010000200_c20230010000231.nc")
        return
    
    # Verificar cobertura temporal
    file_times = []
    for f in valid_files:
        match = re.match(glm_pattern, os.path.basename(f))
        if match:
            file_start = parse_glm_time(match.group(1))
            file_end = parse_glm_time(match.group(2))
            if file_start and file_end:
                file_times.append((file_start, file_end))
    
    if file_times:
        earliest = min(t[0] for t in file_times)
        latest = max(t[1] for t in file_times)
        
        print(f"3. 📅 COBERTURA TEMPORAL DISPONIBLE")
        print(f"   🕒 Más temprano: {earliest}")
        print(f"   🕒 Más tardío: {latest}")
        print(f"   🎯 Solicitado: {start_time} a {end_time}")
        
        if start_time < earliest or end_time > latest:
            print(f"   💡 SOLUCIÓN: Ajustar rango temporal:")
            suggested_start = max(start_time, earliest)
            suggested_end = min(end_time, latest)
            print(f"      Rango sugerido: {suggested_start} a {suggested_end}")
            
            print(f"\n   🚀 COMANDO SUGERIDO:")
            print(f"   bash run_historical_analysis.sh \"{suggested_start.strftime('%Y-%m-%d %H:%M')}\" \"{suggested_end.strftime('%Y-%m-%d %H:%M')}\" --visualize --debug")

def main():
    if len(sys.argv) != 4:
        print("Uso: python diagnose_glm_real.py DATA_DIR START_TIME END_TIME")
        print("Ejemplo: python diagnose_glm_real.py /home/matias/nowcasting/data/raw '2023-01-01 12:00' '2023-01-01 13:00'")
        sys.exit(1)
    
    data_dir = sys.argv[1]
    start_time_str = sys.argv[2]
    end_time_str = sys.argv[3]
    
    try:
        start_time = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
        end_time = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
    except ValueError as e:
        print(f"Error en formato de tiempo: {e}")
        sys.exit(1)
    
    # Ejecutar diagnóstico
    success = diagnose_glm_files(data_dir, start_time, end_time)
    
    if not success:
        suggest_fixes(data_dir, start_time, end_time)
        print(f"\n🚨 DIAGNÓSTICO FALLÓ")
        sys.exit(1)
    else:
        print(f"\n✅ DIAGNÓSTICO EXITOSO")
        print(f"   El sistema GLM está listo para procesar datos")

if __name__ == "__main__":
    main()