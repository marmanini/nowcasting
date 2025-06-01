#!/usr/bin/env python3
import sys
import os

print("🧪 TEST: Iniciando", flush=True)

# Configurar entorno
os.chdir('/home/matias/nowcasting')
sys.path.insert(0, '.')
sys.path.insert(0, 'src')

# Configurar argumentos
sys.argv = [
    'process_historical_data.py',
    '--data_dir', '/home/matias/nowcasting/data/raw',
    '--start_time', '2024-12-23 22:20',
    '--end_time', '2024-12-23 22:25',
    '--debug'
]

print(f"🧪 TEST: Args: {sys.argv}", flush=True)

try:
    print("🧪 TEST: Ejecutando script...", flush=True)
    
    # Importar y ejecutar
    import importlib.util
    spec = importlib.util.spec_from_file_location("process_historical", "scripts/process_historical_data.py")
    module = importlib.util.module_from_spec(spec)
    
    print("🧪 TEST: Módulo cargado", flush=True)
    
    spec.loader.exec_module(module)
    
    print("🧪 TEST: Módulo ejecutado", flush=True)
    
except SystemExit as e:
    print(f"🧪 TEST: SystemExit capturado: {e}", flush=True)
except Exception as e:
    print(f"🧪 TEST: Error: {e}", flush=True)
    import traceback
    traceback.print_exc()

print("🧪 TEST: Completado", flush=True)
