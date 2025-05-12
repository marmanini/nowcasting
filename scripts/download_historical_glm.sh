# scripts/download_historical_glm.sh
#!/bin/bash

# Script de ejemplo para descargar datos GLM históricos

# Definir parámetros de tiempo
DATE="20241224"  # 1 de enero de 2023
HOUR="12"        # 12 UTC
MINUTE="00-10"   # Minutos 00 a 30

# Ejecutar el script de descarga con parámetros específicos
python3 download_glm_data.py --date ${DATE} --hour ${HOUR} --minute ${MINUTE} --debug

# Para procesar inmediatamente los datos descargados, podemos encadenar con el script de procesamiento
# python3 scripts/process_glm_data.py --data_dir data/raw --output_dir outputs --visualize