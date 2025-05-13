#!/bin/bash
# scripts/force_all_storms.sh
#
# Script para forzar la identificación de todas las tormentas visibles
# Autor: Matias
# Fecha: Mayo 2025

# Verificar que se hayan proporcionado al menos 2 argumentos
if [ $# -lt 2 ]; then
    echo "Uso: $0 FECHA_INICIO FECHA_FIN [opciones_adicionales]"
    echo "Ejemplo: $0 \"2024-12-23 22:40\" \"2024-12-23 22:50\" --visualize"
    exit 1
fi

# Capturar los argumentos principales
START_TIME="$1"
END_TIME="$2"
shift 2  # Eliminar los dos primeros argumentos

# Directorio base
NOWCASTING_DIR="/home/matias/nowcasting"
OUTPUT_DIR="${NOWCASTING_DIR}/outputs/force_all_storms"

# Crear directorio de salida
mkdir -p "${OUTPUT_DIR}"

# 5. Ejecutar el análisis con parámetros extremadamente pequeños
echo "Ejecutando análisis con parámetros optimizados..."
python "${NOWCASTING_DIR}/scripts/process_historical_data.py" \
    --data_dir "${NOWCASTING_DIR}/data/raw" \
    --output_dir "${OUTPUT_DIR}" \
    --start_time "${START_TIME}" \
    --end_time "${END_TIME}" \
    --eps 0.015 \
    --min_samples 3 \
    --visualize \
    "$@"

# 7. Informar al usuario
echo "========================================================"
echo "Análisis completado"
echo "Resultados guardados en: ${OUTPUT_DIR}"
echo "========================================================"