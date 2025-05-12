#!/bin/bash
# scripts/run_operational_nowcasting.sh
#
# Script para ejecutar el sistema de nowcasting de forma operativa
# Autor: Matias
# Fecha: Mayo 2025

# Configuración de directorios
NOWCASTING_DIR="/home/matias/nowcasting"
DATA_DIR="${NOWCASTING_DIR}/data/raw"
OUTPUT_DIR="${NOWCASTING_DIR}/outputs"
LOG_DIR="${NOWCASTING_DIR}/logs"

# Asegurar que los directorios existen
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}/images"
mkdir -p "${OUTPUT_DIR}/predictions"
mkdir -p "${LOG_DIR}"

# Configuración de tiempo
CURRENT_DATE=$(date -u +"%Y%m%d")
CURRENT_HOUR=$(date -u +"%H")
CURRENT_MIN=$(date -u +"%M")

# Redondear a decaminuto anterior
ROUNDED_MIN=$(( (CURRENT_MIN / 10) * 10 ))
MINUTE_RANGE="${ROUNDED_MIN}-$((ROUNDED_MIN + 10))"

# El último período de 10 minutos
if [ $ROUNDED_MIN -eq 50 ]; then
    NEXT_HOUR=$(( (CURRENT_HOUR + 1) % 24 ))
    NEXT_DAY=$CURRENT_DATE
    if [ $NEXT_HOUR -eq 0 ] && [ $CURRENT_HOUR -eq 23 ]; then
        # Calcular el siguiente día
        NEXT_DAY=$(date -u -d "${CURRENT_DATE} + 1 day" +"%Y%m%d")
    fi
    END_TIME="${NEXT_DAY}-${NEXT_HOUR}:00"
else
    END_TIME="${CURRENT_DATE}-${CURRENT_HOUR}:$((ROUNDED_MIN + 10))"
fi

# Tiempo de inicio 30 minutos antes
START_TIME=$(date -u -d "${CURRENT_DATE} ${CURRENT_HOUR}:${ROUNDED_MIN} - 30 minutes" +"%Y-%m-%d %H:%M")

echo "========================================================"
echo "Ejecutando nowcasting de descargas eléctricas GLM"
echo "Hora de ejecución: $(date -u)"
echo "Período: ${START_TIME} a ${END_TIME} UTC"
echo "========================================================"

# 1. Descargar datos GLM recientes
echo "1. Descargando datos GLM recientes..."
python ${NOWCASTING_DIR}/scripts/download_glm_data.py --minute "${MINUTE_RANGE}" --debug

# 2. Ejecutar sistema de nowcasting
echo "2. Ejecutando sistema de nowcasting..."
python ${NOWCASTING_DIR}/scripts/run_nowcasting.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_time "${START_TIME}" \
    --end_time "${END_TIME}" \
    --window_minutes 10 \
    --forecast_minutes 20 \
    --eps 0.05 \
    --min_samples 5 \
    --max_distance_km 30 \
    --visualize \
    --debug

# 3. Verificar resultados
TIMESTAMP=$(date -u +"%Y%m%d_%H%M%S")
LATEST_PREDICTION=$(ls -t ${OUTPUT_DIR}/predictions/*.csv 2>/dev/null | head -1)
LATEST_MAP=$(ls -t ${OUTPUT_DIR}/images/*.html 2>/dev/null | head -1)

if [ -f "${LATEST_PREDICTION}" ]; then
    NUM_PREDICTIONS=$(wc -l < "${LATEST_PREDICTION}")
    NUM_PREDICTIONS=$((NUM_PREDICTIONS - 1))  # Restar encabezado
    echo "3. Procesamiento completado con éxito."
    echo "   - Se generaron ${NUM_PREDICTIONS} predicciones."
    echo "   - Último archivo de predicción: $(basename ${LATEST_PREDICTION})"
    
    if [ -f "${LATEST_MAP}" ]; then
        echo "   - Último mapa generado: $(basename ${LATEST_MAP})"
    else
        echo "   - No se generaron mapas en esta ejecución."
    fi
else
    echo "3. Advertencia: No se generaron predicciones en esta ejecución."
fi

echo "========================================================"
echo "Proceso de nowcasting completado a las $(date -u)"
echo "========================================================"

# Opcionalmente, configurar este script para que se ejecute cada 10 minutos con crontab:
# */10 * * * * /home/matias/nowcasting/scripts/run_operational_nowcasting.sh >> /home/matias/nowcasting/logs/operational.log 2>&1