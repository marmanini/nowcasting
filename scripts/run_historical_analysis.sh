#!/bin/bash
# scripts/run_historical_analysis.sh
#
# Script para ejecutar el análisis de datos históricos de GLM
# Autor: Matias
# Fecha: Mayo 2025

# Configuración de directorios
NOWCASTING_DIR="/home/matias/nowcasting"
DATA_DIR="${NOWCASTING_DIR}/data/raw"
OUTPUT_DIR="${NOWCASTING_DIR}/outputs/historical"
LOG_DIR="${NOWCASTING_DIR}/logs"

# Asegurar que los directorios existen
mkdir -p "${DATA_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Verificar argumentos
if [ $# -lt 2 ]; then
    echo "Uso: $0 FECHA_INICIO FECHA_FIN [opciones_adicionales]"
    echo ""
    echo "Ejemplos:"
    echo "  $0 \"2023-01-01 12:00\" \"2023-01-01 13:00\""
    echo "  $0 \"2023-01-01 12:00\" \"2023-01-01 13:00\" --visualize --validation"
    echo ""
    echo "Opciones adicionales:"
    echo "  --window_minutes N     Tamaño de ventana temporal en minutos (default: 10)"
    echo "  --forecast_minutes N   Minutos hacia el futuro para forecast (default: 20)"
    echo "  --visualize            Generar visualizaciones"
    echo "  --animation            Crear animación del evento"
    echo "  --validation           Realizar validación de predicciones"
    echo "  --debug                Habilitar logging de depuración"
    echo ""
    exit 1
fi

# Leer argumentos
START_TIME="$1"
END_TIME="$2"
shift 2

# Crear nombre para los logs
LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/historical_${LOG_TIMESTAMP}.log"

echo "========================================================"
echo "Ejecutando análisis histórico de descargas eléctricas GLM"
echo "Hora de ejecución: $(date)"
echo "Período: ${START_TIME} a ${END_TIME}"
echo "Argumentos adicionales: $@"
echo "Log: ${LOG_FILE}"
echo "========================================================"

# Verificar si se necesita descargar datos
read -p "¿Deseas descargar los datos GLM para este período? (s/n): " DOWNLOAD_DATA

if [[ $DOWNLOAD_DATA =~ ^[Ss]$ ]]; then
    # Extraer componentes de fecha y hora para inicio
    START_YEAR=$(date -d "${START_TIME}" +"%Y")
    START_MONTH=$(date -d "${START_TIME}" +"%m")
    START_DAY=$(date -d "${START_TIME}" +"%d")
    START_HOUR=$(date -d "${START_TIME}" +"%H")
    START_MIN=$(date -d "${START_TIME}" +"%M")
    
    # Extraer componentes de fecha y hora para fin
    END_YEAR=$(date -d "${END_TIME}" +"%Y")
    END_MONTH=$(date -d "${END_TIME}" +"%m")
    END_DAY=$(date -d "${END_TIME}" +"%d")
    END_HOUR=$(date -d "${END_TIME}" +"%H")
    END_MIN=$(date -d "${END_TIME}" +"%M")
    
    echo "Descargando datos GLM para el período..."
    
    # Crear secuencia de fechas si abarca múltiples días
    CURRENT_DATE=$(date -d "${START_YEAR}-${START_MONTH}-${START_DAY}" +"%Y%m%d")
    END_DATE=$(date -d "${END_YEAR}-${END_MONTH}-${END_DAY}" +"%Y%m%d")
    
    while [ "$CURRENT_DATE" -le "$END_DATE" ]; do
        YEAR=${CURRENT_DATE:0:4}
        MONTH=${CURRENT_DATE:4:2}
        DAY=${CURRENT_DATE:6:2}
        
        echo "Descargando datos para $YEAR-$MONTH-$DAY"
        
        # Determinar rango de horas
        if [ "$CURRENT_DATE" = "$(date -d "${START_YEAR}-${START_MONTH}-${START_DAY}" +"%Y%m%d")" ]; then
            START_HOUR_DAY=$START_HOUR
        else
            START_HOUR_DAY=0
        fi
        
        if [ "$CURRENT_DATE" = "$(date -d "${END_YEAR}-${END_MONTH}-${END_DAY}" +"%Y%m%d")" ]; then
            END_HOUR_DAY=$END_HOUR
        else
            END_HOUR_DAY=23
        fi
        
        # Descargar datos para cada hora
        for HOUR in $(seq -f "%02g" $START_HOUR_DAY $END_HOUR_DAY); do
            echo "  Descargando hora $HOUR"
            
            # Determinar minutos
            if [ "$CURRENT_DATE" = "$(date -d "${START_YEAR}-${START_MONTH}-${START_DAY}" +"%Y%m%d")" ] && [ "$HOUR" = "$START_HOUR" ]; then
                START_MIN_HOUR=$START_MIN
            else
                START_MIN_HOUR=0
            fi
            
            if [ "$CURRENT_DATE" = "$(date -d "${END_YEAR}-${END_MONTH}-${END_DAY}" +"%Y%m%d")" ] && [ "$HOUR" = "$END_HOUR" ]; then
                END_MIN_HOUR=$END_MIN
            else
                END_MIN_HOUR=59
            fi
            
            # Calcular rangos de minutos en incrementos de 10
            START_MIN_ROUNDED=$(( ($START_MIN_HOUR / 10) * 10 ))
            END_MIN_ROUNDED=$(( ($END_MIN_HOUR / 10) * 10 ))
            
            for MIN in $(seq -f "%02g" $START_MIN_ROUNDED 10 $END_MIN_ROUNDED); do
                NEXT_MIN=$(printf "%02d" $(( 10#$MIN + 10 )))
                if [ "$NEXT_MIN" -gt "59" ]; then
                    NEXT_MIN="59"
                fi
                
                echo "    Descargando minutos $MIN-$NEXT_MIN"
                python ${NOWCASTING_DIR}/scripts/download_glm_data.py \
                    --date "${CURRENT_DATE}" \
                    --hour "${HOUR}" \
                    --minute "${MIN}-${NEXT_MIN}" \
                    --debug
            done
        done
        
        # Avanzar al siguiente día
        CURRENT_DATE=$(date -d "${YEAR}-${MONTH}-${DAY} + 1 day" +"%Y%m%d")
    done
    
    echo "Descarga de datos completada."
fi

# Ejecutar el script de procesamiento
echo "Ejecutando análisis de datos históricos..."
python ${NOWCASTING_DIR}/scripts/process_historical_data.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_time "${START_TIME}" \
    --end_time "${END_TIME}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

echo "========================================================"
echo "Procesamiento histórico completado a las $(date)"
echo "Ver resultados en: ${OUTPUT_DIR}"
echo "========================================================"