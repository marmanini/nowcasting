#!/bin/bash
# scripts/run_consolidated_nowcasting.sh
#
# Script para ejecutar el sistema consolidado de nowcasting GLM
# Autor: Matias
# Fecha: Mayo 2025

# Configuración de directorios
NOWCASTING_DIR="/home/matias/nowcasting"
DATA_DIR="${NOWCASTING_DIR}/data/raw"
OUTPUT_DIR="${NOWCASTING_DIR}/outputs/consolidated"  # ← CAMBIADO: consolidated en lugar de historical
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
    echo "  $0 \"2024-12-23 22:10\" \"2024-12-23 23:00\""
    echo "  $0 \"2024-12-23 22:10\" \"2024-12-23 23:00\" --uncertainty --ensemble_models"
    echo ""
    echo "=== OPCIONES DEL SISTEMA CONSOLIDADO ==="
    echo "Parámetros del sistema:"
    echo "  --history_minutes N          Minutos de historia a mantener (default: 40)"
    echo "  --min_history_minutes N      Mínimo antes de generar visualizaciones (default: 20)"
    echo "  --window_minutes N           Tamaño de ventana temporal (default: 10)"
    echo ""
    echo "Identificación de celdas:"
    echo "  --eps N                      Parámetro eps para DBSCAN (default: 0.01)"
    echo "  --min_samples N              Parámetro min_samples para DBSCAN (default: 3)"
    echo ""
    echo "Tracking de celdas:"
    echo "  --max_distance_km N          Distancia máxima de tracking en km (default: 30)"
    echo "  --max_speed_kmh N            Velocidad máxima realista km/h (default: 100)"
    echo ""
    echo "Nowcasting:"
    echo "  --forecast_minutes N         Tiempo de pronóstico en minutos (default: 20)"
    echo "  --ensemble_models            Usar ensamble de modelos"
    echo "  --uncertainty                Calcular estimaciones de incertidumbre"
    echo ""
    echo "Visualización:"
    echo "  --debug_visualizations       Generar visualizaciones debug para cada ventana"
    echo "  --no_intermediate_vis        No generar visualizaciones intermedias (default)"
    echo ""
    exit 1
fi

# Leer argumentos
START_TIME="$1"
END_TIME="$2"
shift 2

# Crear nombre para los logs
LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/consolidated_${LOG_TIMESTAMP}.log"

echo "========================================================"
echo "🚀 EJECUTANDO SISTEMA CONSOLIDADO DE NOWCASTING GLM"
echo "⏰ Hora de ejecución: $(date)"
echo "📅 Período: ${START_TIME} a ${END_TIME}"
echo "⚙️  Argumentos adicionales: $@"
echo "📝 Log: ${LOG_FILE}"
echo "📁 Directorio de datos: ${DATA_DIR}"
echo "📁 Directorio de salida: ${OUTPUT_DIR}"
echo "========================================================"

# Verificar si se necesita descargar datos
read -p "¿Deseas descargar los datos GLM para este período? (s/n): " DOWNLOAD_DATA

if [[ $DOWNLOAD_DATA =~ ^[Ss]$ ]]; then
    echo "📡 Iniciando descarga de datos GLM..."
    
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
    
    echo "📅 Descargando datos GLM para el período..."
    
    # Crear secuencia de fechas si abarca múltiples días
    CURRENT_DATE=$(date -d "${START_YEAR}-${START_MONTH}-${START_DAY}" +"%Y%m%d")
    END_DATE=$(date -d "${END_YEAR}-${END_MONTH}-${END_DAY}" +"%Y%m%d")
    
    while [ "$CURRENT_DATE" -le "$END_DATE" ]; do
        YEAR=${CURRENT_DATE:0:4}
        MONTH=${CURRENT_DATE:4:2}
        DAY=${CURRENT_DATE:6:2}
        
        echo "📊 Descargando datos para $YEAR-$MONTH-$DAY"
        
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
            echo "  ⏰ Descargando hora $HOUR"
            
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
                
                echo "    ⏱️  Descargando minutos $MIN-$NEXT_MIN"
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
    
    echo "✅ Descarga de datos completada."
else
    echo "⏭️  Saltando descarga de datos."
fi

# Verificar que existen archivos GLM en el directorio de datos
GLM_FILES=$(find "${DATA_DIR}" -name "*.nc" 2>/dev/null | wc -l)
if [ "$GLM_FILES" -eq 0 ]; then
    echo "⚠️  ADVERTENCIA: No se encontraron archivos GLM (.nc) en ${DATA_DIR}"
    echo "   El sistema seguirá ejecutándose pero puede fallar si no hay datos."
    read -p "¿Continuar de todos modos? (s/n): " CONTINUE
    if [[ ! $CONTINUE =~ ^[Ss]$ ]]; then
        echo "❌ Ejecución cancelada por el usuario."
        exit 1
    fi
else
    echo "✅ Encontrados $GLM_FILES archivos GLM en ${DATA_DIR}"
fi

echo ""
echo "🔄 EJECUTANDO SISTEMA CONSOLIDADO DE NOWCASTING..."
echo "📍 Comando a ejecutar:"
echo "   python ${NOWCASTING_DIR}/consolidated_nowcasting_system.py \\"
echo "     --data_dir \"${DATA_DIR}\" \\"
echo "     --output_dir \"${OUTPUT_DIR}\" \\"
echo "     --start_time \"${START_TIME}\" \\"
echo "     --end_time \"${END_TIME}\" \\"
echo "     $@"
echo ""

# ← CAMBIO PRINCIPAL: Ejecutar consolidated_nowcasting_system.py en lugar de process_historical_data.py
cd "${NOWCASTING_DIR}"
python consolidated_nowcasting_system.py \
    --data_dir "${DATA_DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    --start_time "${START_TIME}" \
    --end_time "${END_TIME}" \
    "$@" 2>&1 | tee "${LOG_FILE}"

# Verificar si la ejecución fue exitosa
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉========================================================"
    echo "✅ SISTEMA CONSOLIDADO COMPLETADO EXITOSAMENTE"
    echo "⏰ Finalizado a las $(date)"
    echo "📁 Resultados en: ${OUTPUT_DIR}"
    echo "📝 Log completo: ${LOG_FILE}"
    echo ""
    
    # Mostrar archivos generados
    echo "📄 Archivos generados:"
    if [ -d "${OUTPUT_DIR}" ]; then
        ls -la "${OUTPUT_DIR}" | tail -10
        echo ""
        
        # Buscar el HTML final
        FINAL_HTML=$(find "${OUTPUT_DIR}" -name "FINAL_consolidated_nowcast_*.html" -type f | head -1)
        if [ -n "$FINAL_HTML" ]; then
            echo "🗺️  MAPA FINAL CONSOLIDADO: $FINAL_HTML"
            echo "   Para ver el mapa, abre este archivo en tu navegador."
        fi
        
        # Mostrar estadísticas
        CELL_FILES=$(find "${OUTPUT_DIR}" -name "tracked_cells_*.geojson" | wc -l)
        PRED_FILES=$(find "${OUTPUT_DIR}" -name "predictions_*.csv" | wc -l)
        echo "📊 Estadísticas: $CELL_FILES archivos de celdas, $PRED_FILES archivos de predicciones"
    fi
    echo "========================================================"
else
    echo ""
    echo "❌========================================================"
    echo "💥 ERROR EN LA EJECUCIÓN DEL SISTEMA CONSOLIDADO"
    echo "⏰ Error ocurrido a las $(date)"
    echo "🔍 Código de salida: $EXIT_CODE"
    echo "📝 Revisa el log para más detalles: ${LOG_FILE}"
    echo ""
    echo "🔧 Posibles soluciones:"
    echo "   1. Verifica que existen datos GLM en ${DATA_DIR}"
    echo "   2. Verifica que los tiempos están en formato correcto"
    echo "   3. Revisa el log para errores específicos"
    echo "   4. Ejecuta con --debug_visualizations para más información"
    echo "========================================================"
fi

exit $EXIT_CODE