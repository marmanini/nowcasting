#!/bin/bash
# run_historical_analysis_FINAL.sh
# Versión final que usa el script Python corregido

set -x
set -e

echo "🛰️ NOWCASTING FINAL (PROBLEMA RESUELTO)"
echo "======================================="

# Verificar argumentos
if [ $# -lt 2 ]; then
    echo "❌ Error: Se requieren al menos 2 argumentos"
    echo "Uso: $0 START_TIME END_TIME [opciones]"
    exit 1
fi

START_TIME="$1"
END_TIME="$2"
shift 2

echo "📅 Tiempo inicio: $START_TIME"
echo "📅 Tiempo fin: $END_TIME"
echo "⚙️ Opciones: $@"

# Cambiar a directorio nowcasting
NOWCASTING_DIR="/home/matias/nowcasting"
cd "$NOWCASTING_DIR" || { echo "❌ No se pudo cambiar a $NOWCASTING_DIR"; exit 1; }

# Verificar script corregido
PYTHON_SCRIPT="scripts/process_historical_data_FIXED.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Script corregido no encontrado: $PYTHON_SCRIPT"
    exit 1
fi

echo "✅ Usando script corregido: $PYTHON_SCRIPT"

# Construir comando Python
PYTHON_CMD="python $PYTHON_SCRIPT"
PYTHON_CMD="$PYTHON_CMD --data_dir /home/matias/nowcasting/data/raw"
PYTHON_CMD="$PYTHON_CMD --start_time '$START_TIME'"
PYTHON_CMD="$PYTHON_CMD --end_time '$END_TIME'"
PYTHON_CMD="$PYTHON_CMD $@"

echo "🐍 Comando: $PYTHON_CMD"

# Crear log
LOG_DIR="$NOWCASTING_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/nowcast_FINAL_$(date +%Y%m%d_%H%M%S).log"

echo "📄 Log: $LOG_FILE"
echo "🚀 Ejecutando..."

# Ejecutar con timeout
timeout 600 bash -c "$PYTHON_CMD" 2>&1 | tee "$LOG_FILE"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "📊 Código de salida: $PYTHON_EXIT_CODE"

if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ NOWCASTING COMPLETADO EXITOSAMENTE"
    echo "📄 Log: $LOG_FILE"
    
    # Mostrar archivos generados
    echo "📁 Archivos generados:"
    find outputs/historical -name "event_*" -type d | tail -1 | xargs ls -la
    
else
    echo "❌ ERROR: Código $PYTHON_EXIT_CODE"
    echo "📄 Ver errores en: $LOG_FILE"
    exit $PYTHON_EXIT_CODE
fi

echo "🎉 PROCESAMIENTO COMPLETADO"
