#!/bin/bash
# run_historical_analysis_improved.sh
# Versión mejorada con diagnósticos

# Habilitar debug y salir en errores
set -x
set -e

echo "🚀 INICIANDO NOWCASTING MEJORADO"
echo "================================"

# Verificar argumentos
if [ $# -lt 2 ]; then
    echo "❌ Error: Se requieren al menos 2 argumentos"
    echo "Uso: $0 START_TIME END_TIME [opciones]"
    exit 1
fi

# Capturar argumentos
START_TIME="$1"
END_TIME="$2"
shift 2

echo "📅 Tiempo inicio: $START_TIME"
echo "📅 Tiempo fin: $END_TIME"
echo "⚙️ Opciones adicionales: $@"

# Verificar directorio de trabajo
NOWCASTING_DIR="/home/matias/nowcasting"
echo "📁 Cambiando a directorio: $NOWCASTING_DIR"
cd "$NOWCASTING_DIR" || { echo "❌ Error: No se pudo cambiar a $NOWCASTING_DIR"; exit 1; }

# Verificar entorno virtual
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️ Advertencia: No hay entorno virtual activo"
else
    echo "🐍 Entorno virtual: $VIRTUAL_ENV"
fi

# Verificar archivos críticos
PYTHON_SCRIPT="scripts/process_historical_data.py"
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Script Python no encontrado: $PYTHON_SCRIPT"
    exit 1
fi
echo "✅ Script Python encontrado: $PYTHON_SCRIPT"

# Construir comando Python
PYTHON_CMD="python $PYTHON_SCRIPT"
PYTHON_CMD="$PYTHON_CMD --data_dir /home/matias/nowcasting/data/raw"
PYTHON_CMD="$PYTHON_CMD --start_time '$START_TIME'"
PYTHON_CMD="$PYTHON_CMD --end_time '$END_TIME'"
PYTHON_CMD="$PYTHON_CMD $@"

echo "🐍 Comando Python a ejecutar:"
echo "   $PYTHON_CMD"

# Crear directorio de logs si no existe
LOG_DIR="$NOWCASTING_DIR/logs"
mkdir -p "$LOG_DIR"

# Crear archivo de log único
LOG_FILE="$LOG_DIR/nowcast_$(date +%Y%m%d_%H%M%S).log"
echo "📄 Log file: $LOG_FILE"

# Ejecutar comando con timeout y logging
echo "🚀 Ejecutando comando Python..."
echo "Inicio: $(date)" | tee "$LOG_FILE"

# Usar timeout de 10 minutos
timeout 600 bash -c "$PYTHON_CMD" 2>&1 | tee -a "$LOG_FILE"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}

echo "Fin: $(date)" | tee -a "$LOG_FILE"
echo "Código de salida Python: $PYTHON_EXIT_CODE" | tee -a "$LOG_FILE"

# Verificar resultado
if [ $PYTHON_EXIT_CODE -eq 0 ]; then
    echo "✅ NOWCASTING COMPLETADO EXITOSAMENTE"
    echo "📄 Log completo: $LOG_FILE"
elif [ $PYTHON_EXIT_CODE -eq 124 ]; then
    echo "⏰ TIMEOUT: El proceso tardó más de 10 minutos"
    echo "📄 Log parcial: $LOG_FILE"
    exit 124
else
    echo "❌ ERROR: El proceso falló con código $PYTHON_EXIT_CODE"
    echo "📄 Log de errores: $LOG_FILE"
    exit $PYTHON_EXIT_CODE
fi

echo "🎉 SCRIPT BASH COMPLETADO"
