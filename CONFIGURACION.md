# Configuración del Sistema de Nowcasting GLM

## Estructura de directorios necesaria:
- data/glm_data/          # Archivos NetCDF del GLM
- output/consolidated/    # Salida del sistema consolidado  
- output/individual_windows/  # Salida del sistema original
- logs/                   # Archivos de log

## Formato de archivos GLM esperado:
OR_GLM-L2-LCFA_G16_s20231223220000_e20231223220200_c20231223220231.nc

## Comandos de ejemplo:

### Sistema consolidado (recomendado):
python consolidated_nowcasting_system.py \
  --data_dir ./data/glm_data \
  --start_time "2024-12-23 22:00" \
  --end_time "2024-12-23 23:30" \
  --output_dir ./output/consolidated \
  --history_minutes 40 \
  --min_history_minutes 20 \
  --uncertainty \
  --ensemble_models

### Sistema original (múltiples HTMLs):
python improved_nowcasting_system.py \
  --data_dir ./data/glm_data \
  --start_time "2024-12-23 22:00" \
  --end_time "2024-12-23 23:30" \
  --output_dir ./output/individual_windows \
  --visualize \
  --uncertainty

## Salidas esperadas del sistema consolidado:
- consolidated_nowcast_YYYYMMDD_HHMMSS.html    # Mapa principal
- consolidated_performance_report.json          # Métricas de rendimiento  
- performance_dashboard_YYYYMMDD_HHMMSS.html    # Dashboard opcional

## Mejoras del sistema consolidado vs original:
1. ✓ Un solo HTML en lugar de múltiples archivos
2. ✓ Historial de tracking de 20-40 minutos
3. ✓ Cálculo y visualización de incertidumbre mejorada
4. ✓ Verificación automática de predicciones anteriores
5. ✓ Métricas de rendimiento en tiempo real
6. ✓ Porcentajes de confianza basados en historial
7. ✓ Dashboard de rendimiento con gráficos
8. ✓ Intervalos de confianza múltiples (40%, 60%, 80%, 90%)
