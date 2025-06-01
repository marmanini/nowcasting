#!/usr/bin/env python3
"""
Ejemplo de uso del sistema consolidado de nowcasting GLM.

Este script demuestra cómo usar el nuevo sistema que genera un solo HTML
con datos históricos de 20-40 minutos y pronósticos con métricas de incertidumbre.
"""

import os
import sys
from datetime import datetime, timedelta

# Asegurar que los módulos locales estén en el path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def run_example_consolidated_analysis():
    """
    Ejecuta un análisis de ejemplo con el sistema consolidado.
    """
    print("=== EJEMPLO DE SISTEMA CONSOLIDADO DE NOWCASTING GLM ===")
    print()
    
    # Configuración de ejemplo
    config = {
        'data_dir': './data/glm_data',  # Directorio con archivos GLM
        'output_dir': './output/consolidated',  # Directorio de salida
        'start_time': '2024-12-23 22:00',  # Tiempo de inicio
        'end_time': '2024-12-23 23:30',    # Tiempo de fin (1.5 horas = suficiente para tracking)
        'history_minutes': 40,              # Mantener 40 minutos de historial
        'min_history_minutes': 20,          # Mínimo 20 minutos antes de generar pronósticos
        'window_minutes': 10,               # Ventanas de 10 minutos
        'forecast_minutes': 20,             # Pronósticos a 20 minutos
        'uncertainty': True,                # Habilitar cálculo de incertidumbre
        'ensemble_models': True             # Usar ensemble de modelos
    }
    
    print("Configuración del análisis:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()
    
    # Construir comando para ejecutar el sistema consolidado
    cmd_parts = [
        'python', 'consolidated_nowcasting_system.py',
        f'--data_dir {config["data_dir"]}',
        f'--start_time "{config["start_time"]}"',
        f'--end_time "{config["end_time"]}"', 
        f'--output_dir {config["output_dir"]}',
        f'--history_minutes {config["history_minutes"]}',
        f'--min_history_minutes {config["min_history_minutes"]}',
        f'--window_minutes {config["window_minutes"]}',
        f'--forecast_minutes {config["forecast_minutes"]}'
    ]
    
    if config['uncertainty']:
        cmd_parts.append('--uncertainty')
    
    if config['ensemble_models']:
        cmd_parts.append('--ensemble_models')
    
    cmd = ' '.join(cmd_parts)
    
    print("Comando a ejecutar:")
    print(cmd)
    print()
    
    # Crear directorio de salida si no existe
    os.makedirs(config['output_dir'], exist_ok=True)
    
    print("Ejecutando análisis consolidado...")
    
    # Ejecutar comando
    import subprocess
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Análisis completado exitosamente!")
            print()
            print("Archivos generados:")
            
            # Listar archivos en el directorio de salida
            if os.path.exists(config['output_dir']):
                for file in os.listdir(config['output_dir']):
                    file_path = os.path.join(config['output_dir'], file)
                    file_size = os.path.getsize(file_path) / 1024  # KB
                    print(f"  📄 {file} ({file_size:.1f} KB)")
            
            print()
            print("Para ver los resultados:")
            print(f"  1. Abre el archivo HTML principal en tu navegador")
            print(f"  2. Revisa el reporte JSON para métricas detalladas")
            print(f"  3. Opcionalmente, abre el dashboard de rendimiento")
            
        else:
            print("❌ Error ejecutando el análisis:")
            print(result.stderr)
            
    except Exception as e:
        print(f"❌ Error: {e}")

def create_test_data_structure():
    """
    Crea una estructura de directorios de ejemplo para datos de prueba.
    """
    print("\n=== CREANDO ESTRUCTURA DE DATOS DE PRUEBA ===")
    
    # Directorios a crear
    directories = [
        '/home/matias/nowcasting/data/raw',
        '/home/matias/nowcasting/outputs/consolidated',
        '/home/matias/nowcasting/outputs/individual_windows',
        '/home/matias/nowcasting/logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Directorio creado: {directory}")
    
    # Crear archivo de configuración de ejemplo
    config_content = """# Configuración del Sistema de Nowcasting GLM

## Estructura de directorios necesaria:
- data/glm_data/          # Archivos NetCDF del GLM
- output/consolidated/    # Salida del sistema consolidado  
- output/individual_windows/  # Salida del sistema original
- logs/                   # Archivos de log

## Formato de archivos GLM esperado:
OR_GLM-L2-LCFA_G16_s20231223220000_e20231223220200_c20231223220231.nc

## Comandos de ejemplo:

### Sistema consolidado (recomendado):
python consolidated_nowcasting_system.py \\
  --data_dir ./data/glm_data \\
  --start_time "2024-12-23 22:00" \\
  --end_time "2024-12-23 23:30" \\
  --output_dir ./output/consolidated \\
  --history_minutes 40 \\
  --min_history_minutes 20 \\
  --uncertainty \\
  --ensemble_models

### Sistema original (múltiples HTMLs):
python improved_nowcasting_system.py \\
  --data_dir ./data/glm_data \\
  --start_time "2024-12-23 22:00" \\
  --end_time "2024-12-23 23:30" \\
  --output_dir ./output/individual_windows \\
  --visualize \\
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
"""
    
    with open('./CONFIGURACION.md', 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print("✓ Archivo de configuración creado: CONFIGURACION.md")
    print()

def show_comparison():
    """
    Muestra una comparación entre el sistema original y el consolidado.
    """
    print("\n=== COMPARACIÓN: SISTEMA ORIGINAL vs CONSOLIDADO ===")
    print()
    
    comparison = [
        ("Característica", "Sistema Original", "Sistema Consolidado"),
        ("─" * 50, "─" * 30, "─" * 30),
        ("Salida", "Un HTML por ventana temporal", "Un solo HTML consolidado"),
        ("Historial", "Solo tiempo actual", "20-40 minutos de historial"),
        ("Tracking", "Básico entre ventanas", "Tracking continuo mejorado"),
        ("Pronósticos", "Sin verificación", "Con verificación automática"),
        ("Incertidumbre", "Básica", "Múltiples intervalos de confianza"),
        ("Métricas", "Estadísticas finales", "Métricas en tiempo real"),
        ("Confianza", "Del modelo únicamente", "Histórica + modelo + método"),
        ("Visualización", "Mapas individuales", "Mapa integrado + dashboard"),
        ("Rendimiento", "Reporte al final", "Monitoreo continuo"),
        ("Usabilidad", "Múltiples archivos", "Un archivo principal")
    ]
    
    for row in comparison:
        print(f"{row[0]:<50} | {row[1]:<30} | {row[2]:<30}")
    
    print()
    print("💡 RECOMENDACIÓN: Usar el sistema consolidado para:")
    print("   • Operaciones en tiempo real")
    print("   • Análisis de rendimiento continuo")
    print("   • Mejor experiencia de usuario")
    print("   • Métricas de confianza más robustas")
    print()
    print("📋 Usar el sistema original para:")
    print("   • Análisis de ventanas específicas")
    print("   • Debugging de algoritmos")
    print("   • Comparación temporal detallada")
    print()

# def validate_dependencies():
#     """
#     Valida que todas las dependencias estén instaladas.
#     """
#     print("\n=== VALIDACIÓN DE DEPENDENCIAS ===")
    
#     required_packages = [
#         ('numpy', 'Operaciones numéricas'),
#         ('pandas', 'Manipulación de datos'),
#         ('geopandas', 'Datos geoespaciales'),
#         ('shapely', 'Geometrías'),
#         ('folium', 'Mapas interactivos'),
#         ('scikit-learn', 'Algoritmos de machine learning'),
#         ('xarray', 'Datos multidimensionales (GLM)'),
#         ('matplotlib', 'Gráficos (opcional para dashboard)')
#     ]
    
#     missing_packages = []
    
#     for package, description in required_packages:
#         try:
#             __import__(package)
#             print(f"✓ {package:<15} - {description}")
#         except ImportError:
#             print(f"❌ {package:<15} - {description} (FALTANTE)")
#             missing_packages.append(package)
    
#     print()
    
#     if missing_packages:
#         print("⚠️  Paquetes faltantes detectados. Para instalarlos:")
#         print(f"pip install {' '.join(missing_packages)}")
#         print()
#         return False
#     else:
#         print("✅ Todas las dependencias están instaladas correctamente!")
#         print()
#         return True

def main():
    """
    Función principal del script de ejemplo.
    """
    print("🚀 SISTEMA CONSOLIDADO DE NOWCASTING GLM")
    print("=" * 60)
    
    # Mostrar comparación
    show_comparison()
    
    # # Validar dependencias
    # dependencies_ok = validate_dependencies()
    
    # if not dependencies_ok:
    #     print("❌ Por favor, instala las dependencias faltantes antes de continuar.")
    #     return
    
    # Crear estructura de directorios
    create_test_data_structure()
    
    # Mostrar opciones
    print("Opciones disponibles:")
    print("1. Ejecutar análisis de ejemplo (requiere datos GLM)")
    print("2. Solo mostrar comandos de ejemplo")
    print("3. Salir")
    print()
    
    try:
        choice = input("Selecciona una opción (1-3): ").strip()
        
        if choice == '1':
            # Verificar si existen datos
            if os.path.exists('/home/matias/nowcasting/data/raw') and os.listdir('/home/matias/nowcasting/data/raw'):
                run_example_consolidated_analysis()
            else:
                print("❌ No se encontraron datos GLM en /home/matias/nowcasting/data/raw")
                print("   Por favor, coloca archivos NetCDF del GLM en ese directorio.")
                print("   Formato esperado: OR_GLM-L2-LCFA_G16_s*.nc")
        
        elif choice == '2':
            print("\n=== COMANDOS DE EJEMPLO ===")
            print()
            print("1. Sistema consolidado básico:")
            print("python consolidated_nowcasting_system.py \\")
            print("  --data_dir ./data/glm_data \\")
            print("  --start_time \"2024-12-23 22:00\" \\")
            print("  --end_time \"2024-12-23 23:30\" \\")
            print("  --output_dir ./output/consolidated")
            print()
            
            print("2. Sistema consolidado con todas las opciones:")
            print("python consolidated_nowcasting_system.py \\")
            print("  --data_dir ./data/glm_data \\")
            print("  --start_time \"2024-12-23 22:00\" \\")
            print("  --end_time \"2024-12-23 23:30\" \\")
            print("  --output_dir ./output/consolidated \\")
            print("  --history_minutes 40 \\")
            print("  --min_history_minutes 20 \\")
            print("  --forecast_minutes 20 \\")
            print("  --uncertainty \\")
            print("  --ensemble_models")
            print()
            
        elif choice == '3':
            print("👋 ¡Hasta luego!")
            
        else:
            print("❌ Opción no válida.")
    
    except KeyboardInterrupt:
        print("\n👋 Proceso interrumpido por el usuario.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()