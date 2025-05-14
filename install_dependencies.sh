#!/bin/bash
# scripts/install_dependencies.sh (corregido)
#
# Script para instalar todas las dependencias necesarias para el sistema de nowcasting
# Autor: Matias
# Fecha: Mayo 2025

echo "===== Instalando dependencias para el sistema de nowcasting GLM ====="

# Crear entorno virtual (opcional)
read -p "¿Deseas crear un entorno virtual para la instalación? (s/n): " CREATE_VENV
if [[ $CREATE_VENV =~ ^[Ss]$ ]]; then
    echo "Creando entorno virtual..."
    python -m venv venv
    
    # Activar entorno virtual según el sistema operativo
    if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
        source venv/Scripts/activate
    else
        source venv/bin/activate
    fi
    
    echo "Entorno virtual activado."
fi

# Actualizar pip
echo "Actualizando pip..."
pip install --upgrade pip

# Instalar dependencias básicas
echo "Instalando dependencias básicas..."
pip install numpy pandas xarray netCDF4 scipy matplotlib

# Instalar dependencias para análisis espacial
echo "Instalando dependencias para análisis espacial..."
pip install geopandas shapely pyproj

# Instalar dependencias para machine learning
echo "Instalando dependencias para machine learning..."
pip install scikit-learn statsmodels

# Instalar dependencias para visualización
echo "Instalando dependencias para visualización..."
pip install folium==0.14.0 branca==0.6.0

# Instalar dependencias para animaciones
echo "Instalando dependencias para animaciones..."
pip install imageio selenium

# Instalar dependencias para Jupyter (opcional)
read -p "¿Deseas instalar Jupyter para trabajar con notebooks? (s/n): " INSTALL_JUPYTER
if [[ $INSTALL_JUPYTER =~ ^[Ss]$ ]]; then
    echo "Instalando Jupyter..."
    pip install jupyter notebook
fi

# Instalar AWS CLI para descarga de datos
read -p "¿Deseas instalar AWS CLI para descarga de datos? (s/n): " INSTALL_AWS
if [[ $INSTALL_AWS =~ ^[Ss]$ ]]; then
    echo "Instalando AWS CLI..."
    pip install boto3 awscli
fi

# Verificar instalación
echo "Verificando instalación..."
python -c "import folium; print('Folium instalado correctamente: versión', folium.__version__)"
python -c "try:
    import folium.plugins
    print('Plugins de folium disponibles')
except (ImportError, AttributeError) as e:
    print('ADVERTENCIA: Plugins de folium no disponibles -', str(e))
"
python -c "import xarray; print('Xarray instalado correctamente: versión', xarray.__version__)"

echo "===== Instalación completa ====="
echo "Puedes ejecutar ahora el sistema de nowcasting con:"
echo "python scripts/process_historical_data.py --start_time \"YYYY-MM-DD HH:MM\" --end_time \"YYYY-MM-DD HH:MM\" --visualize"