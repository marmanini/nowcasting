# src/__init__.py
"""
Paquete para el sistema de nowcasting de descargas eléctricas GLM.
"""

import os
import logging

# Configuración de logging
logger = logging.getLogger(__name__)

# Creación de directorios necesarios
def create_necessary_directories():
    """
    Crea los directorios necesarios para el funcionamiento del sistema.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dirs = [
        os.path.join(base_dir, 'data', 'raw'),
        os.path.join(base_dir, 'data', 'processed'),
        os.path.join(base_dir, 'logs'),
        os.path.join(base_dir, 'outputs', 'images'),
        os.path.join(base_dir, 'outputs', 'predictions')
    ]
    
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.debug(f"Created directory: {directory}")

# Crear directorios al importar el paquete
create_necessary_directories()