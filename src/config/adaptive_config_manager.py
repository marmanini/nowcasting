#!/usr/bin/env python3
"""
Sistema de configuración adaptativa para el nowcasting GLM.
Ajusta parámetros automáticamente según las condiciones meteorológicas.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class TrackingConfig:
    """Configuración para tracking de celdas."""
    max_distance_km: float = 30.0
    max_speed_kmh: float = 100.0
    intensity_weight: float = 0.3
    size_weight: float = 0.2
    shape_weight: float = 0.1
    prediction_weight: float = 0.4
    overlap_threshold: float = 0.1
    adaptive_thresholds: bool = True
    min_track_length: int = 2

@dataclass
class IdentificationConfig:
    """Configuración para identificación de celdas."""
    eps: float = 0.01
    min_samples: int = 3
    use_time_weight: bool = False
    dynamic_eps: bool = True
    eps_range: Tuple[float, float] = (0.005, 0.02)
    min_samples_range: Tuple[int, int] = (2, 8)

@dataclass
class NowcastingConfig:
    """Configuración para nowcasting."""
    forecast_minutes: int = 20
    min_history_points: int = 2
    ensemble_models: bool = True
    uncertainty_quantification: bool = True
    max_forecast_minutes: int = 60
    physics_constraints: bool = True

@dataclass
class SystemConfig:
    """Configuración completa del sistema."""
    tracking: TrackingConfig
    identification: IdentificationConfig
    nowcasting: NowcastingConfig
    window_minutes: int = 10
    output_formats: List[str] = None
    quality_control: bool = True
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ['geojson', 'csv', 'html']

class AdaptiveConfigManager:
    """
    Gestor de configuración adaptativa que ajusta parámetros 
    según las condiciones observadas.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Inicializa el gestor de configuración.
        
        Args:
            config_file: Archivo JSON con configuración base (opcional)
        """
        self.config_file = config_file
        self.base_config = self._load_base_config()
        self.current_config = self._copy_config(self.base_config)
        
        # Historial para adaptación
        self.performance_history = []
        self.condition_history = []
        
        # Reglas de adaptación
        self.adaptation_rules = self._define_adaptation_rules()
    
    def _load_base_config(self) -> SystemConfig:
        """Carga configuración base desde archivo o usa defaults."""
        if self.config_file and os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_dict = json.load(f)
                return self._dict_to_config(config_dict)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}, using defaults")
        
        # Configuración por defecto
        return SystemConfig(
            tracking=TrackingConfig(),
            identification=IdentificationConfig(),
            nowcasting=NowcastingConfig()
        )
    
    def _dict_to_config(self, config_dict: Dict) -> SystemConfig:
        """Convierte diccionario a objeto de configuración."""
        tracking_dict = config_dict.get('tracking', {})
        identification_dict = config_dict.get('identification', {})
        nowcasting_dict = config_dict.get('nowcasting', {})
        
        return SystemConfig(
            tracking=TrackingConfig(**tracking_dict),
            identification=IdentificationConfig(**identification_dict),
            nowcasting=NowcastingConfig(**nowcasting_dict),
            window_minutes=config_dict.get('window_minutes', 10),
            output_formats=config_dict.get('output_formats', ['geojson', 'csv', 'html']),
            quality_control=config_dict.get('quality_control', True)
        )
    
    def _copy_config(self, config: SystemConfig) -> SystemConfig:
        """Crea una copia profunda de la configuración."""
        config_dict = asdict(config)
        return self._dict_to_config(config_dict)
    
    def _define_adaptation_rules(self) -> List[Dict]:
        """Define reglas de adaptación automática."""
        return [
            {
                'name': 'high_storm_density',
                'condition': lambda metrics: metrics.get('storm_density', 0) > 10,
                'adaptations': {
                    'identification.eps': lambda current: min(current * 0.8, 0.015),
                    'tracking.max_distance_km': lambda current: min(current * 0.9, 25)
                }
            },
            {
                'name': 'fast_moving_storms',
                'condition': lambda metrics: metrics.get('avg_storm_speed', 0) > 60,
                'adaptations': {
                    'tracking.max_speed_kmh': lambda current: min(current * 1.2, 150),
                    'tracking.prediction_weight': lambda current: min(current * 1.1, 0.6)
                }
            },
            {
                'name': 'weak_storms',
                'condition': lambda metrics: metrics.get('avg_storm_intensity', 0) < 10,
                'adaptations': {
                    'identification.min_samples': lambda current: max(current - 1, 2),
                    'tracking.intensity_weight': lambda current: max(current * 0.8, 0.1)
                }
            },
            {
                'name': 'tracking_fragmentation',
                'condition': lambda metrics: metrics.get('track_fragmentation', 0) > 0.7,
                'adaptations': {
                    'tracking.max_distance_km': lambda current: min(current * 1.2, 40),
                    'identification.eps': lambda current: min(current * 1.1, 0.02)
                }
            },
            {
                'name': 'poor_nowcast_accuracy',
                'condition': lambda metrics: metrics.get('nowcast_error', 0) > 15,
                'adaptations': {
                    'nowcasting.min_history_points': lambda current: min(current + 1, 5),
                    'nowcasting.ensemble_models': lambda current: True
                }
            }
        ]
    
    def analyze_conditions(self, flash_data: pd.DataFrame, 
                          cells_data: pd.DataFrame = None) -> Dict:
        """
        Analiza condiciones meteorológicas actuales.
        
        Args:
            flash_data: DataFrame con datos de rayos
            cells_data: DataFrame con datos de celdas (opcional)
            
        Returns:
            Diccionario con métricas de condiciones
        """
        conditions = {}
        
        if not flash_data.empty:
            # Densidad de tormentas
            if 'cluster' in flash_data.columns:
                n_clusters = flash_data['cluster'].nunique() - (1 if -1 in flash_data['cluster'].values else 0)
                area_covered = self._estimate_area_covered(flash_data)
                conditions['storm_density'] = n_clusters / max(area_covered, 1)
            
            # Intensidad promedio
            conditions['avg_storm_intensity'] = flash_data.get('flash_energy', pd.Series([0])).mean()
            
            # Distribución espacial
            lat_span = flash_data['flash_lat'].max() - flash_data['flash_lat'].min()
            lon_span = flash_data['flash_lon'].max() - flash_data['flash_lon'].min()
            conditions['spatial_extent'] = np.sqrt(lat_span**2 + lon_span**2) * 111  # km aproximados
        
        if cells_data is not None and not cells_data.empty:
            # Velocidad promedio de celdas
            if 'velocity_lat' in cells_data.columns and 'velocity_lon' in cells_data.columns:
                velocities = np.sqrt(cells_data['velocity_lat']**2 + cells_data['velocity_lon']**2) * 111
                conditions['avg_storm_speed'] = velocities.mean()
            
            # Tamaño promedio de celdas
            conditions['avg_cell_size'] = cells_data.get('area_km2', pd.Series([0])).mean()
        
        return conditions
    
    def _estimate_area_covered(self, flash_data: pd.DataFrame) -> float:
        """Estima área cubierta por rayos en km²."""
        if len(flash_data) < 3:
            return 1.0
        
        lat_span = flash_data['flash_lat'].max() - flash_data['flash_lat'].min()
        lon_span = flash_data['flash_lon'].max() - flash_data['flash_lon'].min()
        
        # Aproximación simple del área
        area_deg2 = lat_span * lon_span
        area_km2 = area_deg2 * 111 * 111  # Conversión aproximada
        
        return max(area_km2, 1.0)
    
    def adapt_configuration(self, performance_metrics: Dict, 
                          condition_metrics: Dict) -> bool:
        """
        Adapta la configuración basándose en métricas de rendimiento y condiciones.
        
        Args:
            performance_metrics: Métricas de rendimiento del sistema
            condition_metrics: Métricas de condiciones meteorológicas
            
        Returns:
            True si se realizaron adaptaciones, False en caso contrario
        """
        adaptations_made = False
        combined_metrics = {**performance_metrics, **condition_metrics}
        
        # Evaluar reglas de adaptación
        for rule in self.adaptation_rules:
            if rule['condition'](combined_metrics):
                logger.info(f"Activando regla de adaptación: {rule['name']}")
                
                for param_path, adaptation_func in rule['adaptations'].items():
                    # Obtener valor actual
                    current_value = self._get_config_value(param_path)
                    
                    # Aplicar adaptación
                    new_value = adaptation_func(current_value)
                    
                    # Establecer nuevo valor
                    self._set_config_value(param_path, new_value)
                    
                    logger.info(f"Adaptado {param_path}: {current_value} → {new_value}")
                    adaptations_made = True
        
        # Guardar historial
        self.performance_history.append({
            'timestamp': datetime.now(),
            'metrics': performance_metrics.copy(),
            'adaptations_made': adaptations_made
        })
        
        self.condition_history.append({
            'timestamp': datetime.now(),
            'conditions': condition_metrics.copy()
        })
        
        return adaptations_made
    
    def _get_config_value(self, param_path: str):
        """Obtiene valor de configuración usando notación de puntos."""
        parts = param_path.split('.')
        obj = self.current_config
        
        for part in parts:
            obj = getattr(obj, part)
        
        return obj
    
    def _set_config_value(self, param_path: str, value):
        """Establece valor de configuración usando notación de puntos."""
        parts = param_path.split('.')
        obj = self.current_config
        
        # Navegar hasta el objeto padre
        for part in parts[:-1]:
            obj = getattr(obj, part)
        
        # Establecer el valor final
        setattr(obj, parts[-1], value)
    
    def get_configuration_for_scenario(self, scenario: str) -> SystemConfig:
        """
        Devuelve configuración optimizada para un escenario específico.
        
        Args:
            scenario: Tipo de escenario ('isolated_storms', 'squall_line', 
                     'multicell', 'supercell', 'weak_convection')
        """
        config = self._copy_config(self.base_config)
        
        if scenario == 'isolated_storms':
            # Tormentas aisladas - tracking más permisivo
            config.tracking.max_distance_km = 35
            config.tracking.intensity_weight = 0.4
            config.identification.eps = 0.015
            
        elif scenario == 'squall_line':
            # Línea de turbonada - énfasis en velocidad y dirección
            config.tracking.max_speed_kmh = 120
            config.tracking.prediction_weight = 0.5
            config.nowcasting.physics_constraints = True
            
        elif scenario == 'multicell':
            # Sistema multicelular - manejo de splits/mergers
            config.tracking.max_distance_km = 40
            config.tracking.overlap_threshold = 0.2
            config.identification.eps = 0.02
            
        elif scenario == 'supercell':
            # Supercélula - tracking de alta calidad
            config.tracking.min_track_length = 3
            config.nowcasting.min_history_points = 3
            config.nowcasting.uncertainty_quantification = True
            
        elif scenario == 'weak_convection':
            # Convección débil - parámetros más sensibles
            config.identification.min_samples = 2
            config.identification.eps = 0.008
            config.tracking.intensity_weight = 0.2
        
        return config
    
    def save_configuration(self, filename: str):
        """Guarda configuración actual a archivo JSON."""
        config_dict = asdict(self.current_config)
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        logger.info(f"Configuración guardada en: {filename}")
    
    def load_configuration(self, filename: str):
        """Carga configuración desde archivo JSON."""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            self.current_config = self._dict_to_config(config_dict)
            logger.info(f"Configuración cargada desde: {filename}")
            
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def generate_performance_report(self, output_file: str):
        """Genera reporte de rendimiento histórico."""
        if not self.performance_history:
            logger.warning("No hay datos de rendimiento para reportar")
            return
        
        # Crear DataFrame con historial
        performance_df = pd.DataFrame([
            {
                'timestamp': entry['timestamp'],
                'adaptations_made': entry['adaptations_made'],
                **entry['metrics']
            }
            for entry in self.performance_history
        ])
        
        # Guardar CSV
        performance_df.to_csv(output_file.replace('.txt', '.csv'), index=False)
        
        # Generar reporte textual
        with open(output_file, 'w') as f:
            f.write("REPORTE DE RENDIMIENTO ADAPTATIVO\n")
            f.write("=" * 40 + "\n\n")
            
            f.write(f"Período: {performance_df['timestamp'].min()} a {performance_df['timestamp'].max()}\n")
            f.write(f"Total de evaluaciones: {len(performance_df)}\n")
            f.write(f"Adaptaciones realizadas: {performance_df['adaptations_made'].sum()}\n\n")
            
            # Estadísticas de métricas
            numeric_cols = performance_df.select_dtypes(include=[np.number]).columns
            stats = performance_df[numeric_cols].describe()
            
            f.write("ESTADÍSTICAS DE MÉTRICAS:\n")
            f.write(stats.to_string())
            
        logger.info(f"Reporte de rendimiento guardado en: {output_file}")

class ConfigurationValidator:
    """Valida configuraciones para detectar valores problemáticos."""
    
    @staticmethod
    def validate_config(config: SystemConfig) -> List[str]:
        """
        Valida configuración y devuelve lista de advertencias.
        
        Returns:
            Lista de strings con advertencias/errores
        """
        warnings = []
        
        # Validar tracking
        if config.tracking.max_distance_km > 100:
            warnings.append("max_distance_km muy alto (>100 km), puede causar asociaciones incorrectas")
        
        if config.tracking.max_speed_kmh > 200:
            warnings.append("max_speed_kmh muy alto (>200 km/h), poco realista para tormentas")
        
        weight_sum = (config.tracking.intensity_weight + 
                     config.tracking.size_weight + 
                     config.tracking.shape_weight + 
                     config.tracking.prediction_weight)
        
        if abs(weight_sum - 1.0) > 0.1:
            warnings.append(f"Los pesos de tracking no suman ~1.0 (suma actual: {weight_sum:.2f})")
        
        # Validar identificación
        if config.identification.eps > 0.05:
            warnings.append("eps muy alto (>0.05), puede agrupar tormentas separadas")
        
        if config.identification.min_samples < 2:
            warnings.append("min_samples muy bajo (<2), puede crear clusters espurios")
        
        # Validar nowcasting
        if config.nowcasting.forecast_minutes > 120:
            warnings.append("forecast_minutes muy alto (>120 min), precisión será baja")
        
        if config.nowcasting.min_history_points > 10:
            warnings.append("min_history_points muy alto (>10), pocas predicciones se generarán")
        
        return warnings

def create_config_from_args(args) -> SystemConfig:
    """
    Crea configuración desde argumentos de línea de comandos.
    
    Args:
        args: Objeto con argumentos parseados
        
    Returns:
        Configuración del sistema
    """
    tracking_config = TrackingConfig(
        max_distance_km=getattr(args, 'max_distance_km', 30),
        max_speed_kmh=getattr(args, 'max_speed_kmh', 100),
        intensity_weight=getattr(args, 'intensity_weight', 0.3),
        size_weight=getattr(args, 'size_weight', 0.2),
        prediction_weight=getattr(args, 'prediction_weight', 0.4)
    )
    
    identification_config = IdentificationConfig(
        eps=getattr(args, 'eps', 0.01),
        min_samples=getattr(args, 'min_samples', 3),
        use_time_weight=getattr(args, 'use_time_weight', False)
    )
    
    nowcasting_config = NowcastingConfig(
        forecast_minutes=getattr(args, 'forecast_minutes', 20),
        min_history_points=getattr(args, 'min_history_points', 2),
        ensemble_models=getattr(args, 'ensemble_models', True),
        uncertainty_quantification=getattr(args, 'uncertainty', True)
    )
    
    return SystemConfig(
        tracking=tracking_config,
        identification=identification_config,
        nowcasting=nowcasting_config,
        window_minutes=getattr(args, 'window_minutes', 10)
    )

# Ejemplo de uso
if __name__ == "__main__":
    # Crear gestor de configuración
    config_manager = AdaptiveConfigManager()
    
    # Simular análisis de condiciones
    conditions = {
        'storm_density': 15,  # Alta densidad
        'avg_storm_speed': 45,  # Velocidad normal
        'avg_storm_intensity': 8  # Intensidad baja
    }
    
    # Simular métricas de rendimiento
    performance = {
        'track_fragmentation': 0.8,  # Alta fragmentación
        'nowcast_error': 12  # Error aceptable
    }
    
    # Adaptar configuración
    adapted = config_manager.adapt_configuration(performance, conditions)
    
    if adapted:
        print("Configuración adaptada exitosamente")
        print(f"Nueva eps: {config_manager.current_config.identification.eps}")
        print(f"Nueva max_distance: {config_manager.current_config.tracking.max_distance_km}")
    
    # Validar configuración
    validator = ConfigurationValidator()
    warnings = validator.validate_config(config_manager.current_config)
    
    if warnings:
        print("\nAdvertencias de configuración:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("\nConfiguración válida")