# verification_and_performance_metrics.py

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class PerformanceVerificationSystem:
    """
    Sistema de verificación y métricas de rendimiento para el nowcasting GLM.
    """
    
    def __init__(self):
        """Inicializa el sistema de verificación."""
        self.verification_history = []
        self.performance_stats = {
            'position_errors': [],
            'intensity_errors': [],
            'confidence_calibration': {},
            'method_performance': {}
        }
    
    def verify_prediction(self, prediction, actual_observation, tolerance_minutes=25):
        """
        Verifica una predicción contra la observación real.
        
        Args:
            prediction: Diccionario con la predicción realizada
            actual_observation: Diccionario con la observación real
            tolerance_minutes: Tolerancia temporal para considerar válida la verificación
            
        Returns:
            dict: Resultado de la verificación con métricas detalladas
        """
        # Verificar si los tiempos coinciden dentro de la tolerancia
        pred_time = pd.to_datetime(prediction['pred_time'])
        obs_time = pd.to_datetime(actual_observation['timestamp'])
        
        time_diff_minutes = abs((pred_time - obs_time).total_seconds() / 60)
        
        if time_diff_minutes > tolerance_minutes:
            logger.warning(f"Tiempo fuera de tolerancia: {time_diff_minutes:.1f} min > {tolerance_minutes} min")
            return None
        
        # Calcular errores
        position_error = self._calculate_position_error(prediction, actual_observation)
        intensity_error = self._calculate_intensity_error(prediction, actual_observation)
        area_error = self._calculate_area_error(prediction, actual_observation)
        
        # Verificar si la predicción estuvo dentro de la zona de incertidumbre
        uncertainty_radius = prediction.get('expected_error_km', float('inf'))
        within_uncertainty = position_error <= uncertainty_radius
        
        # Crear resultado de verificación
        verification_result = {
            'timestamp': obs_time,
            'track_id': prediction['track_id'],
            'forecast_method': prediction.get('forecast_method', 'unknown'),
            'lead_time_minutes': prediction.get('lead_time_min', 20),
            
            # Errores de posición
            'position_error_km': position_error,
            'predicted_uncertainty_km': uncertainty_radius,
            'within_uncertainty': within_uncertainty,
            
            # Errores de intensidad
            'intensity_error_absolute': intensity_error['absolute'],
            'intensity_error_percentage': intensity_error['percentage'],
            'intensity_bias': intensity_error['bias'],
            
            # Errores de área
            'area_error_km2': area_error['absolute'],
            'area_error_percentage': area_error['percentage'],
            'area_bias': area_error['bias'],
            
            # Confianza y calibración
            'predicted_confidence': prediction.get('confidence_percentage', 0),
            'confidence_well_calibrated': self._assess_confidence_calibration(
                prediction.get('confidence_percentage', 0), 
                within_uncertainty
            ),
            
            # Información adicional
            'prediction_coordinates': [prediction['pred_lat'], prediction['pred_lon']],
            'actual_coordinates': [actual_observation['centroid_lat'], actual_observation['centroid_lon']],
            'prediction_time': pred_time,
            'observation_time': obs_time,
            'time_difference_minutes': time_diff_minutes
        }
        
        # Almacenar en historial
        self.verification_history.append(verification_result)
        self._update_performance_statistics(verification_result)
        
        return verification_result
    
    def _calculate_position_error(self, prediction, observation):
        """Calcula el error de posición usando distancia Haversine."""
        from math import radians, cos, sin, asin, sqrt
        
        # Coordenadas predichas
        pred_lat = radians(prediction['pred_lat'])
        pred_lon = radians(prediction['pred_lon'])
        
        # Coordenadas observadas
        obs_lat = radians(observation['centroid_lat'])
        obs_lon = radians(observation['centroid_lon'])
        
        # Fórmula de Haversine
        dlon = obs_lon - pred_lon
        dlat = obs_lat - pred_lat
        a = sin(dlat/2)**2 + cos(pred_lat) * cos(obs_lat) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radio de la Tierra en km
        r = 6371
        
        return r * c
    
    def _calculate_intensity_error(self, prediction, observation):
        """Calcula errores de intensidad (número de rayos)."""
        pred_intensity = prediction.get('pred_n_flashes', 0)
        obs_intensity = observation.get('n_flashes', 0)
        
        absolute_error = abs(pred_intensity - obs_intensity)
        
        if obs_intensity > 0:
            percentage_error = (absolute_error / obs_intensity) * 100
            bias = ((pred_intensity - obs_intensity) / obs_intensity) * 100
        else:
            percentage_error = 100 if pred_intensity > 0 else 0
            bias = 100 if pred_intensity > 0 else 0
        
        return {
            'absolute': absolute_error,
            'percentage': percentage_error,
            'bias': bias  # Positivo = sobreestimación, Negativo = subestimación
        }
    
    def _calculate_area_error(self, prediction, observation):
        """Calcula errores de área."""
        pred_area = prediction.get('pred_area', 0)
        obs_area = observation.get('area_km2', 0)
        
        absolute_error = abs(pred_area - obs_area)
        
        if obs_area > 0:
            percentage_error = (absolute_error / obs_area) * 100
            bias = ((pred_area - obs_area) / obs_area) * 100
        else:
            percentage_error = 100 if pred_area > 0 else 0
            bias = 100 if pred_area > 0 else 0
        
        return {
            'absolute': absolute_error,
            'percentage': percentage_error,
            'bias': bias
        }
    
    def _assess_confidence_calibration(self, predicted_confidence, was_accurate):
        """
        Evalúa si la confianza predicha está bien calibrada.
        
        Una predicción está bien calibrada si:
        - Alta confianza (>70%) → La predicción fue correcta
        - Baja confianza (<50%) → La predicción fue incorrecta
        - Confianza media (50-70%) → Ambos casos son aceptables
        """
        if predicted_confidence > 70:
            return was_accurate  # Debería ser correcta
        elif predicted_confidence < 50:
            return not was_accurate  # Debería ser incorrecta
        else:
            return True  # Confianza media - ambos casos aceptables
    
    def _update_performance_statistics(self, verification_result):
        """Actualiza las estadísticas de rendimiento del sistema."""
        # Agregar errores a las listas históricas
        self.performance_stats['position_errors'].append(
            verification_result['position_error_km']
        )
        self.performance_stats['intensity_errors'].append(
            verification_result['intensity_error_percentage']
        )
        
        # Actualizar rendimiento por método
        method = verification_result['forecast_method']
        if method not in self.performance_stats['method_performance']:
            self.performance_stats['method_performance'][method] = {
                'count': 0,
                'position_errors': [],
                'success_rate': 0,
                'avg_confidence': 0
            }
        
        method_stats = self.performance_stats['method_performance'][method]
        method_stats['count'] += 1
        method_stats['position_errors'].append(verification_result['position_error_km'])
        method_stats['success_rate'] = np.mean([
            v['within_uncertainty'] for v in self.verification_history 
            if v['forecast_method'] == method
        ])
        method_stats['avg_confidence'] = np.mean([
            v['predicted_confidence'] for v in self.verification_history 
            if v['forecast_method'] == method
        ])
    
    def get_performance_summary(self):
        """
        Genera un resumen completo del rendimiento del sistema.
        
        Returns:
            dict: Resumen de métricas de rendimiento
        """
        if not self.verification_history:
            return {
                'message': 'No hay verificaciones disponibles',
                'total_verifications': 0
            }
        
        # Estadísticas generales
        position_errors = self.performance_stats['position_errors']
        intensity_errors = self.performance_stats['intensity_errors']
        
        within_uncertainty_count = sum(1 for v in self.verification_history if v['within_uncertainty'])
        total_verifications = len(self.verification_history)
        
        # Métricas por intervalo de confianza
        confidence_bins = [(0, 50), (50, 70), (70, 85), (85, 100)]
        confidence_analysis = {}
        
        for low, high in confidence_bins:
            bin_verifications = [
                v for v in self.verification_history 
                if low <= v['predicted_confidence'] < high
            ]
            
            if bin_verifications:
                success_rate = np.mean([v['within_uncertainty'] for v in bin_verifications])
                avg_error = np.mean([v['position_error_km'] for v in bin_verifications])
                
                confidence_analysis[f'{low}-{high}%'] = {
                    'count': len(bin_verifications),
                    'success_rate': success_rate * 100,
                    'avg_position_error': avg_error,
                    'well_calibrated': abs(success_rate * 100 - (low + high) / 2) < 15
                }
        
        # Análisis por método de pronóstico
        method_analysis = {}
        for method, stats in self.performance_stats['method_performance'].items():
            if stats['position_errors']:
                method_analysis[method] = {
                    'count': stats['count'],
                    'success_rate': stats['success_rate'] * 100,
                    'avg_position_error': np.mean(stats['position_errors']),
                    'std_position_error': np.std(stats['position_errors']),
                    'avg_confidence': stats['avg_confidence']
                }
        
        # Tendencias temporales (últimas 10 verificaciones vs primeras 10)
        temporal_analysis = {}
        if len(self.verification_history) >= 20:
            recent_verifications = self.verification_history[-10:]
            early_verifications = self.verification_history[:10]
            
            recent_success_rate = np.mean([v['within_uncertainty'] for v in recent_verifications])
            early_success_rate = np.mean([v['within_uncertainty'] for v in early_verifications])
            
            recent_avg_error = np.mean([v['position_error_km'] for v in recent_verifications])
            early_avg_error = np.mean([v['position_error_km'] for v in early_verifications])
            
            temporal_analysis = {
                'improvement_in_success_rate': (recent_success_rate - early_success_rate) * 100,
                'improvement_in_accuracy': early_avg_error - recent_avg_error,
                'learning_trend': 'improving' if recent_success_rate > early_success_rate else 'declining'
            }
        
        return {
            'total_verifications': total_verifications,
            'overall_success_rate': (within_uncertainty_count / total_verifications) * 100,
            
            'position_accuracy': {
                'mean_error_km': np.mean(position_errors),
                'median_error_km': np.median(position_errors),
                'std_error_km': np.std(position_errors),
                'percentile_90_km': np.percentile(position_errors, 90),
                'best_error_km': np.min(position_errors),
                'worst_error_km': np.max(position_errors)
            },
            
            'intensity_accuracy': {
                'mean_error_pct': np.mean(intensity_errors),
                'median_error_pct': np.median(intensity_errors),
                'std_error_pct': np.std(intensity_errors)
            },
            
            'confidence_calibration': confidence_analysis,
            'method_performance': method_analysis,
            'temporal_trends': temporal_analysis,
            
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self):
        """Genera recomendaciones basadas en el análisis de rendimiento."""
        recommendations = []
        
        if not self.verification_history:
            return ["Necesita más datos para generar recomendaciones"]
        
        # Analizar rendimiento general
        success_rate = np.mean([v['within_uncertainty'] for v in self.verification_history]) * 100
        avg_error = np.mean(self.performance_stats['position_errors'])
        
        if success_rate < 60:
            recommendations.append(
                "⚠️ Tasa de acierto baja (<60%). Considerar ajustar parámetros de incertidumbre."
            )
        elif success_rate > 85:
            recommendations.append(
                "✅ Excelente tasa de acierto (>85%). El sistema está bien calibrado."
            )
        
        if avg_error > 15:
            recommendations.append(
                "📏 Error promedio alto (>15km). Considerar mejorar modelos de predicción."
            )
        elif avg_error < 5:
            recommendations.append(
                "🎯 Error promedio excelente (<5km). Precisión muy buena."
            )
        
        # Analizar métodos
        method_performance = self.performance_stats['method_performance']
        if method_performance:
            best_method = max(
                method_performance.keys(),
                key=lambda m: method_performance[m]['success_rate']
            )
            worst_method = min(
                method_performance.keys(),
                key=lambda m: method_performance[m]['success_rate']
            )
            
            if len(method_performance) > 1:
                recommendations.append(
                    f"🏆 Mejor método: {best_method} "
                    f"({method_performance[best_method]['success_rate']*100:.1f}% éxito)"
                )
                recommendations.append(
                    f"⚠️ Método a mejorar: {worst_method} "
                    f"({method_performance[worst_method]['success_rate']*100:.1f}% éxito)"
                )
        
        # Analizar calibración de confianza
        high_conf_verifications = [
            v for v in self.verification_history if v['predicted_confidence'] > 70
        ]
        if high_conf_verifications:
            high_conf_success = np.mean([v['within_uncertainty'] for v in high_conf_verifications])
            if high_conf_success < 0.7:
                recommendations.append(
                    "🎲 Confianza alta mal calibrada. El sistema es demasiado optimista."
                )
        
        return recommendations
    
    def export_verification_report(self, filename=None):
        """
        Exporta un reporte detallado de verificaciones.
        
        Args:
            filename: Nombre del archivo (opcional)
            
        Returns:
            str: Ruta del archivo generado
        """
        if filename is None:
            filename = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_verifications': len(self.verification_history),
                'report_version': '1.0'
            },
            'performance_summary': self.get_performance_summary(),
            'detailed_verifications': self.verification_history,
            'performance_statistics': self.performance_stats
        }
        
        import json
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Reporte de verificación exportado a: {filename}")
        return filename


# Ejemplo de uso integrado en el sistema principal
class IntegratedVerificationExample:
    """
    Ejemplo de cómo integrar el sistema de verificación con el nowcasting.
    """
    
    def __init__(self):
        self.verification_system = PerformanceVerificationSystem()
        self.previous_predictions = {}  # Almacenar predicciones por track_id
    
    def process_new_window(self, current_cells, new_predictions, timestamp):
        """
        Procesa una nueva ventana temporal con verificación integrada.
        
        Args:
            current_cells: GeoDataFrame con celdas actuales
            new_predictions: DataFrame con nuevas predicciones
            timestamp: Timestamp de la ventana actual
        """
        # 1. Verificar predicciones anteriores
        verification_results = []
        
        for track_id, previous_pred in self.previous_predictions.items():
            # Buscar si el track sigue activo
            current_track_cells = current_cells[current_cells['track_id'] == track_id]
            
            if not current_track_cells.empty:
                actual_observation = current_track_cells.iloc[0].to_dict()
                actual_observation['timestamp'] = timestamp
                
                # Verificar predicción
                verification = self.verification_system.verify_prediction(
                    previous_pred, actual_observation
                )
                
                if verification:
                    verification_results.append(verification)
                    logger.info(
                        f"Track {track_id}: Error = {verification['position_error_km']:.1f}km, "
                        f"Dentro de incertidumbre = {verification['within_uncertainty']}"
                    )
        
        # 2. Almacenar nuevas predicciones para verificación futura
        self.previous_predictions.clear()
        for _, pred in new_predictions.iterrows():
            self.previous_predictions[pred['track_id']] = pred.to_dict()
        
        # 3. Obtener métricas actualizadas
        performance_summary = self.verification_system.get_performance_summary()
        
        return verification_results, performance_summary


# Funciones de utilidad para análisis de rendimiento
def calculate_skill_scores(verification_results):
    """
    Calcula scores de habilidad estándar para evaluación de pronósticos.
    
    Args:
        verification_results: Lista de resultados de verificación
        
    Returns:
        dict: Scores de habilidad calculados
    """
    if not verification_results:
        return {}
    
    position_errors = [v['position_error_km'] for v in verification_results]
    
    # Root Mean Square Error (RMSE)
    rmse = np.sqrt(np.mean(np.array(position_errors)**2))
    
    # Mean Absolute Error (MAE)
    mae = np.mean(position_errors)
    
    # Success Rate (dentro de zona de incertidumbre)
    success_rate = np.mean([v['within_uncertainty'] for v in verification_results])
    
    # Bias (sesgo promedio)
    intensity_biases = [v['intensity_bias'] for v in verification_results]
    avg_intensity_bias = np.mean(intensity_biases)
    
    return {
        'rmse_km': rmse,
        'mae_km': mae,
        'success_rate': success_rate,
        'avg_intensity_bias_pct': avg_intensity_bias,
        'sample_size': len(verification_results)
    }

def generate_performance_plots(verification_system, output_dir='./plots'):
    """
    Genera gráficos de rendimiento del sistema de verificación.
    
    Args:
        verification_system: Instancia de PerformanceVerificationSystem
        output_dir: Directorio para guardar los gráficos
    """
    try:
        import matplotlib.pyplot as plt
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        verification_history = verification_system.verification_history
        
        if not verification_history:
            logger.warning("No hay datos de verificación para graficar")
            return
        
        # Gráfico 1: Error de posición vs tiempo
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        timestamps = [v['observation_time'] for v in verification_history]
        position_errors = [v['position_error_km'] for v in verification_history]
        confidences = [v['predicted_confidence'] for v in verification_history]
        
        # Error vs tiempo
        ax1.scatter(timestamps, position_errors, alpha=0.6, c=confidences, cmap='RdYlGn')
        ax1.set_title('Error de Posición vs Tiempo')
        ax1.set_ylabel('Error (km)')
        ax1.grid(True, alpha=0.3)
        
        # Histograma de errores
        ax2.hist(position_errors, bins=20, alpha=0.7, edgecolor='black')
        ax2.set_title('Distribución de Errores de Posición')
        ax2.set_xlabel('Error (km)')
        ax2.set_ylabel('Frecuencia')
        ax2.grid(True, alpha=0.3)
        
        # Confianza vs precisión
        within_uncertainty = [v['within_uncertainty'] for v in verification_history]
        ax3.scatter(confidences, position_errors, c=within_uncertainty, cmap='RdYlGn', alpha=0.6)
        ax3.set_title('Confianza Predicha vs Error Real')
        ax3.set_xlabel('Confianza (%)')
        ax3.set_ylabel('Error (km)')
        ax3.grid(True, alpha=0.3)
        
        # Tasa de éxito por método
        methods = list(set(v['forecast_method'] for v in verification_history))
        method_success_rates = []
        
        for method in methods:
            method_verifications = [v for v in verification_history if v['forecast_method'] == method]
            success_rate = np.mean([v['within_uncertainty'] for v in method_verifications]) * 100
            method_success_rates.append(success_rate)
        
        ax4.bar(methods, method_success_rates, alpha=0.7, color=['blue', 'green', 'orange', 'red'][:len(methods)])
        ax4.set_title('Tasa de Éxito por Método')
        ax4.set_ylabel('Tasa de Éxito (%)')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'verification_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Gráficos de rendimiento guardados en: {plot_path}")
        return plot_path
        
    except ImportError:
        logger.warning("Matplotlib no disponible para generar gráficos")
        return None
    except Exception as e:
        logger.error(f"Error generando gráficos: {e}")
        return None