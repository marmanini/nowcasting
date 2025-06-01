#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para evaluar el rendimiento del sistema de nowcasting.
"""

import os
import glob
import argparse
from src.evaluation.nowcasting_evaluator import evaluate_nowcasting_performance

def main():
    parser = argparse.ArgumentParser(description='Evalúa el rendimiento del sistema de nowcasting.')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--predictions-dir', type=str, default='predictions', 
                        help='Directorio con archivos CSV de predicciones')
    parser.add_argument('--output-dir', type=str, default='evaluation', 
                        help='Directorio para guardar resultados')
    
    args = parser.parse_args()
    
    # Buscar archivos de datos
    geojson_files = sorted(glob.glob(f"{args.geojson_dir}/cells_*.geojson"))
    prediction_files = sorted(glob.glob(f"{args.predictions_dir}/predictions_*.csv"))
    
    if not geojson_files or not prediction_files:
        print("No se encontraron archivos de datos o predicciones.")
        return
    
    print(f"Evaluando {len(prediction_files)} archivos de predicción...")
    
    # Ejecutar evaluación
    results = evaluate_nowcasting_performance(
        geojson_files,
        prediction_files,
        output_dir=args.output_dir
    )
    
    # Mostrar resultados
    if results and 'aggregated_metrics' in results:
        print("\n=== Resultados de la Evaluación ===")
        print(f"POD: {results['aggregated_metrics']['probability_of_detection']:.3f}")
        print(f"FAR: {results['aggregated_metrics']['false_alarm_ratio']:.3f}")
        print(f"CSI: {results['aggregated_metrics']['critical_success_index']:.3f}")
        print(f"Error medio de posición: {results['aggregated_metrics']['mean_position_error_km']:.2f} km")
        print(f"\nResultados guardados en: {args.output_dir}")
    else:
        print("No se pudieron generar métricas de evaluación.")

if __name__ == "__main__":
    main()