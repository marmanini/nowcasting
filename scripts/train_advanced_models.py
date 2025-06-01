#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para entrenar modelos avanzados de nowcasting.
"""

import os
import glob
import argparse
from src.models.advanced_nowcast_predictor import load_and_process_track_data, train_and_evaluate_models

def main():
    parser = argparse.ArgumentParser(description='Entrena modelos avanzados de nowcasting.')
    parser.add_argument('--geojson-dir', type=str, default='data', 
                        help='Directorio con archivos GeoJSON de celdas')
    parser.add_argument('--output-dir', type=str, default='models', 
                        help='Directorio para guardar modelos')
    
    args = parser.parse_args()
    
    # Buscar archivos de datos
    geojson_files = sorted(glob.glob(f"{args.geojson_dir}/cells_*.geojson"))
    
    if not geojson_files:
        print("No se encontraron archivos de celdas.")
        return
    
    print(f"Cargando datos de {len(geojson_files)} archivos...")
    
    # Cargar y procesar datos
    tracks_data = load_and_process_track_data(geojson_files)
    
    if not tracks_data:
        print("No se pudieron procesar datos de tracks.")
        return
    
    print(f"Datos procesados: {len(tracks_data)} tracks")
    print("Entrenando modelos...")
    
    # Entrenar y evaluar modelos
    results = train_and_evaluate_models(tracks_data, output_dir=args.output_dir)
    
    # Mostrar resultados
    if 'best_model' in results:
        best_model = results['best_model']
        metrics = results[best_model]['evaluation']['metrics']
        
        print(f"\n=== Mejor modelo: {best_model.upper()} ===")
        print(f"RMSE posicional: {metrics['position_rmse']:.2f} km")
        print(f"Error mediano: {metrics['position_median_error']:.2f} km")
        print(f"RMSE intensidad: {metrics['intensity_rmse']:.2f}")
        print(f"RMSE área: {metrics['area_rmse']:.2f} km²")
        print(f"\nModelos guardados en: {args.output_dir}")
    else:
        print("No se pudo determinar el mejor modelo.")

if __name__ == "__main__":
    main()