# src/data_analyzer.py
import pandas as pd
import numpy as np

def analyze_real_data_ranges():
    """Анализирует реальные данные и выводит статистику по признакам"""
    df_2024 = pd.read_csv("data/raex_2024.csv")
    
    stats = {}
    for feature in feature_order:
        if feature in df_2024.columns:
            stats[feature] = {
                'min': df_2024[feature].min(),
                'max': df_2024[feature].max(),
                'mean': df_2024[feature].mean(),
                'median': df_2024[feature].median(),
                'top_10_avg': df_2024[feature].head(10).mean(),
                'bottom_10_avg': df_2024[feature].tail(10).mean()
            }
    
    print("=== СТАТИСТИКА РЕАЛЬНЫХ ДАННЫХ ===")
    for feat, values in stats.items():
        print(f"\n{feat}:")
        print(f"  Min: {values['min']:.2f}, Max: {values['max']:.2f}")
        print(f"  Mean: {values['mean']:.2f}, Median: {values['median']:.2f}")
        print(f"  Топ-10: {values['top_10_avg']:.2f}, Слабые: {values['bottom_10_avg']:.2f}")
    
    return stats

if __name__ == "__main__":
    analyze_real_data_ranges()