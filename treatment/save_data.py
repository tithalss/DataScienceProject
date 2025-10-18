import pandas as pd
import os

def save_processed_data(df, filename='processed_data_players.csv'):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    output_dir = os.path.join(base_dir, 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    df.to_csv(output_path, index=False)
    print(f"✅ Dados salvos em: {output_path}")
