import pandas as pd
import os

def load_dataset(filename: str):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, 'data', filename)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {data_path}")
    if filename == 'players_dataset_1.csv':
        df = pd.read_csv(data_path, sep=';')
    else:
        df = pd.read_csv(data_path)
    return df


def load_all_datasets():
    datasets = {
        "players_dataset_1": load_dataset("players_dataset_1.csv"),
        "players_dataset_2": load_dataset("players_dataset_2.csv")
    }
    return datasets
