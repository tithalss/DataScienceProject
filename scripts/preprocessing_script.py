import pandas as pd
from treatment.load_dataset import load_all_datasets
from treatment.pre_processing import (standardize_columns, remove_duplicates, fill_missing, convert_types, merge_datasets)
from treatment.save_data import save_processed_data


def run_preprocessing():
    datasets = load_all_datasets()
    df1 = datasets['players_dataset_1']
    df2 = datasets['players_dataset_2']

    df1 = df1.rename(columns={'name': 'player'})
    df1 = standardize_columns(df1)
    df2 = standardize_columns(df2)

    df1 = remove_duplicates(df1)
    df2 = remove_duplicates(df2)

    df1 = fill_missing(df1)
    df2 = fill_missing(df2)

    types1 = {col: 'int' for col in ['age', 'crossing', 'finishing', 'heading_accuracy']}
    types2 = {col: 'int' for col in ['age', 'mp', 'gls', 'ast']}
    df1 = convert_types(df1, types1)
    df2 = convert_types(df2, types2)

    df_merged = merge_datasets(df1, df2, on_cols=['player'], how='inner')
    df_merged = df_merged.drop_duplicates(subset='player', keep='first')
    df_merged = df_merged.loc[:, ~df_merged.columns.duplicated()]

    numeric_cols = df_merged.select_dtypes(include='number').columns
    df_merged[numeric_cols] = df_merged[numeric_cols].astype(float)

    df_merged = pd.concat([
        df_merged,
        pd.DataFrame({
            'goals_per_match': df_merged['gls'] / df_merged['mp'].replace(0, 1),
            'assists_per_match': df_merged['ast'] / df_merged['mp'].replace(0, 1)
        })
    ], axis=1).copy()

    save_processed_data(df_merged, filename='players_merged.csv')
    print("Pré-processamento concluído! Arquivo salvo em processed_data/players_merged.csv")
