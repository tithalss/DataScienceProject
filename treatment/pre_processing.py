import pandas as pd

def standardize_columns(df):
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
    return df

def remove_duplicates(df):
    return df.drop_duplicates()

def fill_missing(df, value=0):
    return df.fillna(value)

def convert_types(df, columns_types):
    for col, dtype in columns_types.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df

def merge_datasets(df1, df2, on_cols, how='outer'):
    on_cols = [col.lower() for col in on_cols]
    return pd.merge(df1, df2, on=on_cols, how=how)
