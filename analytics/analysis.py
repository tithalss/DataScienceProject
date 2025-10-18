import pandas as pd

def descriptive_statistics(df):
    numeric_cols = df.select_dtypes(include='number').columns

    stats = pd.DataFrame(index=numeric_cols)
    stats['mean'] = df[numeric_cols].mean()
    stats['median'] = df[numeric_cols].median()
    stats['mode'] = df[numeric_cols].mode().iloc[0]
    stats['std'] = df[numeric_cols].std()
    stats['min'] = df[numeric_cols].min()
    stats['max'] = df[numeric_cols].max()
    return stats

def top_players_by_metric(df, metric, top_n=10):
    return df[['player', metric]].sort_values(by=metric, ascending=False).head(top_n)
