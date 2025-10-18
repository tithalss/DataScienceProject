import pandas as pd
from treatment.load_dataset import load_all_datasets
from treatment.save_data import save_processed_data
from treatment.pre_processing import standardize_columns, remove_duplicates, fill_missing, convert_types, merge_datasets
from analytics.analysis import top_players_by_metric
from analytics.kpis import calculate_all_kpis, top_players_by_kpi

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

df_merged['goals_per_match'] = df_merged['gls'] / df_merged['mp'].replace(0, 1)
df_merged['assists_per_match'] = df_merged['ast'] / df_merged['mp'].replace(0, 1)

save_processed_data(df_merged, filename='players_merged.csv')

columns_of_interest = [
    'age_x', 'crossing', 'finishing', 'heading_accuracy', 'short_passing',
    'volleys', 'dribbling', 'curve', 'fk_accuracy', 'long_passing',
    'ball_control', 'acceleration', 'sprint_speed', 'agility', 'reactions',
    'balance', 'shot_power', 'jumping', 'stamina', 'strength',
    'long_shots', 'aggression', 'interceptions', 'positioning', 'vision',
    'penalties', 'composure', 'defensive_awareness', 'standing_tackle',
    'slide_tackle', 'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
    'gk_reflexes', 'best_overall_rating', 'mp', 'gls', 'ast', 'goals_per_match',
    'assists_per_match'
]
columns_of_interest = [col for col in columns_of_interest if col in df_merged.columns]

df_numeric = df_merged[columns_of_interest]

stats = pd.DataFrame(index=df_numeric.columns)
stats['mean'] = df_numeric.mean()
stats['median'] = df_numeric.median()
stats['mode'] = df_numeric.mode().iloc[0]
stats['std'] = df_numeric.std()
stats['min'] = df_numeric.min()
stats['max'] = df_numeric.max()

print("Estatísticas descritivas e métricas básicas:")
print(stats)

top_scorers = top_players_by_metric(df_merged, 'gls', top_n=10)
top_assisters = top_players_by_metric(df_merged, 'ast', top_n=10)

print("\nTop 10 jogadores por gols:")
print(top_scorers)

print("\nTop 10 jogadores por assistências:")
print(top_assisters)

df = pd.read_csv('processed_data/players_merged.csv')

df = calculate_all_kpis(df)
save_processed_data(df, 'players_with_kpis.csv')

top_overall = top_players_by_kpi(df, kpi='overall_kpi', top_n=10)
print(top_overall[['player', 'overall_kpi']])