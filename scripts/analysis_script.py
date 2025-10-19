import pandas as pd
from analytics.analysis import top_players_by_metric
from analytics.kpis import calculate_all_kpis, top_players_by_kpi
from treatment.save_data import save_processed_data


def run_analysis():
    df = pd.read_csv('processed_data/players_merged.csv')

    columns_of_interest = [col for col in df.columns if df[col].dtype != 'object']
    df_numeric = df[columns_of_interest]

    stats = pd.DataFrame(index=df_numeric.columns)
    stats['mean'] = df_numeric.mean()
    stats['median'] = df_numeric.median()
    stats['mode'] = df_numeric.mode().iloc[0]
    stats['std'] = df_numeric.std()
    stats['min'] = df_numeric.min()
    stats['max'] = df_numeric.max()

    print("📊 Estatísticas descritivas:")
    print(stats)

    top_scorers = top_players_by_metric(df, 'gls', top_n=10)
    top_assisters = top_players_by_metric(df, 'ast', top_n=10)

    print("\n⚽ Top 10 jogadores por gols:")
    print(top_scorers[['player', 'gls']])

    print("\n🎯 Top 10 jogadores por assistências:")
    print(top_assisters[['player', 'ast']])

    df = calculate_all_kpis(df)
    save_processed_data(df, 'players_with_kpis.csv')

    top_overall = top_players_by_kpi(df, kpi='overall_kpi', top_n=10)
    print("\n🏆 Top 10 jogadores por Overall KPI:")
    print(top_overall[['player', 'overall_kpi']])
