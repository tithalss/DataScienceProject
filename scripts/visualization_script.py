import pandas as pd
from visualization.plots import plot_top_players, plot_correlation_matrix, plot_distribution
from visualization.dashboard import dashboard_summary


def run_visualization():
    df = pd.read_csv('processed_data/players_with_kpis.csv')

    plot_top_players(df, 'overall_kpi', top_n=10)
    plot_distribution(df, 'goals_per_match')

    kpi_cols = [
        'goals_per_match', 'assists_per_match', 'passes_per_match',
        'tackles_per_match', 'pass_accuracy', 'speed', 'overall_kpi'
    ]

    plot_correlation_matrix(df, kpi_cols)
    dashboard_summary(df)
    print("📈 Visualizações geradas com sucesso!")
