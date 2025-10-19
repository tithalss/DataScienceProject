import matplotlib.pyplot as plt
import seaborn as sns

def plot_top_players(df, metric, top_n=10):
    top = df.sort_values(by=metric, ascending=False).head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='overall_kpi', y='player', data=df, hue='player', palette='viridis', legend=False)
    plt.title(f'Top {top_n} Jogadores por {metric}')
    plt.xlabel(metric.replace('_', ' ').title())
    plt.ylabel('Jogador')
    plt.tight_layout()
    plt.show()

def plot_correlation_matrix(df, cols):
    plt.figure(figsize=(12, 8))
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de Correlação dos KPIs')
    plt.tight_layout()
    plt.show()

def plot_distribution(df, col):
    plt.figure(figsize=(8, 5))
    sns.histplot(df[col], bins=20, kde=True, color='steelblue')
    plt.title(f'Distribuição de {col}')
    plt.xlabel(col.replace('_', ' ').title())
    plt.ylabel('Frequência')
    plt.tight_layout()
    plt.show()
