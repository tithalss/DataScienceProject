import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from mlxtend.frequent_patterns import apriori, association_rules


def run_ml_analysis():
    df = pd.read_csv('processed_data/players_with_kpis.csv')

    kpi_cols = ['goals_per_match', 'assists_per_match', 'passes_per_match', 'tackles_per_match', 'pass_accuracy', 'speed', 'overall_kpi']

    X = df[kpi_cols].fillna(0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(X_scaled).astype(str)
    print("K-Means cluster distribution:")
    print(df['kmeans_cluster'].value_counts())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['goals_per_match'], y=df['assists_per_match'],
                    hue=df['kmeans_cluster'], palette='tab10')
    plt.title('K-Means Clusters: Gols x Assistências')
    plt.show()

    dbscan = DBSCAN(eps=1.5, min_samples=5)
    df['dbscan_cluster'] = dbscan.fit_predict(X_scaled).astype(str)
    print("DBSCAN cluster distribution (-1 = outlier):")
    print(df['dbscan_cluster'].value_counts())

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df['speed'], y=df['overall_kpi'],
                    hue=df['dbscan_cluster'], palette='tab10')
    plt.title('DBSCAN Clusters: Velocidade x Overall KPI')
    plt.show()

    df_bin = pd.DataFrame({
        'high_goals': df['goals_per_match'] > 0.3,
        'high_assists': df['assists_per_match'] > 0.2,
        'high_pass_accuracy': df['pass_accuracy'] > 0.8,
        'high_speed': df['speed'] > 70,
        'high_overall': df['overall_kpi'] > df['overall_kpi'].median()
    })

    df_bin = df_bin.astype(bool)
    frequent_itemsets = apriori(df_bin, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    print("\nApriori Association Rules:")
    print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    plt.figure(figsize=(8, 6))
    sns.heatmap(df_bin.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlação entre indicadores binários')
    plt.show()

    return df
