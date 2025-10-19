## Link Datasets

- https://www.kaggle.com/datasets/cihan063/soccer-players
- https://www.kaggle.com/datasets/hubertsidorowicz/football-players-stats-2024-2025
---

## Funcionalidades

1. **Coleta de Dados**
   - Importação e integração de múltiplas fontes de dados de jogadores.
   - Dados brutos: atributos físicos, estatísticas de jogos, gols, assistências etc.

2. **Limpeza e Pré-processamento**
   - Remoção de valores nulos ou inconsistentes.
   - Transformação e normalização de colunas.
   - Criação de indicadores binários (ex: `high_overall`).

3. **Estatísticas Descritivas**
   - Cálculo de média, mediana, moda e desvio padrão.
   - Identificação de outliers e análise inicial do dataset.

4. **Criação de KPIs**
   - Indicadores como `goals_per_match`, `assists_per_match` e `overall_kpi`.
   - Permite comparar jogadores e gerar insights estratégicos.

5. **Visualização de Dados**
   - Gráficos de dispersão, histogramas, heatmaps e dashboards.
   - Storytelling visual para entender padrões e distribuições.

6. **Feature Engineering**
   - Criação de novas features a partir de KPIs.
   - Normalização e transformação binária de colunas importantes.

7. **Modelagem Preditiva**
   - **Regressão Linear:** prevê valores numéricos (ex: gols por partida).  
     - Métricas: R² e MSE.
   - **Regressão Logística:** classifica jogadores em categorias binárias (ex: high scorer).  
     - Métricas: Acurácia, Precisão, Recall, F1-Score.
   - **Árvores de Decisão e KNN:** classificação de jogadores com base em KPIs.

8. **Algoritmos de Machine Learning**
   - **Clusterização:** K-Means e DBSCAN para agrupar jogadores com perfis semelhantes.
   - **Regras de Associação (Apriori):** identificação de padrões frequentes entre KPIs.

9. **Métricas de Avaliação**
   - Precisão, Recall, F1-Score, matriz de confusão e validação cruzada.
   - Monitoramento de overfitting e performance dos modelos.
---