import plotly.express as px

def dashboard_summary(df):
    fig1 = px.bar(df.sort_values('overall_kpi', ascending=False).head(10),
                  x='player', y='overall_kpi', title='Top 10 Jogadores - KPI Geral')

    fig2 = px.scatter(df, x='goals_per_match', y='assists_per_match',
                      size='overall_kpi', color='best_position',
                      hover_name='player', title='Relação entre Gols, Assistências e KPI Geral')

    fig3 = px.box(df, y='overall_kpi', color='best_position', title='Distribuição do KPI por Posição')

    fig1.show()
    fig2.show()
    fig3.show()
