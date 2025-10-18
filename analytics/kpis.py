def calculate_offensive_kpis(df):
    df['goals_per_match'] = df['gls'] / df['mp'].replace(0, 1)
    df['assists_per_match'] = df['ast'] / df['mp'].replace(0, 1)
    df['contrib_off'] = df['gls'] + df['ast']
    df['contrib_off_per_match'] = df['contrib_off'] / df['mp'].replace(0, 1)
    return df

def calculate_defensive_kpis(df):
    df['tackles_per_match'] = (df['tkl'] + df['tkl+int']) / df['mp'].replace(0, 1)
    df['interceptions_per_match'] = df['int'] / df['mp'].replace(0, 1)
    df['clean_tackles'] = df['tklw'] / df['tkl'].replace(0, 1)
    return df

def calculate_passing_kpis(df):
    df['passes_per_match'] = df['att'] / df['mp'].replace(0, 1)
    df['key_passes_per_match'] = df['kp'] / df['mp'].replace(0, 1)
    df['pass_accuracy'] = df['cmp%'] / 100
    return df

def calculate_physical_kpis(df):
    df['speed'] = (df['sprint_speed'] + df['acceleration']) / 2
    df['stamina_value'] = df['stamina']
    df['strength_value'] = df['strength']
    return df

def calculate_goalkeeper_kpis(df):
    df['saves_per_match'] = df['saves'] / df['mp'].replace(0, 1)
    df['save_percentage'] = df['save%'] / 100
    return df

def calculate_overall_kpi(df):
    df['overall_kpi'] = (
        df['contrib_off_per_match'] * 0.4 +
        df['tackles_per_match'] * 0.3 +
        df['pass_accuracy'] * 0.3
    )
    return df

def calculate_all_kpis(df):
    df = calculate_offensive_kpis(df)
    df = calculate_defensive_kpis(df)
    df = calculate_passing_kpis(df)
    df = calculate_physical_kpis(df)
    df = calculate_goalkeeper_kpis(df)
    df = calculate_overall_kpi(df)
    return df

def top_players_by_kpi(df, kpi='overall_kpi', top_n=10):
    return df.sort_values(by=kpi, ascending=False).head(top_n)
