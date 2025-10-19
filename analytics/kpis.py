def calculate_offensive_kpis(df):
    df['goals_per_match'] = df['gls'] / df['mp'].replace(0, 1)
    df['assists_per_match'] = df['ast'] / df['mp'].replace(0, 1)
    df['contrib_off'] = df['gls'] + df['ast']
    df['contrib_off_per_match'] = df['contrib_off'] / df['mp'].replace(0, 1)
    df['goal_conversion_rate'] = df['gls'] / df['shots'].replace(0, 1) if 'shots' in df.columns else 0
    df['shots_per_match'] = df['shots'] / df['mp'].replace(0, 1) if 'shots' in df.columns else 0
    return df


def calculate_defensive_kpis(df):
    df['tackles_per_match'] = (df['tkl'] + df['tkl+int']) / df['mp'].replace(0, 1)
    df['interceptions_per_match'] = df['int'] / df['mp'].replace(0, 1)
    df['clean_tackles'] = df['tklw'] / df['tkl'].replace(0, 1)
    df['duels_won_rate'] = df['tklw'] / (df['tklw'] + df['tkl_lost']).replace(0, 1) if 'tkl_lost' in df.columns else 0
    df['blocks_per_match'] = df['blocks'] / df['mp'].replace(0, 1) if 'blocks' in df.columns else 0
    return df


def calculate_passing_kpis(df):
    df['passes_per_match'] = df['att'] / df['mp'].replace(0, 1)
    df['key_passes_per_match'] = df['kp'] / df['mp'].replace(0, 1)
    df['pass_accuracy'] = df['cmp%'] / 100
    df['long_pass_accuracy'] = df['cmp_long'] / df['att_long'].replace(0, 1) if 'cmp_long' in df.columns else 0
    df['progressive_passes_per_match'] = df['prgP'] / df['mp'].replace(0, 1) if 'prgP' in df.columns else 0
    return df


def calculate_physical_kpis(df):
    df['speed'] = (df['sprint_speed'] + df['acceleration']) / 2
    df['stamina_value'] = df['stamina']
    df['strength_value'] = df['strength']
    df['agility_score'] = df['agility']
    df['physical_intensity'] = (df['stamina'] + df['strength'] + df['agility']) / 3
    return df


def calculate_goalkeeper_kpis(df):
    df['saves_per_match'] = df['saves'] / df['mp'].replace(0, 1) if 'saves' in df.columns else 0
    df['save_percentage'] = df['save%'] / 100 if 'save%' in df.columns else 0
    df['clean_sheets_per_match'] = df['cs'] / df['mp'].replace(0, 1) if 'cs' in df.columns else 0
    df['gk_distribution_accuracy'] = df['cmp_gk_passes'] / df['att_gk_passes'].replace(0, 1) if 'cmp_gk_passes' in df.columns else 0
    return df


def calculate_overall_kpi(df):
    df['overall_kpi'] = (
        df.get('contrib_off_per_match', 0) * 0.3 +
        df.get('tackles_per_match', 0) * 0.2 +
        df.get('pass_accuracy', 0) * 0.2 +
        df.get('speed', 0) * 0.1 +
        df.get('physical_intensity', 0) * 0.1 +
        df.get('duels_won_rate', 0) * 0.1
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
