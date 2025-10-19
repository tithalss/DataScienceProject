import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from learning.ml_metrics import calculate_classification_metrics, generate_confusion_matrix, run_cross_validation

def run_model_evaluation():
    df = pd.read_csv('processed_data/players_with_kpis.csv')

    median_finishing = df['finishing'].median()
    df['high_finishing'] = (df['finishing'] > median_finishing).astype(int)
    target_col = 'high_finishing'

    feature_cols = [
        'goals_per_match', 'assists_per_match', 'passes_per_match',
        'tackles_per_match', 'pass_accuracy', 'speed', 'overall_kpi'
    ]

    X = df[feature_cols].fillna(0)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision, recall, f1 = calculate_classification_metrics(y_test, y_pred)

    cm, report = generate_confusion_matrix(y_test, y_pred)

    cv_accuracy = run_cross_validation(model, X, y, cv_folds=5)

    print("✅ Métricas de Classificação:")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}\n")
    print("✅ Matriz de Confusão:")
    print(cm)
    print("\nRelatório de Classificação:\n", report)
    print(f"✅ Validação Cruzada (Acurácia média): {cv_accuracy:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'classification_report': report,
        'cv_accuracy': cv_accuracy
    }
