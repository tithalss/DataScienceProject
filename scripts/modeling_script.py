import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report


def run_modeling():
    df = pd.read_csv("processed_data/players_with_kpis.csv")

    regression_features = ["crossing", "finishing", "short_passing", "overall_kpi"]
    regression_target = "goals_per_match"

    df = df.dropna(subset=regression_features + [regression_target])

    X = df[regression_features]
    y = df[regression_target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred = linear_model.predict(X_test)

    print("[Regressão Linear]")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")
    print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}\n")

    df["high_scorer"] = (df["goals_per_match"] > 0.3).astype(int)
    logistic_features = regression_features
    logistic_target = "high_scorer"

    X = df[logistic_features]
    y = df[logistic_target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train, y_train)
    y_pred = logistic_model.predict(X_test)

    print("[Regressão Logística]")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_model.fit(X_train, y_train)
    y_pred_tree = tree_model.predict(X_test)

    print("[Árvore de Decisão]")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_tree):.4f}")

    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)

    print("[KNN]")
    print(f"Acurácia: {accuracy_score(y_test, y_pred_knn):.4f}")

    print("\nModelagem preditiva concluída!")
