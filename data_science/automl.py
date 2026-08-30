import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

def detect_task_type(df, target_col):
    unique_vals = df[target_col].nunique()
    if df[target_col].dtype == "object" or unique_vals <= 10:
        return "classification"
    return "regression"

def prepare_features(df, target_col):
    df = df.copy()
    df = df.dropna(subset=[target_col])

    # Drop columns that are clearly not useful as features (too many unique values, likely IDs/names/free text)
    drop_cols = []
    id_like_cols = [c for c in df.columns if c.lower() in ["id", "passengerid", "index", "unnamed: 0"]]
    for col in df.columns:
        if col == target_col:
            continue
        if col in id_like_cols:
            drop_cols.append(col)
        elif df[col].dtype == "object" and df[col].nunique() > 50:
            drop_cols.append(col)
    df = df.drop(columns=drop_cols)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Fill missing numeric values with median, categorical with mode
    for col in X.columns:
        if X[col].dtype in ["int64", "float64"]:
            X[col] = X[col].fillna(X[col].median())
        else:
            X[col] = X[col].fillna(X[col].mode()[0] if not X[col].mode().empty else "unknown")

    # Encode categorical columns
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))

    if y.dtype == "object":
        le_target = LabelEncoder()
        y = le_target.fit_transform(y.astype(str))

    return X, y, drop_cols

def run_automl(csv_path, target_col):
    df = pd.read_csv(csv_path)

    if target_col not in df.columns:
        return {"error": f"Column '{target_col}' not found. Available columns: {list(df.columns)}"}

    task_type = detect_task_type(df, target_col)
    X, y, dropped_cols = prepare_features(df, target_col)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if task_type == "classification":
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 3),
            "f1_score": round(f1_score(y_test, preds, average="weighted"), 3)
        }
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics = {
            "rmse": round(mean_squared_error(y_test, preds) ** 0.5, 3),
            "r2_score": round(r2_score(y_test, preds), 3)
        }

    feature_importance = dict(zip(X.columns, model.feature_importances_.round(3)))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

    return {
        "task_type": task_type,
        "target_column": target_col,
        "dropped_columns": dropped_cols,
        "metrics": metrics,
        "top_features": dict(list(feature_importance.items())[:5])
    }

if __name__ == "__main__":
    result = run_automl("data/titanic.csv", "Survived")
    print(f"Task type: {result['task_type']}")
    print(f"Dropped columns (too many unique values): {result['dropped_columns']}")
    print(f"Metrics: {result['metrics']}")
    print(f"Top 5 important features: {result['top_features']}")