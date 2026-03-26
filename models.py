import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                              recall_score, f1_score, roc_auc_score)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import joblib

from mlxtend.frequent_patterns import apriori, association_rules
from utils import preprocess, encode_for_model

RANDOM_STATE = 42


def train_classification(df, target_col="Interest_in_MoodCart"):
    df_p = preprocess(df)
    X, y = encode_for_model(df_p, target_col=target_col)

    # FIX: y can be None if target col missing — guard it
    if y is None:
        raise ValueError(f"Target column '{target_col}' not found in dataframe.")

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=RANDOM_STATE, stratify=y_enc
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTree":       DecisionTreeClassifier(random_state=RANDOM_STATE),
        "RandomForest":       RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "KNN":                KNeighborsClassifier(n_neighbors=5),
    }

    results = []
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # FIX: roc_auc only for binary; Interest_in_MoodCart has 3 classes → skip safely
        roc = None
        if hasattr(model, "predict_proba") and len(le.classes_) == 2:
            y_prob = model.predict_proba(X_test)[:, 1]
            roc = round(roc_auc_score(y_test, y_prob), 4)

        results.append({
            "Model":     name,
            "Accuracy":  round(acc, 4),
            "Precision": round(prec, 4),
            "Recall":    round(rec, 4),
            "F1 Score":  round(f1, 4),
            "ROC-AUC":   roc,
        })
        trained[name] = model

    best_name  = sorted(results, key=lambda x: x["F1 Score"], reverse=True)[0]["Model"]
    best_model = trained[best_name]

    return pd.DataFrame(results), best_model, le, X.columns


def train_regression(df, target_col="Monthly_Spend"):
    df_p = preprocess(df)
    # FIX: ensure target is numeric
    df_p[target_col] = pd.to_numeric(df_p[target_col], errors="coerce")
    df_p = df_p.dropna(subset=[target_col])

    X, y = encode_for_model(df_p, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    models = {
        "LinearRegression":      LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
    }

    scores  = {}
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        r2 = model.score(X_test, y_test)
        scores[name] = round(r2, 4)
        trained[name] = model

    best_name = max(scores, key=scores.get)
    return scores, trained[best_name]


def train_clustering(df, k=4):
    df_p = preprocess(df)
    # FIX: drop target col before clustering so it doesn't leak
    if "Interest_in_MoodCart" in df_p.columns:
        df_p = df_p.drop(columns=["Interest_in_MoodCart"])
    X, _ = encode_for_model(df_p, target_col=None)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    km     = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)
    return labels


def association_mining(df, min_support=0.05):
    if "Product_Combinations" not in df.columns:
        return pd.DataFrame()

    s = df["Product_Combinations"].fillna("").apply(
        lambda x: [t.strip() for t in str(x).split("|") if t.strip()]
    )
    items = sorted({i for sub in s for i in sub})
    if not items:
        return pd.DataFrame()

    # FIX: apriori needs boolean dtype, not int
    onehot = pd.DataFrame(False, index=df.index, columns=items)
    for idx, lst in s.items():
        for it in lst:
            if it in onehot.columns:
                onehot.at[idx, it] = True

    freq = apriori(onehot, min_support=min_support, use_colnames=True)
    if freq.empty:
        return pd.DataFrame()

    # FIX: mlxtend >= 0.21 requires num_itemsets argument
    try:
        rules = association_rules(freq, metric="confidence",
                                   min_threshold=0.3,
                                   num_itemsets=len(freq))
    except TypeError:
        # Older mlxtend — no num_itemsets param
        rules = association_rules(freq, metric="confidence", min_threshold=0.3)

    rules = rules.sort_values(by=["confidence", "lift"], ascending=False)
    # FIX: frozenset columns → readable strings for display
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
    return rules[["antecedents", "consequents", "support", "confidence", "lift"]].reset_index(drop=True)


def save_model(model, le, feature_cols, prefix="model"):
    joblib.dump(model,             f"{prefix}.joblib")
    joblib.dump(le,                f"{prefix}_le.joblib")
    joblib.dump(list(feature_cols), f"{prefix}_cols.joblib")


def load_model(prefix="model"):
    model = joblib.load(f"{prefix}.joblib")
    le    = joblib.load(f"{prefix}_le.joblib")
    cols  = joblib.load(f"{prefix}_cols.joblib")
    return model, le, cols


def predict_new(df_new, model, le, cols):
    df_p = preprocess(df_new)
    X, _  = encode_for_model(df_p, target_col=None)
    # FIX: align columns — add missing cols as 0, drop extra cols
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X     = X[cols]
    preds = model.predict(X)
    return le.inverse_transform(preds)
