#!/usr/bin/env python3
"""
Feature Engineering and Selection on the Titanic dataset.

What this script does
1. Loads the raw data (data/titanic.csv, or downloads it if missing)
2. Cleans it (fills missing values)
3. Engineers new features and encodes categorical variables
4. Splits into train and test sets (80/20, fixed random state)
5. Scales continuous features (scaler fitted on the training set only)
6. Selects features with six statistical / model-based methods and a vote
7. Compares models on baseline, all-engineered and selected feature sets
8. Saves the selected-feature dataset, charts and an auto-generated report

Run from anywhere:   python src/feature_engineering.py
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import (VarianceThreshold, f_classif, mutual_info_classif,
                                       chi2, RFE, SelectFromModel)
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_STATE = 42
TEST_SIZE = 0.2
VAR_THRESHOLD = 0.01     # drop features that are almost constant
CORR_THRESHOLD = 0.90    # drop one of two features correlated above this
TOP_K = 10               # each selection method picks its top K features
MIN_VOTES = 3            # keep features chosen by at least this many methods
URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

BASELINE_COLS = ["Pclass", "Sex_female", "Age", "SibSp", "Parch", "Fare", "Embarked_C", "Embarked_Q"]
CONTINUOUS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "TicketGroupSize",
              "FarePerPerson", "LogFare", "AgeXClass", "NameLength"]

NEW_FEATURES = {
    "Title_*": ("Name", "Title (Mr, Mrs, Miss, Master, Rare) extracted from the name; captures sex, age and social status together. One-hot encoded with Mr as the baseline."),
    "FamilySize": ("SibSp, Parch", "SibSp + Parch + 1: people travelling together."),
    "IsAlone": ("FamilySize", "1 if the passenger travelled alone."),
    "HasCabin": ("Cabin", "1 if a cabin number was recorded (Cabin is mostly missing, so the flag is more useful than the value)."),
    "Deck_*": ("Cabin", "Deck letter from the cabin number, one-hot encoded with 'unknown' as the baseline."),
    "TicketGroupSize": ("Ticket", "Number of passengers sharing the same ticket."),
    "FarePerPerson": ("Fare, TicketGroupSize", "Fare divided by ticket group size, because a ticket fare can cover a whole group."),
    "LogFare": ("Fare", "log(1 + Fare) to reduce the strong right skew."),
    "IsChild": ("Age", "1 if age is under 13."),
    "AgeXClass": ("Age, Pclass", "Interaction: Age multiplied by Pclass."),
    "NameLength": ("Name", "Number of characters in the name."),
    "Sex_female": ("Sex", "Binary encoding of Sex (1 = female)."),
    "Embarked_*": ("Embarked", "One-hot encoded port of embarkation with Southampton (S) as the baseline."),
}


# ----------------------------------------------------------------------------- loading and cleaning
def load_data(root):
    path = root / "data" / "titanic.csv"
    if path.exists():
        return pd.read_csv(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(URL)
    df.to_csv(path, index=False)
    return df


def clean_data(raw):
    """Fill missing values (same approach as the earlier cleaning task)."""
    df = raw.copy()
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    df["Fare"] = df["Fare"].fillna(df.groupby("Pclass")["Fare"].transform("median"))
    df["Age"] = df["Age"].fillna(df.groupby(["Pclass", "Sex"])["Age"].transform("median"))
    df["Age"] = df["Age"].fillna(df["Age"].median())
    return df


# ----------------------------------------------------------------------------- feature engineering
def engineer_features(df):
    """Create new features and encode categorical variables. Returns (X, y)."""
    out = pd.DataFrame(index=df.index)
    for col in ["Pclass", "Age", "SibSp", "Parch", "Fare"]:
        out[col] = df[col]
    out["Sex_female"] = (df["Sex"].str.strip().str.lower() == "female").astype(int)

    title = df["Name"].str.extract(r",\s*([^\.]+)\.", expand=False).str.strip()
    title = title.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
    title = title.where(title.isin(["Mr", "Mrs", "Miss", "Master"]), "Rare").fillna("Rare")
    deck = df["Cabin"].str[0].fillna("U")

    out["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    out["IsAlone"] = (out["FamilySize"] == 1).astype(int)
    out["HasCabin"] = df["Cabin"].notna().astype(int)
    out["TicketGroupSize"] = df.groupby("Ticket")["Ticket"].transform("count")
    out["FarePerPerson"] = df["Fare"] / out["TicketGroupSize"]
    out["LogFare"] = np.log1p(df["Fare"])
    out["IsChild"] = (df["Age"] < 13).astype(int)
    out["AgeXClass"] = df["Age"] * df["Pclass"]
    out["NameLength"] = df["Name"].str.len()

    dummies = pd.concat([pd.get_dummies(df["Embarked"], prefix="Embarked"),
                         pd.get_dummies(title, prefix="Title"),
                         pd.get_dummies(deck, prefix="Deck")], axis=1).astype(int)
    dummies = dummies.drop(columns=["Embarked_S", "Title_Mr", "Deck_U"], errors="ignore")
    X = pd.concat([out, dummies], axis=1)
    return X, df["Survived"]


# ----------------------------------------------------------------------------- feature selection
def variance_filter(X_train):
    vt = VarianceThreshold(VAR_THRESHOLD).fit(X_train)
    dropped = list(X_train.columns[~vt.get_support()])
    return [c for c in X_train.columns if c not in dropped], dropped


def correlation_filter(X_train, y_train, cols):
    """For each pair correlated above the threshold, drop the one less related to the target."""
    corr = X_train[cols].corr().abs()
    target_corr = X_train[cols].corrwith(y_train).abs()
    dropped = {}
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            if a in dropped or b in dropped:
                continue
            if corr.loc[a, b] > CORR_THRESHOLD:
                loser, winner = (a, b) if target_corr[a] < target_corr[b] else (b, a)
                dropped[loser] = (winner, float(corr.loc[a, b]))
    kept = [c for c in cols if c not in dropped]
    return kept, dropped


def select_features(X_train_scaled, X_train_raw, y_train, cols):
    """Six methods each pick their top K features; features get one vote per method."""
    Xs = X_train_scaled[cols]
    scores = pd.DataFrame(index=cols)

    f_vals, _ = f_classif(Xs, y_train)
    scores["ANOVA_F"] = f_vals

    binary = [c in cols and set(X_train_raw[c].unique()) <= {0, 1} for c in cols]
    scores["MutualInfo"] = mutual_info_classif(Xs, y_train, discrete_features=binary, random_state=RANDOM_STATE)

    X_pos = pd.DataFrame(MinMaxScaler().fit_transform(X_train_raw[cols]), columns=cols, index=X_train_raw.index)
    chi_vals, _ = chi2(X_pos, y_train)
    scores["Chi2"] = chi_vals

    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE).fit(Xs, y_train)
    scores["RF_Importance"] = rf.feature_importances_

    k = min(TOP_K, len(cols))
    rfe = RFE(LogisticRegression(max_iter=1000), n_features_to_select=k).fit(Xs, y_train)
    lasso = SelectFromModel(LassoCV(cv=5, random_state=RANDOM_STATE), threshold=-np.inf, max_features=k).fit(Xs, y_train)

    votes = pd.DataFrame(index=cols)
    for name in ["ANOVA_F", "MutualInfo", "Chi2", "RF_Importance"]:
        votes[name] = 0
        votes.loc[scores[name].sort_values(ascending=False).head(k).index, name] = 1
    votes["RFE_LogReg"] = rfe.support_.astype(int)
    votes["Lasso"] = lasso.get_support().astype(int)
    votes["Votes"] = votes.sum(axis=1)
    votes["RF_Importance_value"] = scores["RF_Importance"]
    votes = votes.sort_values(["Votes", "RF_Importance_value"], ascending=False)
    selected = list(votes.index[votes["Votes"] >= MIN_VOTES])
    return votes, scores, selected


# ----------------------------------------------------------------------------- evaluation
def evaluate(feature_sets, Xtr, Xte, ytr, yte):
    cv = StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE)
    models = {"Logistic Regression": lambda: LogisticRegression(max_iter=1000),
              "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)}
    rows = []
    for set_name, cols in feature_sets.items():
        for m_name, make in models.items():
            cv_scores = cross_val_score(make(), Xtr[cols], ytr, cv=cv, scoring="accuracy")
            model = make().fit(Xtr[cols], ytr)
            pred, proba = model.predict(Xte[cols]), model.predict_proba(Xte[cols])[:, 1]
            rows.append({"Feature set": set_name, "Features": len(cols), "Model": m_name,
                         "CV accuracy": cv_scores.mean(), "CV std": cv_scores.std(),
                         "Test accuracy": accuracy_score(yte, pred), "Test F1": f1_score(yte, pred),
                         "Test ROC AUC": roc_auc_score(yte, proba)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- charts
def make_charts(votes, selected, X_train_raw, results, img_dir):
    img_dir.mkdir(parents=True, exist_ok=True)
    # 1. votes
    v = votes.sort_values(["Votes", "RF_Importance_value"])
    fig, ax = plt.subplots(figsize=(8, max(5, 0.28 * len(v))))
    ax.barh(v.index, v["Votes"], color=["#d6432f" if f in selected else "#9a9a9a" for f in v.index])
    ax.axvline(MIN_VOTES - 0.5, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Votes (number of selection methods that chose the feature, out of 6)")
    ax.set_ylabel("Feature")
    ax.set_title("Feature selection votes")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#d6432f", label=f"Selected (at least {MIN_VOTES} votes)"),
                       Patch(color="#9a9a9a", label="Not selected")], loc="lower right")
    plt.tight_layout(); plt.savefig(img_dir / "feature_votes.png", dpi=150); plt.close()

    # 2. correlation heatmap of selected features
    corr = X_train_raw[selected].corr()
    fig, ax = plt.subplots(figsize=(0.55 * len(selected) + 4, 0.55 * len(selected) + 3))
    im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(selected))); ax.set_xticklabels(selected, rotation=90)
    ax.set_yticks(range(len(selected))); ax.set_yticklabels(selected)
    plt.colorbar(im, label="Correlation")
    ax.set_title("Correlation between selected features")
    plt.tight_layout(); plt.savefig(img_dir / "selected_correlation.png", dpi=150); plt.close()

    # 3. model comparison
    pivot = results.pivot(index="Feature set", columns="Model", values="Test accuracy")
    pivot = pivot.loc[results["Feature set"].drop_duplicates()]
    ax = pivot.plot(kind="bar", figsize=(8, 5), color=["#2a7fbf", "#d6432f"], rot=15)
    ax.set_ylabel("Test accuracy"); ax.set_xlabel("Feature set")
    ax.set_ylim(max(0, pivot.min().min() - 0.1), min(1, pivot.max().max() + 0.05))
    ax.set_title("Test accuracy by feature set and model")
    ax.legend(title="Model")
    plt.tight_layout(); plt.savefig(img_dir / "model_comparison.png", dpi=150); plt.close()


# ----------------------------------------------------------------------------- report
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path, info):
    res = info["results"].copy()
    res["CV accuracy"] = res.apply(lambda r: f"{r['CV accuracy']:.3f} +/- {r['CV std']:.3f}", axis=1)
    res_tbl = res.drop(columns="CV std")
    votes_tbl = info["votes"].drop(columns="RF_Importance_value").reset_index().rename(columns={"index": "Feature"})
    votes_tbl["Selected"] = votes_tbl["Feature"].isin(info["selected"]).map({True: "yes", False: "no"})

    notes = []
    for m in info["results"]["Model"].unique():
        sub = info["results"][info["results"]["Model"] == m]
        spread = sub["CV accuracy"].max() - sub["CV accuracy"].min()
        best = sub.loc[sub["CV accuracy"].idxmax(), "Feature set"]
        verdict = ("the differences between feature sets are smaller than the fold-to-fold variation, so the sets perform about the same"
                   if spread < sub["CV std"].max() else "the gap between feature sets is larger than the fold-to-fold variation")
        notes.append(f"- **{m}:** highest cross-validated accuracy with the *{best}* set; {verdict}.")
    sel_row = info["results"][info["results"]["Feature set"] == "Selected features"]
    all_row = info["results"][info["results"]["Feature set"] == "All engineered features"]

    nl = "\n"
    corr_lines = (nl.join(f"  - `{k}` dropped (correlation {v[1]:.2f} with `{v[0]}`, which is more related to the target)"
                          for k, v in info["corr_dropped"].items()) or "  - No pair of features exceeded the threshold.")
    var_lines = ", ".join(f"`{c}`" for c in info["var_dropped"]) or "none"
    new_rows = nl.join(f"| `{k}` | {v[0]} | {v[1]} |" for k, v in NEW_FEATURES.items())

    text = f"""# Feature Engineering and Selection Report

*This report is generated automatically by `src/feature_engineering.py`; every number comes from the run.*

## 1. Goal and dataset
Prepare the Titanic passenger data for machine learning, then find a smaller set of features that predicts survival (`Survived`: 1 = survived) as well as the full set. The data has {info['n_rows']} passengers and the original columns PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin and Embarked (Kaggle *Titanic: Machine Learning from Disaster*, `train.csv`).

## 2. Process overview
1. **Cleaning:** missing Embarked filled with the most common port, missing Age with the median age of the passenger's class and sex, missing Fare with the class median.
2. **Feature creation and encoding:** {len(info['candidates'])} candidate features were built (table below).
3. **Train/test split:** {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} stratified split ({info['n_train']} train / {info['n_test']} test), `random_state={RANDOM_STATE}`. All selection steps use the training set only, to avoid data leakage.
4. **Scaling:** `StandardScaler` (mean 0, standard deviation 1) applied to the continuous features, fitted on the training set and then applied to the test set. Binary and one-hot columns are left as 0/1.
5. **Filtering:** low-variance and highly correlated features removed.
6. **Selection:** six methods each choose their top {TOP_K} features; features chosen by at least {MIN_VOTES} methods are kept.
7. **Evaluation:** the selected set is compared with a baseline and with all engineered features.

## 3. Features created and encoded
| Feature | Built from | Description |
|---|---|---|
{new_rows}

Original columns kept as features: Pclass, Age, SibSp, Parch, Fare.

## 4. Filtering
- **Low variance** (variance below {VAR_THRESHOLD}): dropped {var_lines}. Almost-constant columns carry no information.
- **High correlation** (above {CORR_THRESHOLD}): when two features were this correlated, the one less related to the target was dropped:
{corr_lines}

{len(info['cols_after_filter'])} features remained for selection.

## 5. Selection methods
| Method | Type | Idea |
|---|---|---|
| ANOVA F-test | statistical (filter) | Does the feature's mean differ between survivors and non-survivors? |
| Mutual information | statistical (filter) | How much does knowing the feature reduce uncertainty about the target? Captures non-linear links. |
| Chi-squared | statistical (filter) | Dependence between the (0-1 scaled) feature and the target. |
| RFE with logistic regression | wrapper | Repeatedly fits the model and removes the weakest feature. |
| Random forest importance | embedded | How much each feature helps the trees split. |
| Lasso (L1) | embedded | Shrinks the coefficients of useless features towards zero. |

### Votes (1 = chosen by that method)
{md_table(votes_tbl, "{:.0f}")}

![Votes](../images/feature_votes.png)

## 6. Selected features
**{len(info['selected'])} features** were chosen by at least {MIN_VOTES} of the 6 methods:

{nl.join(f"{i + 1}. `{f}` ({int(info['votes'].loc[f, 'Votes'])} votes)" for i, f in enumerate(info['selected']))}

![Correlation of selected features](../images/selected_correlation.png)

## 7. Model performance
Baseline = {", ".join(f"`{c}`" for c in BASELINE_COLS)}. Cross-validated (CV) accuracy uses 5 stratified folds on the training set; test metrics use the held-out test set.

{md_table(res_tbl)}

![Model comparison](../images/model_comparison.png)

{nl.join(notes)}

With {len(info['selected'])} features instead of {len(info['all_features'])}, the selected set reaches a test accuracy of {sel_row['Test accuracy'].min():.3f} to {sel_row['Test accuracy'].max():.3f} across the two models, compared with {all_row['Test accuracy'].min():.3f} to {all_row['Test accuracy'].max():.3f} for all engineered features.

## 8. Conclusions and limitations
- Feature selection gives a smaller, simpler set of inputs; the table above shows how that compares with using everything.
- The test set is small ({info['n_test']} rows), so differences of a few tenths of a percent are not meaningful.
- Cleaning statistics (medians and modes) and `TicketGroupSize` were computed on the full dataset before splitting. They do not use the target, so the leakage risk is small, but a production pipeline would compute them on the training set only.
- Feature importance shows association with survival, not cause.

## 9. How to reproduce
```bash
pip install -r requirements.txt
python src/feature_engineering.py
```
Outputs: `data/titanic_features.csv`, `docs/feature_scores.csv`, `docs/feature_engineering_report.md`, and charts in `images/`.
"""
    path.write_text(text, encoding="utf-8")


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Feature engineering and selection (Titanic)")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parent.parent), help="repository root")
    root = Path(parser.parse_args().root)

    raw = load_data(root)
    df = clean_data(raw)
    X, y = engineer_features(df)
    print(f"Engineered {X.shape[1]} candidate features for {len(X)} passengers")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    cont = [c for c in CONTINUOUS if c in X.columns]
    scaler = StandardScaler().fit(X_train[cont])
    Xtr_s, Xte_s = X_train.copy(), X_test.copy()
    Xtr_s[cont] = scaler.transform(X_train[cont])
    Xte_s[cont] = scaler.transform(X_test[cont])

    cols, var_dropped = variance_filter(X_train)
    cols, corr_dropped = correlation_filter(X_train, y_train, cols)
    print(f"After filtering: {len(cols)} features (dropped by variance: {var_dropped}; by correlation: {list(corr_dropped)})")

    votes, scores, selected = select_features(Xtr_s, X_train, y_train, cols)
    print(f"Selected {len(selected)} features: {selected}")

    feature_sets = {"Baseline (original)": BASELINE_COLS,
                    "All engineered features": list(X.columns),
                    "Selected features": selected}
    results = evaluate(feature_sets, Xtr_s, Xte_s, y_train, y_test)
    print(results.round(3).to_string(index=False))

    (root / "data").mkdir(exist_ok=True); (root / "docs").mkdir(exist_ok=True)
    pd.concat([X[selected], y], axis=1).to_csv(root / "data" / "titanic_features.csv", index=False)
    votes.drop(columns="RF_Importance_value").join(scores.add_suffix("_score")).to_csv(root / "docs" / "feature_scores.csv")
    make_charts(votes, selected, X_train, results, root / "images")
    write_report(root / "docs" / "feature_engineering_report.md",
                 {"n_rows": len(X), "n_train": len(X_train), "n_test": len(X_test), "candidates": list(X.columns),
                  "all_features": list(X.columns), "var_dropped": var_dropped, "corr_dropped": corr_dropped,
                  "cols_after_filter": cols, "votes": votes, "selected": selected, "results": results})
    print("Saved report to docs/feature_engineering_report.md")


if __name__ == "__main__":
    main()
