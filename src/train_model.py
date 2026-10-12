from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


# ============================================================
# 1. LOAD DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
file_path = PROJECT_ROOT / "data" / "Real estate valuation data set.xlsx"

df = pd.read_excel(file_path)

print("=" * 60)
print("HOUSE PRICE PREDICTION MODEL")
print("=" * 60)


# ============================================================
# 2. REMOVE ID COLUMN
# ============================================================

df = df.drop(columns=["No"])


# ============================================================
# 3. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["Y house price of unit area"])

y = df["Y house price of unit area"]


print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Y house price of unit area")


# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 5. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# ============================================================
# 6. FEATURE SELECTION
# ============================================================

selector = SelectKBest(
    score_func=f_regression,
    k=5
)

X_train_selected = selector.fit_transform(
    X_train_scaled,
    y_train
)

X_test_selected = selector.transform(
    X_test_scaled
)


selected_features = X.columns[
    selector.get_support()
]

print("\nSelected Features:")
print(selected_features.tolist())


# ============================================================
# 7. PCA
# ============================================================

pca = PCA(
    n_components=0.95
)

X_train_pca = pca.fit_transform(
    X_train_scaled
)

X_test_pca = pca.transform(
    X_test_scaled
)

print("\nPCA Components:")
print(pca.n_components_)

print(
    "Explained Variance:",
    round(
        pca.explained_variance_ratio_.sum(),
        4
    )
)


# ============================================================
# 8. DEFINE MODELS
# ============================================================

models = {

    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        )
}


# ============================================================
# 9. TRAIN MODELS
# ============================================================

results = []

for name, model in models.items():

    # Use selected features for the models
    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_test_selected
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2 Score": r2
    })


# ============================================================
# 10. MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.sort_values(
        by="RMSE"
    ).to_string(index=False)
)


# ============================================================
# 11. SELECT BEST MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(),
    "Model"
]

print("\nBest Model:")
print(best_model_name)


# ============================================================
# 12. TRAIN BEST MODEL AGAIN
# ============================================================

best_model = models[best_model_name]

best_model.fit(
    X_train_selected,
    y_train
)


# ============================================================
# 13. SAVE MODEL
# ============================================================

models_dir = PROJECT_ROOT / "models"
models_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(
    best_model,
    models_dir / "house_price_model.pkl"
)

joblib.dump(
    scaler,
    models_dir / "scaler.pkl"
)

joblib.dump(
    selector,
    models_dir / "selector.pkl"
)


# ============================================================
# 14. SAVE FEATURE NAMES
# ============================================================

joblib.dump(
    selected_features.tolist(),
    models_dir / "selected_features.pkl"
)


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

reports_dir = PROJECT_ROOT / "reports"
reports_dir.mkdir(parents=True, exist_ok=True)

sorted_results = results_df.sort_values(by="RMSE")
comparison_report_path = reports_dir / "model_comparison.csv"
comparison_plot_path = reports_dir / "model_comparison.png"
training_report_path = reports_dir / "model_training_report.md"

sorted_results.to_csv(comparison_report_path, index=False)

model_names = sorted_results["Model"].tolist()
bar_colors = [
    "#f97316" if name == best_model_name else "#3b82f6"
    for name in model_names
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for axis, metric, title in zip(
    axes,
    ["MAE", "RMSE", "R2 Score"],
    ["Mean Absolute Error", "Root Mean Squared Error", "R² Score"],
):
    bars = axis.bar(model_names, sorted_results[metric], color=bar_colors)
    axis.set_title(title)
    axis.set_ylabel(metric)
    axis.tick_params(axis="x", labelrotation=20)
    axis.grid(axis="y", alpha=0.25)
    axis.bar_label(bars, fmt="%.2f", padding=3)

fig.suptitle(f"Model Comparison (Best RMSE: {best_model_name})")
fig.tight_layout()
fig.savefig(comparison_plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)

report_lines = [
    "# House Price Model Training Report",
    "",
    f"- Updated: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
    f"- Dataset: `{file_path.relative_to(PROJECT_ROOT)}`",
    f"- Training samples: {len(X_train)}",
    f"- Testing samples: {len(X_test)}",
    f"- Best model (lowest RMSE): {best_model_name}",
    f"- PCA components: {pca.n_components_}",
    f"- PCA explained variance: {pca.explained_variance_ratio_.sum():.4f}",
    f"- Selected features: {', '.join(selected_features.tolist())}",
    "",
    "## Visual model comparison",
    "",
    "![Bar charts comparing MAE, RMSE, and R² for each model](model_comparison.png)",
    "",
    "## Model comparison",
    "",
    "| Model | MAE | RMSE | R2 Score |",
    "|---|---:|---:|---:|",
]

for _, result in sorted_results.iterrows():
    report_lines.append(
        f"| {result['Model']} | {result['MAE']:.4f} "
        f"| {result['RMSE']:.4f} | {result['R2 Score']:.4f} |"
    )

training_report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETED")
print("=" * 60)

print("\nSaved files:")

print(models_dir / "house_price_model.pkl")
print(models_dir / "scaler.pkl")
print(models_dir / "selector.pkl")
print(models_dir / "selected_features.pkl")
print(comparison_report_path)
print(comparison_plot_path)
print(training_report_path)