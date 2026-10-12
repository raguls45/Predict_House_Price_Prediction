# House Price Model Training Report

- Updated: 2026-10-08 23:13:17 India Standard Time
- Dataset: `data\Real estate valuation data set.xlsx`
- Training samples: 331
- Testing samples: 83
- Best model (lowest RMSE): Random Forest
- PCA components: 5
- PCA explained variance: 0.9732
- Selected features: X2 house age, X3 distance to the nearest MRT station, X4 number of convenience stores, X5 latitude, X6 longitude

## Visual model comparison

![Bar charts comparing MAE, RMSE, and R² for each model](model_comparison.png)

## Model comparison

| Model | MAE | RMSE | R2 Score |
|---|---:|---:|---:|
| Random Forest | 4.2098 | 5.8115 | 0.7987 |
| Gradient Boosting | 4.0505 | 5.8118 | 0.7987 |
| Linear Regression | 5.3501 | 7.3879 | 0.6746 |
