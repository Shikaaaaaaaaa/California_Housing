# California Housing Price Prediction

A machine learning project that predicts median house values in California districts using demographic, geographic, and housing features.

This project was built as part of my machine learning learning path and rebuilt in my own version to practice a complete end-to-end workflow: data exploration, preprocessing, model comparison, hyperparameter tuning, and final evaluation.

## Project Goal

The goal of this project is to predict `median_house_value` using features such as:

- longitude
- latitude
- housing median age
- total rooms
- total bedrooms
- population
- households
- median income
- ocean proximity

## Dataset

The project uses the California Housing dataset, which includes information about housing districts in California.

**Target variable:**
- `median_house_value`

## Workflow

### 1. Data Exploration
I started by exploring the dataset structure and checking:
- data types
- missing values
- distributions
- correlations
- geographic patterns

### 2. Stratified Train/Test Split
Since `median_income` is an important feature, I created `income_cat` and used stratified splitting to preserve the distribution of income categories in both the training and test sets.

### 3. Preprocessing
I built separate preprocessing pipelines for:

**Numerical features:**
- median imputation
- standard scaling

**Categorical feature (`ocean_proximity`):**
- most frequent imputation
- one-hot encoding

Then I merged both pipelines using `ColumnTransformer`.

### 4. Model Training
I trained and compared three regression models:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

### 5. Model Comparison with Cross-Validation
I used 10-fold cross-validation with RMSE to compare model performance fairly.

### 6. Hyperparameter Tuning
Since Random Forest gave the best cross-validation result, I tuned it using `GridSearchCV`.

### 7. Final Evaluation
Finally, I evaluated the best tuned model on the unseen test set.

## Model Performance

### Cross-Validation Results
| Model | Mean RMSE | Std RMSE |
|---|---:|---:|
| Random Forest Regressor | 49,598.73 | 1,208.41 |
| Linear Regression | 64,906.39 | 2,148.86 |
| Decision Tree Regressor | 69,490.53 | 781.52 |

### Best Hyperparameters
```python
{
    'max_depth': None,
    'min_samples_split': 2,
    'n_estimators': 200
}
```

### Final Test Results
- **MSE:** 2,215,874,005.61
- **MAE:** 30,864.02
- **RMSE:** 47,073.07
- **R²:** 0.8299

## Key Takeaways

- `median_income` was one of the strongest predictors.
- Random Forest outperformed Linear Regression and Decision Tree.
- Proper preprocessing and fair model comparison were essential.
- Training performance alone was misleading, so cross-validation and final test evaluation were necessary.

## Tools and Libraries

- Python
- pandas
- NumPy
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

## Project Structure

```bash
California-Housing/
│
├── California_Housing.ipynb
├── README.md
└── assets/
```

## How to Run

1. Clone this repository.
2. Open the notebook in Jupyter Notebook or Google Colab.
3. Install the required libraries if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

4. Run the notebook cells in order.

## What I Practiced in This Project

- exploratory data analysis
- feature-target separation
- stratified sampling
- preprocessing pipelines
- `ColumnTransformer`
- model comparison
- cross-validation
- hyperparameter tuning with `GridSearchCV`
- final regression evaluation

## Future Improvements

- add feature importance visualization
- create a cleaner portfolio version of the notebook
- build a dashboard version of the project in Power BI
- experiment with additional models and feature engineering

## Author

**Ahmed Fathy**

If you want to connect or discuss the project, feel free to reach out through my portfolio or GitHub profile.
