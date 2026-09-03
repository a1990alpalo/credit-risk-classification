# Credit Risk Classification

A machine-learning project that uses logistic regression to classify loans as healthy or high-risk. The project demonstrates reproducible preprocessing, stratified sampling, class-imbalance analysis, and evaluation metrics appropriate for credit-risk classification.

## Project Results

The scaled logistic regression model performed strongly across both loan classes.

| Metric | Result |
|---|---:|
| Accuracy | 0.995 |
| Balanced accuracy | 0.986 |
| High-risk precision | 0.87 |
| High-risk recall | 0.98 |
| High-risk F1-score | 0.92 |
| ROC-AUC | 0.996 |
| Average precision | 0.862 |

The model correctly identified 611 of 625 high-risk loans and missed only 14.

### Confusion Matrix

| Actual class | Predicted healthy | Predicted high-risk |
|---|---:|---:|
| Healthy | 18,669 | 90 |
| High-risk | 14 | 611 |

High-risk recall is particularly important in this analysis because a false negative represents a risky loan incorrectly classified as healthy.

## Dataset

The dataset contains 77,536 loan records with seven predictive features and one target variable.

### Features

- `loan_size`
- `interest_rate`
- `borrower_income`
- `debt_to_income`
- `num_of_accounts`
- `derogatory_marks`
- `total_debt`

### Target

- `loan_status = 0`: Healthy loan
- `loan_status = 1`: High-risk loan

The target is imbalanced:

- 75,036 healthy loans
- 2,500 high-risk loans

Because high-risk loans represent approximately 3.22% of the dataset, accuracy alone is not sufficient for evaluating the model.

## Methodology

1. Load and validate the lending dataset.
2. Separate the predictive features from the loan-status target.
3. Create stratified training and testing datasets.
4. Standardize the features with `StandardScaler`.
5. Train a logistic regression classifier.
6. Generate class predictions and high-risk probabilities.
7. Evaluate the model using:
   - Confusion matrix
   - Precision, recall, and F1-score
   - Balanced accuracy
   - ROC-AUC
   - Average precision

The scaler and classifier are combined in a scikit-learn `Pipeline`, which ensures that preprocessing is fitted only on the training data and helps prevent data leakage.

## Repository Structure

- `data/lending_data.csv`: Lending dataset
- `notebooks/credit_risk_classification.ipynb`: Complete analysis
- `requirements.txt`: Python dependencies
- `.gitignore`: Files excluded from version control

## Run the Project

Clone the repository:

```bash
git clone https://github.com/a1990alpalo/credit-risk-classification.git
cd credit-risk-classification
```

Create and activate a virtual environment using Windows Git Bash:

```bash
py -3.14 -m venv .venv
source .venv/Scripts/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab notebooks/credit_risk_classification.ipynb
```

Select the virtual environment as the notebook kernel, then restart the kernel and run all cells.

## Technologies

- Python
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- JupyterLab

## Limitations

This project is an educational classification analysis and should not be used to make real lending decisions. A production credit-risk system would also require fairness testing, probability calibration, cross-validation, monitoring for data drift, and review for applicable lending regulations.

## Author

Alberto Medina
[GitHub Profile](https://github.com/a1990alpalo)