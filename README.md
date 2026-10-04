# A04: Model Interpretability (xAI)

**OPIM 5512 – Data Science using Python, University of Connecticut (Fall 2026)**  
**Author:** Parvathi Meghanath

## Overview
This assignment builds a classification model to predict whether a loan application is approved (`Loan_Status`), then uses explainable AI (xAI) techniques to understand what the model relies on and how each feature affects its predictions.

## Dataset
- **File:** `train_loan_imbalanced.csv` (614 loan applications, 11 predictors)
- **Target:** `Loan_Status`, recoded as 1 = approved, 0 = rejected
- **Class balance:** about 69% approved and 31% rejected (imbalanced)
- **Predictors:** Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area

## Approach
1. **Cleaning:** dropped `Loan_ID` (an identifier) and recoded the target to 1/0.
2. **Train/test split:** 80/20, stratified to keep the approval rate the same in both sets.
3. **Preprocessing:** median imputation for numeric features; most-frequent imputation and one-hot encoding for categorical features, all inside a scikit-learn pipeline to avoid data leakage.
4. **Model tuning:** Random Forest tuned with GridSearchCV over 24 settings using 5-fold cross-validation, scored on ROC AUC because the classes are imbalanced.
5. **Permutation importance:** each feature shuffled 20 times on the test set to measure the drop in AUC; the top 5 are shown in a boxplot.
6. **Partial dependence plots:** drawn for the top 5 features on the same 0–1 Y axis so their effects can be compared directly.

## Results
| Metric | Value |
|---|---|
| Best model | Random Forest (balanced class weight, max depth 4, min samples per leaf 5, 150 trees) |
| Cross-validated AUC | 0.753 |
| Test AUC | 0.765 |
| Test accuracy | 0.72 |

**Top 5 features (mean drop in test AUC):**

| Rank | Feature | Importance |
|---|---|---|
| 1 | Credit_History | 0.268 |
| 2 | Property_Area | 0.025 |
| 3 | Married | 0.015 |
| 4 | LoanAmount | 0.009 |
| 5 | Loan_Amount_Term | 0.004 |

## Key Findings
- **Credit history drives the model:** predicted approval rises from about 0.27 to 0.59 when an applicant has a qualifying credit history.
- **Property area matters somewhat:** Semiurban applicants have the highest predicted approval (0.60), ahead of Urban (0.51) and Rural (0.49).
- **Married applicants** are slightly more likely to be approved (0.56 vs. 0.50).
- **Larger loans and longer terms** lower approval only slightly (from about 0.56 to 0.51).
- **Income barely matters:** ApplicantIncome and CoapplicantIncome have near-zero importance, which was surprising.

## Repository Structure
| File | Description |
|---|---|
| `A04_xAI.ipynb` | Main notebook with code, outputs, plots and interpretation |
| `train_loan_imbalanced.csv` | Loan dataset (downloaded automatically by the notebook) |
| `.gitignore` | Excludes the virtual environment and cache files |
| `README.md` | Project overview |

## How to Run
1. Clone the repository and open the folder in VS Code.
2. Create a virtual environment and install the packages: `pip install pandas numpy matplotlib scikit-learn gdown ipykernel`
3. Open `A04_xAI.ipynb`, select the `.venv` kernel, and click **Run All**.

## Tools
Python · pandas · scikit-learn · matplotlib · VS Code · Jupyter
