<div align="center">

# 🏦 **Loan Default Prediction System**
### *Data Documentation & Preprocessing Summary*
📊 *A comprehensive overview of dataset structure, feature insights, and preprocessing pipeline.*

</div>

---

## 🎯 **Objective**
The primary objective of this project is to develop an intelligent **Machine Learning Model** capable of predicting the probability of **loan default** based on a borrower's demographic, financial, and behavioral attributes.  
In addition, the model is designed to recommend **personalized recovery actions** to assist financial institutions in minimizing loss and improving portfolio health.

---

## 📁 **Dataset Overview**
The **Loan Default Prediction Dataset** contains detailed information about loan applicants and their financial activity.  
It comprises **255,347 records** and **18 attributes**, covering both **numerical** and **categorical** variables.

- **Target Variable:** `Default`  
  - `1` → Loan was defaulted  
  - `0` → Loan was successfully repaid  

Each record represents a unique loan applicant and captures key aspects such as income, credit history, employment stability, and loan characteristics.

---

## 💡 **Feature Insights**

### 1. 🧾 Financial Health Indicators  
These variables directly influence a borrower’s capacity to manage debt obligations.

| Feature | Description | Insight |
|----------|--------------|----------|
| **Income** | Monthly income of the borrower | Higher income → lower default risk |
| **CreditScore** | Creditworthiness indicator | Lower scores → higher default risk |
| **DTIRatio** | Debt-to-Income ratio `(Debt / Income)` | Higher ratio → increased financial stress |
| **LoanAmount** | Total loan amount sanctioned | Higher loan amounts → greater risk exposure |

---

### 2. 👔 Stability & Responsibility Indicators  
These variables capture the borrower's lifestyle consistency and reliability.

| Feature | Description | Insight |
|----------|--------------|----------|
| **MonthsEmployed** | Duration of employment in months | Longer employment → stable income |
| **EmploymentType** | Type of employment (Full-time, Part-time, etc.) | Unemployed or part-time → higher risk |
| **Age** | Age of the borrower | Younger borrowers → more prone to default |
| **MaritalStatus** | Marital status | Married → potentially dual-income stability |
| **HasDependents** | Indicates financial dependents | More dependents → increased financial burden |
| **HasMortgage** | Existing mortgage ownership | Mortgage paid responsibly → strong credit behavior |
| **Education** | Highest educational qualification | Higher education → higher and more stable income |

---

### 3. 💳 Loan Characteristics  

| Feature | Description | Insight |
|----------|--------------|----------|
| **InterestRate** | Loan interest rate | Higher rates → higher repayment stress |
| **LoanTerm** | Loan duration (in months/years) | Minimal impact on default tendency |
| **LoanPurpose** | Purpose for which loan was taken | Business loans → relatively higher risk |
| **HasCoSigner** | Presence of a co-signer | Co-signer reduces overall default risk |
| **NumCreditLines** | Number of existing credit lines | Higher count → potential over-leverage |

---

## 🔍 **Exploratory Data Analysis (EDA)**

### Key Numerical Insights
- **Age:** Younger borrowers default more frequently.  
- **Income:** Strong negative correlation with default probability.  
- **CreditScore:** Weak predictive power in this dataset — possibly normalized or correlated with other features.  
- **LoanAmount:** Higher loan amounts → greater default tendency.  
- **MonthsEmployed:** Shorter employment duration → higher default likelihood.  
- **InterestRate:** Higher rates → higher default probability.  
- **DTIRatio:** Weak positive correlation with default.  
- **LoanTerm:** No substantial impact detected.  
- **NumCreditLines:** Slightly higher default rate with more open credit lines.

### Key Categorical Insights
Categorical variables such as **Education**, **EmploymentType**, and **MaritalStatus** show limited standalone predictive power but contribute meaningfully when combined with numeric variables.

---

## ⚙️ **Data Preprocessing Pipeline**

### 1. **One-Hot Encoding (OHE)**
All categorical variables were encoded into numeric format to ensure compatibility with ML algorithms.

- Method: `pandas.get_dummies()` / `sklearn.OneHotEncoder`  
- Parameter: `drop_first=True` (to prevent multicollinearity)  
- **Encoded Columns:**  
  `Education`, `EmploymentType`, `MaritalStatus`, `HasMortgage`, `HasDependents`, `LoanPurpose`, `HasCoSigner`  

| Stage | Shape |
|--------|--------|
| **Before Encoding** | (255,347, 18) |
| **After Encoding** | (255,347, 26) |

---

### 2. **Train–Test Split**
- Split ratio: **80% Training**, **20% Testing**  
- Stratified sampling (`stratify=y`) for balanced class distribution  
- Reproducibility ensured with `random_state=42`  
- Excluded `LoanID` as it does not contribute to prediction

---

### 3. **Feature Standardization**
Numerical features were scaled using **`StandardScaler`** from Scikit-learn to normalize feature magnitudes.

| Step | Description |
|------|--------------|
| **Fit** | Computed mean & standard deviation on training set |
| **Transform** | Applied scaling to both train & test sets |
| **Outcome** | Mean = 0, Std = 1 across numerical features |

**Scaled Numerical Features:**  
`Age`, `Income`, `LoanAmount`, `CreditScore`, `MonthsEmployed`, `NumCreditLines`, `InterestRate`, `LoanTerm`, `DTIRatio`

**Categorical (One-Hot Encoded):**  
`Education_High School`, `Education_Master’s`, `Education_PhD`,  
`EmploymentType_Part-time`, `EmploymentType_Self-employed`,  
`EmploymentType_Unemployed`, `MaritalStatus_Married`, `MaritalStatus_Single`

---

## ✅ **Final Summary**
- Total Records: **255,347**
- Final Features: **26**
- Target Variable: **Default (0 = repaid, 1 = defaulted)**
- Techniques Used: **EDA, One-Hot Encoding, Train-Test Split, Feature Scaling**
- Numerical features normalized for balanced learning.  
- Data is **ready for model development and evaluation.**

---

<div align="center">

### ✨ *Prepared with precision for the Loan Default Prediction ML Pipeline.*  
**© 2025 Loan Risk Intelligence Initiative**

</div>
