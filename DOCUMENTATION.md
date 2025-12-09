# 📊 Loan Default Prediction Dataset - Documentation

## 🎯 Goal

To develop a machine learning model that predicts the risk of a borrower defaulting on a loan and recommends personalized recovery actions.

---

## 📋 About Dataset

The **Loan Default Prediction Dataset** focuses on predicting whether a borrower will default on a loan based on various demographic, financial, and loan-related attributes.

### Dataset Statistics
- **Total Records:** 255,347
- **Total Columns:** 18
- **Data Types:** Numerical and Categorical variables
- **Target Variable:** `Default` (1 = Defaulted, 0 = Not Defaulted)

Each record represents a loan applicant with details related to their personal profile, employment, financial status, and loan information.

---

## 💰 Deep-dive into Financial Parameters

### 1. Financial Health Indicators

These features directly measure a borrower's financial standing and ability to handle debt.

#### **Income**
Higher income generally means a greater capacity to repay loans, lowering the risk of default.

#### **CreditScore**
A highly significant predictor. A high credit score indicates a history of responsible borrowing and timely repayments, suggesting a lower default risk. Conversely, a low score signals higher risk.

#### **DTIRatio (Debt-to-Income Ratio)**
This crucial metric shows how much of a borrower's monthly income goes towards paying off debts.

**Formula:**
```
DTI = (Total Monthly Debt Payments) / (Gross Monthly Income)
```

- **Lower DTI** → Better (borrower isn't over-leveraged)
- **High DTI** → Major red flag (too much debt relative to income)

#### **LoanAmount**
A very large loan amount relative to the borrower's income can increase financial strain, potentially raising the default risk.

---

### 2. Stability and Responsibility Indicators

These features provide context about the borrower's life stability and sense of responsibility, which can indirectly signal their reliability.

#### **MonthsEmployed & EmploymentType**
- Long and stable employment history → Strong positive signal
- Consistent and reliable income source
- Frequent job changes or unemployment → Risk factors

#### **Age**
While not always a direct cause, age can be a proxy for financial stability and experience. Older borrowers may have more stable careers and assets.

#### **MaritalStatus & HasDependents**
- Having dependents might increase financial strain
- Married applicants might have dual-income households (lower risk)

#### **HasMortgage**
Having a mortgage and paying it on time is a strong indicator of financial responsibility and creditworthiness.

#### **Education**
Higher levels of education often correlate with higher, more stable incomes, which can lower the risk of default.

---

### 3. Loan Characteristics

#### **InterestRate**
- Higher interest rates → Higher monthly payments → Increased default chance
- Rates are often higher for riskier borrowers

#### **LoanTerm**
- Longer loan term → Lower monthly payments but more interest over time
- Very long terms might indicate borrower can't afford higher payments

#### **LoanPurpose**
The reason for the loan can be indicative of risk:
- Business venture loan → Higher risk
- Home improvement loan → Lower risk

#### **HasCoSigner**
The presence of a co-signer significantly reduces lender risk, as there is a second person legally responsible for repayment.

#### **NumCreditLines**
Shows how much credit the borrower already has access to. A very high number of open credit lines could indicate over-extension and heavy reliance on debt.

---

## 🔍 Exploratory Data Analysis

### Goal
To analyze the features and their correlation with the target variable (default).

---

## 📈 Conclusions on Numeric Features

### **Age** ✅
- **Finding:** Defaulters tend to be younger
- **Insight:** Financial maturity and stability (often correlated with age) influence repayment reliability
- **Predictive Value:** Useful discriminative feature

### **Income** ✅
- **Finding:** Borrowers with lower income have higher default likelihood
- **Insight:** Limited financial capacity increases repayment risk
- **Predictive Value:** Important feature for default prediction

### **Credit Score** ⚠️
- **Finding:** Does NOT show strong relationship with default
- **Insight:** Default proportion remains nearly constant across credit score ranges
- **Predictive Value:** May not be highly predictive in this dataset

**Possible Reasons:**
- Dataset might have synthetic or normalized credit scores
- Credit score may already be factored into correlated variables (interest rate, income)
- Population might be mostly mid-score borrowers (limited variation)

### **Loan Amount** ✅
- **Finding:** Higher loan amounts correlate with higher default rates
- **Insight:** Larger loans increase financial burden
- **Predictive Value:** Useful predictor

### **Months Employed** ✅
- **Finding:** Defaulters are more common among people with fewer months employed
- **Insight:** Shorter employment duration → Higher default risk
- **Predictive Value:** Strong indicator of repayment reliability

### **Interest Rate** ✅
- **Finding:** Higher interest rates correlate with higher default rates
- **Insight:** Increased monthly payment burden leads to defaults
- **Predictive Value:** Useful predictor for default risk

### **DTI Ratio** ⚠️
- **Finding:** Weak positive relationship with default
- **Insight:** Higher DTI → Slightly higher default, but not strong
- **Predictive Value:** Not a strong standalone predictor

### **Loan Term** ❌
- **Finding:** No significant influence on default likelihood
- **Insight:** Borrowers with shorter or longer terms show similar default behavior
- **Predictive Value:** Not a strong predictor in this dataset

### **Number of Credit Lines** ✅
- **Finding:** More credit lines → Slightly higher default risk
- **Insight:** Multiple open accounts may increase financial burden
- **Predictive Value:** Minor but useful predictor

---

## 📊 Conclusions on Categorical Features

### Overall Finding
These categorical features don't show strong separation between defaulters and non-defaulters — their influence seems limited.

### Predictive Value
- **Standalone:** Limited predictive power
- **Combined:** Might add small value when combined with numeric features (credit score, income, DTI, interest rate)

**Key Insight:** Categorical features alone don't strongly predict default but may contribute to ensemble models.

---

## 🔧 One Hot Encoding

### Goal
To convert categorical (non-numeric) data into a numerical format that machine learning models can process.

### Dataset Transformation

#### **Before Encoding:**
- Shape: `(255,347 × 18)`
- Contains both numerical and categorical columns

#### **After Encoding:**
- Shape: `(255,347 × 26)`
- All categorical columns converted to numeric dummy variables

### Categorical Columns Encoded

The following **7 categorical columns** were one-hot encoded:

1. `Education`
2. `EmploymentType`
3. `MaritalStatus`
4. `HasMortgage`
5. `HasDependents`
6. `LoanPurpose`
7. `HasCoSigner`

### Encoding Method

**Performed One-Hot Encoding** using `pandas.get_dummies()` or `sklearn.OneHotEncoder`:

- **Used `drop_first=True`** → Avoids multicollinearity (dummy variable trap)
- Each categorical feature expanded into multiple binary (0/1) columns
- One binary column per category was dropped to prevent redundancy

### Conclusion
Successfully transformed categorical variables into numerical format suitable for machine learning algorithms while preventing multicollinearity issues.

---

## 🔀 Data Splitting and Standard Scaling

### Goal
To perform train-test split on the dataset and standardize feature values using StandardScaler from scikit-learn.

---

## 📊 Train–Test Split

### Objective
To divide the dataset into training and testing subsets to evaluate model performance on unseen data.

### Implementation Details

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)
```

### Parameters Used
- **Test Size:** 20% (80% training, 20% testing)
- **random_state=42:** Ensures reproducibility
- **stratify=y:** Maintains equal distribution of target classes in both subsets

### Feature Selection
> ⚠️ **NOTE:** The column `LoanID` was excluded as it does not contribute to prediction.

### X_train Columns

#### **Numerical Features:**
- Age
- Income
- LoanAmount
- CreditScore
- MonthsEmployed
- NumCreditLines
- InterestRate
- LoanTerm
- DTIRatio

#### **Categorical Features (One-Hot Encoded):**
- Education_High School
- Education_Master's
- Education_PhD
- EmploymentType_Part-time
- EmploymentType_Self-employed
- EmploymentType_Unemployed
- MaritalStatus_Married
- MaritalStatus_Single

---

## 📏 StandardScaler (Feature Standardization)

### Objective
To scale numerical features so that they have **mean = 0** and **standard deviation = 1**, ensuring equal importance for all features during model training.

### Implementation

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### Process
1. **Fit on Training Data:** Scaler learns mean and standard deviation from training data only
2. **Transform Training Data:** Apply scaling to training set
3. **Transform Test Data:** Apply same scaling parameters to test set (prevents data leakage)
4. **Preserve Structure:** Features converted back to DataFrames with original column names and indices

### What Gets Scaled?
- ✅ **Numerical columns:** Standardized to mean=0, std=1
- ❌ **Categorical dummy variables:** Remain as binary (0/1) values

---

## ✅ Final Conclusion

### Data Preparation Summary

1. **Dataset Split:**
   - 80% training data
   - 20% testing data
   - Balanced class distribution maintained
   - Reproducible results ensured

2. **Feature Scaling:**
   - Numerical columns normalized using StandardScaler
   - Categorical dummy variables (0/1) remained unaffected
   - No data leakage between train and test sets

3. **Ready for Modeling:**
   - Features properly encoded and scaled
   - Target variable balanced across splits
   - Dataset prepared for machine learning algorithms

---

## 📚 Key Takeaways

| Aspect | Status | Notes |
|--------|--------|-------|
| **Strong Predictors** | ✅ | Income, Age, MonthsEmployed, InterestRate, LoanAmount |
| **Weak Predictors** | ⚠️ | CreditScore, DTIRatio, LoanTerm |
| **Categorical Features** | ⚠️ | Limited standalone value, useful in combinations |
| **Data Quality** | ✅ | Properly encoded, scaled, and split |
| **Model Readiness** | ✅ | Dataset ready for ML algorithms |

---

<div align="center">

### 🎯 Dataset is now ready for Machine Learning Model Training!

</div>
