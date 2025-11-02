## **Loan Default Prediction Dataset**

**Project Goal:**  
To develop a machine learning model that predicts the risk of a borrower defaulting on a loan and recommends personalized recovery actions.

**About Dataset:**  
The **Loan Default Prediction Dataset** focuses on predicting whether a borrower will default on a loan based on various demographic, financial, and loan-related attributes.

The dataset consists of **255,347 records** and **18 columns**, including both numerical and categorical variables. Each record represents a loan applicant with details related to their personal profile, employment, financial status, and loan information. The target variable, **“Default”**, indicates whether the loan was defaulted (1) or not (0).

**Deep-dive into financial parameters:**

**1\. Financial Health Indicators:** These features directly measure a borrower's financial standing and ability to handle debt.

* **Income:** Higher income generally means a greater capacity to repay loans, lowering the risk of default.  
* **CreditScore:** This is a highly significant predictor. A high credit score indicates a history of responsible borrowing and timely repayments, suggesting a lower default risk. Conversely, a low score signals higher risk.   
* **DTIRatio (Debt-to-Income Ratio)**: This crucial metric shows how much of a borrower's monthly income goes towards paying off debts. It's calculated as:  
  DTI \= (Gross Monthly Income) / (Total Monthly Debt Payments​)  
  A lower DTI is better, as it suggests the borrower isn't over-leveraged and has enough income left over to handle new loan payments. A high DTI is a major red flag.  
* **LoanAmount:** A very large loan amount relative to the borrower's income can increase the financial strain, potentially raising the default risk.

**2\. Stability and Responsibility Indicators:** These features provide context about the borrower's life stability and sense of responsibility, which can indirectly signal their reliability.

* **MonthsEmployed & EmploymentType:** A long and stable employment history (e.g., many months in a full-time job) is a strong positive signal. It suggests a consistent and reliable source of income. Frequent job changes or unemployment are risk factors.  
* **Age:** While not always a direct cause, age can be a proxy for financial stability and experience. Older borrowers may have more stable careers and assets.  
* **MaritalStatus & HasDependents:** These factors can influence financial stability. For example, having dependents might increase financial strain, but a married applicant might have a dual-income household, which could lower risk.  
* **HasMortgage:** Having a mortgage and paying it on time is a strong indicator of financial responsibility and creditworthiness.  
* **Education**: Higher levels of education often correlate with higher, more stable incomes, which can lower the risk of default.

**3\. Loan Characteristics:**

* **InterestRate:** Higher interest rates lead to higher monthly payments, which can strain a borrower's budget and increase the chance of default. Rates are often higher for riskier borrowers in the first place.  
* **LoanTerm:** A longer loan term means lower monthly payments but more interest paid over time. A very long term might be taken by someone who can't afford higher payments, indicating potential risk.  
* **LoanPurpose:** The reason for the loan can be indicative of risk. For example, a loan for a business venture might be riskier than a loan for a home improvement project.  
* **HasCoSigner:** The presence of a co-signer (someone who agrees to pay the debt if the primary borrower cannot) significantly reduces the risk for the lender, as there is a second person legally responsible for the repayment.  
* **NumCreditLines:** This shows how much credit the borrower already has access to. A very high number of open credit lines could indicate that the borrower is over-extended and relies heavily on debt.


## **Exploratory Data Analysis**

**Goal:**  
To analyze the features and their co-relation with the target variable(default).

**Conclusions on Numeric Features:**

1. **Age:** It is a useful feature in this loan dataset. It provides discriminative information — defaulters tend to be younger, indicating that financial maturity and stability (often correlated with age) influence repayment reliability.  
2. **Income:** Borrowers with lower income levels tend to have a higher likelihood of defaulting on loans. This suggests that limited financial capacity increases repayment risk, making Income an important predictive feature for assessing default probability.  
3. **Credit Score:** The credit score does not show a strong relationship with loan default in this dataset. The default proportion remains nearly constant across credit score ranges, suggesting that CreditScore may not be a highly predictive feature for default prediction here.  
   Possible Reasons:  
* The dataset might have **synthetic or normalized credit scores** (not real-world distribution).  
* Credit score may already be factored into other correlated variables (like interest rate or income).  
* If the population is mostly mid-score borrowers, variation may not be enough to show differences.  
4. **Loan Amount:** Borrowers with higher loan amounts tend to default more often compared to those with smaller loans.  
5. **Months Employed:** Defaulters are more common among people with fewer months employed, and their count drops steadily as employment duration increases. Individuals with shorter employment durations are more likely to default on their loans, while those with stable and longer job histories tend to repay reliably.  
6. **Interest Rates:** This suggests that as the interest rate increases, the likelihood of default tends to rise. Therefore, interest rate is a useful predictor for default risk — higher interest loans are more likely to be defaulted on.

7. **DTI Ratio:** The DTI ratio shows a weak positive relationship with default probability — people with higher DTI ratios *might* default slightly more often, but it’s not a strong or standalone predictor. Other variables (like income, credit score, or loan amount) likely play a more significant role.  
8. **Loan Term:** Loan term does not appear to influence the likelihood of default. Borrowers with shorter or longer loan durations have nearly the same default behavior, indicating that loan term is not a strong predictor of default risk in this dataset.  
9.  **Number of Credit Lines:** Borrowers with more credit lines are slightly more prone    to default. This could indicate that having multiple open credit accounts increases financial burden or repayment risk, leading to higher default probability.

**Conclusions on Categorical Features:**  
These categorical features don’t show strong separation between defaulters and non-defaulters — their influence seems limited. They might add small predictive value when combined with numeric features (like credit score, income, DTI, interest rate, etc.), but on their own, they don’t strongly predict default. 


## **One Hot Encoding**

**Goal:**  
To convert categorical (non-numeric) data into a numerical format that can be understood and processed by machine learning models using OHE.

### **Dataset Overview:**

* **Before Encoding:**  
   Shape → (255347, 18\)  
   (Contains both numerical and categorical columns)

* **After Encoding:**  
   Shape → (255347, 26\)  
   (All categorical columns converted to numeric dummy variables)

### **Categorical Columns Encoded:**

The following 7 categorical columns were one-hot encoded:

1. Education  
2. EmploymentType  
3. MaritalStatus  
4. HasMortgage  
5. HasDependents  
6. LoanPurpose  
7. HasCoSigner

**Conclusion:**

Performed One-Hot Encoding (via pandas.get\_dummies() or sklearn.OneHotEncoder) :

1. Used drop\_first=True → to avoid multicollinearity (dummy variable trap).  
2. Each categorical feature was expanded into multiple binary (0/1) columns representing each category.  
3. Which resulted in dropping of one binary column per unique category. 


## **Data Splitting and Standard Scalar**

**Goal:**  
To perform train\_test\_split on the dataset and standardize the feature values using Standard Scalar from Scikit learn library.

### **Train–Test Split:**

To divide the dataset into **training** and **testing** subsets to evaluate model performance on unseen data.

1. The dataset was divided into training and testing subsets using **train\_test\_split** from scikit-learn.  
2. The training data helps the model learn, while the testing data is used to evaluate its performance.  
3. random\_state=42 was used for reproducibility.  
4. stratify=y ensures equal distribution of target classes in both subsets.

NOTE: The column LoanID was excluded as it does not contribute to prediction.

**X\_train Columns :** 

* **Numerical Features:** Age, Income, LoanAmount, CreditScore, MonthsEmployed,    NumCreditLines, InterestRate, LoanTerm, DTIRatio.  
* **Categorical Features:** Education\_High School, Education\_Master’s, Education\_PhD, EmploymentType\_Part-time, EmploymentType\_Self-employed, EmploymentType\_Unemployed, MaritalStatus\_Married, MaritalStatus\_Single

**StandardScaler (Feature Standardization):**

To scale numerical features so that they have mean \= 0 and standard deviation \= 1, ensuring equal importance for all features during model training.

1. **StandardScaler** from scikit-learn was used to standardize feature values.  
2. The scaler was **fitted only on training data** to learn mean and standard deviation.  
3. The same scaling parameters were applied to the test data using **transform()** to prevent **data leakage**.  
4. After scaling, all features were converted back into DataFrames with original column names and indices.

**Conclusion:**

1. The dataset was split into **80% training** and **20% testing** data while ensuring balanced class distribution and reproducibility.  
2. Numerical columns were normalized and categorical dummy variables (0/1) remained unaffected.



