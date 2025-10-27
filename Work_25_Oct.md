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
