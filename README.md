# ⭐ CreditPathAI — AI-Driven Loan Default Prediction & Credit Risk Intelligence

> End-to-end ML system for predicting loan defaults and empowering data-driven lending decisions.

**👨‍💻 Developed by:** Pavan Doddavarapu  
**📚 Program:** Springboard Infosys Virtual Internship Program • 2025  
**🎓 Mentor:** Dr. N. Jagan Mohan

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Academic Details](#-academic-details)
- [Objectives](#-objectives)
- [Dataset Overview](#-dataset-overview)
- [Dataset Sources](#-dataset-sources)
- [Project Structure](#️-project-structure)
- [Technologies Used](#️-technologies-used)
- [Installation & Setup](#️-installation--setup)
- [Training the Models](#-training-the-models)
- [Running the Streamlit Application](#-running-the-streamlit-application)
- [Business Impact](#-business-impact)
- [Model Performance Summary](#-model-performance-summary)
- [Key Insights](#-key-insights)
- [End-to-End Workflow](#-end-to-end-workflow)
- [Future Enhancements](#-future-enhancements)
- [Troubleshooting](#️-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)
- [Contact](#-contact)

---

## 📘 Project Overview

**CreditPathAI** is a complete Machine Learning ecosystem designed to predict loan defaults, understand borrower behavior, and support financial institutions in making accurate, risk-aware lending decisions.

### It integrates:

- 🌐 A fully interactive **Streamlit web application**
- 🤖 Multiple **machine learning models**
- 📊 Extensive **EDA** on Kaggle & Microsoft datasets
- 🔍 **Preprocessing pipelines**
- 📈 **Model comparison & insights**
- 📂 **Production-ready project structure**

The system analyzes borrower demographics, financial metrics, collateral attributes, and loan characteristics to generate **real-time default risk predictions**.

---

## 🎓 Academic Details

| Field | Details |
|-------|---------|
| **Developer** | Pavan Doddavarapu |
| **Program** | Springboard Infosys Virtual Internship |
| **Mentor** | Dr. N. Jagan Mohan |
| **Year** | 2025 |

---

## 🎯 Objectives

- ✅ Build an ML-based engine to predict loan default probability
- ✅ Create an interactive Streamlit interface for real-time predictions
- ✅ Compare 7 machine learning models using consistent pipelines
- ✅ Analyze loan applicant behavior using EDA, feature engineering, and model interpretation
- ✅ Improve lending strategies through data-driven insights
- ✅ Reduce credit loss by identifying high-risk profiles early

---

## 📊 Dataset Overview

The system utilizes **24+ features** from borrower demographics, loan attributes, financial indicators, and property characteristics:

### Demographics
- Gender
- Age Group
- Region (North, South, Central, North-East)

### Financial Indicators
- Credit Score
- Annual Income
- Debt-to-Income Ratio (DTI)
- Credit Type (CRIF, CIBIL, EXP, etc.)

### Loan Details
- Loan Amount
- Loan Term
- Loan Purpose
- Interest Rate
- Loan Limit (Conforming/Non-Conforming)

### Property & Collateral
- Property Value
- Occupancy Type
- Total Units

### Target Variable
- **Loan Default:** `0` = No Default, `1` = Default

---

## 📂 Dataset Sources

### 1️⃣ Kaggle Loan Default Dataset
- Rich loan application dataset
- Used as the primary dataset for model training

### 2️⃣ Microsoft Loan Credit Risk Dataset
- Enterprise-grade borrower + loan data
- Used for cross-validation and enhancing generalization

---

## 🏗️ Project Structure

```
CreditPathAI/
│
├── streamlit_app/
│   ├── app.py                     # Main Streamlit UI
│   ├── utils.py                   # Model loading & prediction
│   ├── requirements.txt           # App-specific dependencies
│   ├── models/                    # Trained model pipelines (*.joblib)
│   └── __pycache__/
│
├── notebooks/
│   ├── eda_report.ipynb           # Exploratory Data Analysis
│   ├── main.ipynb                 # ML pipeline development
│   ├── preprocessing1.ipynb
│   └── preprocessing2.ipynb
│
├── microsoft_notebooks/
│   ├── eda_report.ipynb
│   └── microsoft_loan_default.ipynb
│
├── Loan_Default.csv               # Training dataset
├── Loan.txt                       # Microsoft loan dataset
├── Loan_Prod.txt
├── Borrower.txt
├── Borrower_Prod.txt
│
├── Model_comparison.xlsx          # Model metric results
├── requirements.txt               # Global dependencies
├── .gitignore
├── LICENSE
└── README.md
```

> ⚠️ **Note:** Model files (`*.joblib`) are intentionally excluded from Git using `.gitignore`.

---

## 🛠️ Technologies Used

### Languages
- Python 3.13

### Machine Learning
- Scikit-learn
- XGBoost
- Joblib (model serialization)

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### App Framework
- Streamlit

### Development Tools
- Jupyter Notebook
- Git & GitHub

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/springboardmentor891v/CreditPathAI.git
cd CreditPathAI
```

### 2️⃣ Create a Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # macOS/Linux
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
pip install -r streamlit_app/requirements.txt
```

---

## 🧠 Training the Models

Run the preprocessing + training notebook:

```bash
jupyter notebook notebooks/preprocessing2.ipynb
```

Once executed, model pipelines will appear in:
```
streamlit_app/models/
```

---

## 🚀 Running the Streamlit Application

```bash
streamlit run streamlit_app/app.py
```

App will open at:  
👉 **http://localhost:8501**

---

## 📌 Business Impact

CreditPathAI offers strategic benefits for lending operations:

- ✅ **Reduced Credit Loss** through proactive default prediction
- ✅ **Efficient Underwriting** via automated risk scoring
- ✅ **Improved Profitability** by identifying safe borrowers
- ✅ **Optimized Recovery Strategies** for high-risk applicants
- ✅ **Real-time Decision-Making** integrated through a clean web UI
- ✅ **Model-flexibility** allowing selection of preferred classifier

---

## 📈 Model Performance Summary

The project evaluates:

| Model | Strength |
|-------|----------|
| **Logistic Regression** | Interpretable baseline |
| **Random Forest** | Strong accuracy, handles nonlinearity |
| **XGBoost** | Top performance, imbalance handling |
| **Decision Tree** | Explainable structure |
| **KNN** | Instance-based predictions |
| **Gaussian NB** | Fast, probabilistic |
| **Bernoulli NB** | Great for binary feature patterns |

Each model is compared using:
- Precision
- Recall
- F1-Score

Results are stored in `Model_comparison.xlsx`.

---

## 💡 Key Insights

- 🔍 **Credit Score & DTI ratio** are primary predictors of default
- 📊 Higher loan amount with lower income → **increased risk**
- ✅ **Pre-approval** strongly reduces default probability
- 🏠 **Property occupancy type** impacts repayment behavior
- 🌍 **Regional variations** influence loan outcome distribution
- 🔄 Microsoft dataset cross-validation improves reliability

---

## 🔄 End-to-End Workflow

1. **Data Collection** → Kaggle + Microsoft datasets
2. **EDA** → Feature patterns, correlations, distribution checks
3. **Preprocessing** → Encoding, Scaling, Handling missing values, Train-test splitting
4. **Model Training** → 7 ML models with consistent pipelines
5. **Evaluation** → Metric comparison + confusion matrices
6. **Deployment** → Streamlit interface + model loader

---

## 🚧 Future Enhancements

- 🔍 SHAP / LIME model explainability
- 🧬 Advanced feature engineering
- 🧠 Neural network experimentation
- 🌐 REST API for production integration
- 🧪 A/B testing for model choice
- 📊 Continuous model monitoring
- 🔄 Auto-retraining pipeline

---

## 🛠️ Troubleshooting

### 1. Model file missing?
Run the training notebook—models are intentionally not tracked in Git.

### 2. Version conflicts?
Use exact versions in `requirements.txt`.

### 3. Streamlit not launching?
Ensure venv is active and dependencies installed.

### 4. Import errors in notebooks?
Reinstall packages:
```bash
pip install -r requirements.txt
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License**.

---

## 📞 Contact

**👨‍💻 Developer:** Pavan Doddavarapu  
**📧 GitHub:** Open an issue on [GitHub](https://github.com/springboardmentor891v/CreditPathAI) for queries or suggestions

---

<div align="center">
  
### ⭐ If you find this project helpful, please give it a star!

</div>
