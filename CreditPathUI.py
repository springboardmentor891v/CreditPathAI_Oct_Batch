# app.py
import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import io

# -------------------------
# Model files
# -------------------------
models = {
    "Logistic Regression": "notebooks/notebooks/models/Logistic_Regression.pkl",
    "Random Forest": "notebooks/notebooks/models/Random_Forest.pkl",
    "XGBoost": "notebooks/notebooks/models/XGBoost.pkl",
    "Gradient Boosting": "notebooks/notebooks/models/Gradient_Boosting.pkl",
    "K-Nearest Neighbors (KNN)": "notebooks/notebooks/models/K-Nearest_Neighbors_(KNN).pkl",
    "AdaBoost": "notebooks/notebooks/models/AdaBoost.pkl",
    "Gaussian Naive Bayes": "notebooks/notebooks/models/Gaussian_Naive_Bayes.pkl",
    "LightGBM": "notebooks/notebooks/models/LightGBM.pkl"
}

# -------------------------
# Page Config & CSS
# -------------------------
st.set_page_config(page_title="Loan Default Prediction", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
.stButton>button { background-color: #4B0082; color: white; font-size:16px; font-weight:bold; border-radius:12px; padding:10px 20px; transition:0.3s; }
.stButton>button:hover { background-color: #6A0DAD; color: #f2f2f2; transform: scale(1.05); }
.info-box { background-color: #f9f5ff; border-radius: 15px; padding: 25px; margin: 40px auto; width: 80%; box-shadow: 0px 4px 12px rgba(0,0,0,0.2); font-size: 16px; line-height: 1.6; color: #333; }
.sidebar-title { font-size: 19px !important; font-weight: bold !important; color: #6A0DAD !important; margin-bottom: 8px !important; border-bottom: 2px solid #ddd; padding-bottom: 4px; }
.section-heading { text-align: center; font-size: 22px; color: #4B0082; margin-bottom: 12px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# App Heading
# -------------------------
st.markdown("""
<h1 style='text-align: center; color: #4B0082; margin-bottom:0;'>Loan Default Prediction</h1>
<p style='text-align: center; color: #555; margin-top:0;'>AI-powered app to assess loan risks using multiple ML models</p>
""", unsafe_allow_html=True)

# -------------------------
# Tabs
# -------------------------
tab1, tab2, tab3, tab4 = st.tabs(["📘 Overview", "📂 DATASET", "📊 MODEL METRICS", "🎯 PREDICTION"])


# -------------------------
# Tab 1: Overview
# -------------------------

import streamlit as st

with tab1:
    # Main Heading
    # Project Overview Heading
    st.markdown("<h3 style='color:#4B0082; text-align:center; font-weight:bold;'>Project Overview – CreditPath AI</h3>", unsafe_allow_html=True)

    # Overview Card
    st.markdown(
        """
        <style>
        .overview-card {
            background-color: #f5f5ff;
            border-radius: 15px;
            padding: 30px 25px;
            font-size: 18px;
            line-height: 1.8;
            box-shadow: 0px 5px 18px rgba(75, 0, 130, 0.2);
            max-width: 1000px;
            margin: 20px auto;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .overview-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 10px 25px rgba(75, 0, 130, 0.3);
        }
        </style>

        <div class="overview-card">
            <b style='color:#4B0082;'>CreditPath AI</b> is a <b>machine learning-based credit risk management system</b> developed under the Infosys Springboard Internship 6.0 (AI Batch 1). It focuses on predicting loan default risks and offering personalized recovery strategies to help financial institutions make smarter, data-driven lending decisions.
            <br><br>
            Loan defaults are a major concern in the financial sector, leading to increased <b>Non-Performing Assets (NPAs)</b> and financial losses. Traditional recovery methods are <b>manual, slow, and non-personalized</b>, which makes early identification of risky borrowers difficult.
            <br><br>
            <b style='color:#4B0082;'>CreditPath AI</b> uses <b>machine learning algorithms</b> to analyze borrower profiles, financial history, and behavioral patterns to detect potential defaulters early. The system supports <b>targeted recovery strategies</b>, <b>optimized risk management</b>, and <b>data-driven decision-making</b>.
        </div>
        """,
        unsafe_allow_html=True
    )


    # --- Methodology Section ---
    st.markdown("<h3 style='color:#4B0082; text-align:center; font-weight:bold; margin-top:30px;'>⚙️ Methodology</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        /* Timeline Container */
        .timeline-container {
            position: relative;
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            flex-wrap: nowrap;
            margin: 50px auto;
            max-width: 1200px;
            padding: 0 20px;
        }

        /* Horizontal Line */
        .timeline-container::before {
            content: "";
            position: absolute;
            top: 0; /* line at top */
            left: 5%;
            right: 5%;
            height: 4px;
            background-color: #4B0082;
            border-radius: 2px;
            z-index: 0;
        }

        /* Timeline Step */
        .timeline-step {
            position: relative;
            width: 18%;
            text-align: center;
            z-index: 1;
        }

        /* Dot Style (below line) */
        .timeline-dot {
            width: 20px;
            height: 20px;
            background-color: #fff;
            border: 4px solid #4B0082;
            border-radius: 50%;
            margin: 0 auto;
            position: relative;
            top: 3px; /* below the line */
            z-index: 2;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .timeline-dot:hover {
            transform: scale(1.2);
            box-shadow: 0px 0px 12px rgba(75, 0, 130, 0.5);
        }

        /* Timeline Titles */
        .timeline-title {
            color: #4B0082;
            font-weight: 800;
            font-size: 18px;
            margin: 15px 0 10px 0;
        }

        /* Timeline Cards */
        .timeline-card {
            background-color: #f8f6ff;
            box-shadow: 0px 4px 15px rgba(75, 0, 130, 0.15);
            border-radius: 15px;
            padding: 20px;
            font-size: 16px;
            line-height: 1.6;
            margin-top: 10px; /* below titles */
            min-height: 150px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .timeline-card:hover {
            transform: translateY(-8px);
            box-shadow: 0px 8px 25px rgba(75, 0, 130, 0.25);
        }

        /* Responsive Design */
        @media (max-width: 1000px) {
            .timeline-container {
                flex-direction: column;
                align-items: center;
                margin: 30px auto;
            }
            .timeline-container::before {
                display: none;
            }
            .timeline-step {
                width: 90%;
                margin-bottom: 30px;
            }
            .timeline-dot {
                top: 0px;
            }
        }
        </style>

        <div class="timeline-container">
            <div class="timeline-step">
                <div class="timeline-dot"></div>
                <div class="timeline-title">Data Collection</div>
                <div class="timeline-card">
                    Loan application records, repayment histories, demographic data, and behavioral variables were gathered from financial institution databases and publicly available datasets.
                </div>
            </div>
            <div class="timeline-step">
                <div class="timeline-dot"></div>
                <div class="timeline-title">Data Cleaning</div>
                <div class="timeline-card">
                    Missing and inconsistent values were handled using imputation or removal. Outliers were treated to prevent bias. Data formats were standardized for consistency.
                </div>
            </div>
            <div class="timeline-step">
                <div class="timeline-dot"></div>
                <div class="timeline-title">Feature Engineering</div>
                <div class="timeline-card">
                    Derived new features such as repayment ratio, income-to-loan ratio, and late payment frequency. Categorical variables were encoded and numeric features normalized.
                </div>
            </div>
            <div class="timeline-step">
                <div class="timeline-dot"></div>
                <div class="timeline-title">Model Training</div>
                <div class="timeline-card">
                    Machine learning models like Logistic Regression, Random Forest, and XGBoost were used to predict loan default. Class imbalance handled using SMOTE or undersampling.
                </div>
            </div>
            <div class="timeline-step">
                <div class="timeline-dot"></div>
                <div class="timeline-title">Evaluation</div>
                <div class="timeline-card">
                    Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Confusion matrix assessed ability to identify defaulters correctly.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Key Highlights Section - Styled Cards with Icons
    st.markdown("<h3 style='color:#4B0082; text-align:center; font-weight:bold;'>🚀 Key Highlights</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .highlight-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            gap: 20px;
            margin: 20px 0;
        }
        .highlight-card {
            background-color: #f5f5ff;
            border-left: 6px solid #4B0082;
            border-radius: 12px;
            padding: 20px;
            flex: 1 1 45%;
            font-size: 17px;
            line-height: 1.6;
            box-shadow: 0px 4px 12px rgba(75, 0, 130, 0.15);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .highlight-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 8px 20px rgba(75, 0, 130, 0.25);
        }
        .highlight-icon {
            font-size: 24px;
            margin-right: 10px;
            vertical-align: middle;
            color: #4B0082;
        }
        </style>

        <div class="highlight-container">
            <div class="highlight-card"><span class="highlight-icon">💡</span> Combines <b>loan default prediction</b> with <b>personalized recovery strategy</b>.</div>
            <div class="highlight-card"><span class="highlight-icon">📊</span> Integrates <b>behavioral and demographic profiling</b> for better segmentation.</div>
            <div class="highlight-card"><span class="highlight-icon">⚡</span> Uses <b>open-source, scalable AI models</b> for cost-effective deployment.</div>
            <div class="highlight-card"><span class="highlight-icon">🎯</span> Early identification of risky borrowers to reduce financial losses.</div>
        </div>
        """,
        unsafe_allow_html=True
    )
   
    # --- Goal Section ---
    st.markdown("<h3 style='color:#4B0082; text-align:center; font-weight:bold;'>🎯 Goal</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .goal-card {
            background: linear-gradient(135deg, #e6e6ff, #f5f5ff);
            border-radius: 15px;
            padding: 25px 30px;
            font-size: 18px;
            line-height: 1.8;
            text-align: center;
            max-width: 900px;
            margin: 20px auto;
            box-shadow: 0px 5px 18px rgba(75, 0, 130, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .goal-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 10px 25px rgba(75, 0, 130, 0.3);
        }
        </style>

        <div class="goal-card">
            Empower financial institutions with an <b>AI-powered, data-driven solution</b> that enhances credit risk assessment, minimizes losses, and streamlines recovery strategies.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --- Project Details Section ---
    st.markdown("<h3 style='color:#4B0082; text-align:center; font-weight:bold;'>👨‍🏫 Project Details</h3>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        .project-card {
            background-color: #f5f5ff;
            border-radius: 15px;
            padding: 25px 30px;
            font-size: 17px;
            line-height: 1.7;
            box-shadow: 0px 5px 18px rgba(75, 0, 130, 0.15);
            max-width: 1000px;
            margin: 20px auto;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .project-card:hover {
            transform: translateY(-5px);
            box-shadow: 0px 10px 25px rgba(75, 0, 130, 0.25);
        }
        .project-links a {
            display: inline-block;
            margin-right: 15px;
            margin-top: 10px;
            padding: 8px 15px;
            background-color: #4B0082;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: background-color 0.3s ease;
        }
        .project-links a:hover {
            background-color: #6a1ab2;
        }
        </style>

        <div class="project-card">
            <b>Project Title:</b> CreditPath AI – Predictive Modeling for Loan Default Risk & Personalized Recovery Strategy<br>
            <b>Mentor:</b> Dr. N. Jagan Mohan<br>
            <b>Internship:</b> Infosys Springboard Internship 6.0 (AI Batch 1)<br>
            <b>Developed By:</b> Interns of AI Batch 1<br><br>
            <div class="project-links">
                <a href="https://github.com/springboardmentor891v/CreditPathAI" target="_blank">GitHub Repository</a>
                <a href="https://docs.google.com/presentation/d/1743-zttuDo6GzD-c47bCAFfF496LD7-6/edit?usp=sharing&ouid=113778290627699741743&rtpof=true&sd=true" target="_blank">Project PPT</a>
                <a href="https://drive.google.com/file/d/1z5Se4mBIWlF6qVVvrDrIQclzoxDJz3It/view?usp=drivesdk" target="_blank">Demo Video</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


    



# -------------------------
# Tab 2: Dataset
# -------------------------
with tab2:
    st.markdown("<h2 style='color:#4B0082; text-align:center;'>Dataset</h2>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state["uploaded_df"] = df
        st.success("✅ Dataset successfully uploaded!")
        
        # -------------------------
        # Dataset Preview
        # -------------------------
        st.markdown("<h3 style='color:#4B0082;'>Dataset Preview (First 10 Rows)</h3>", unsafe_allow_html=True)
        st.dataframe(df.head(10).style.set_table_styles([
            {'selector': 'th', 'props': [('color', 'black'), ('font-weight','bold')]},
            {'selector': 'td', 'props': [('color', 'black')]}
        ]))

         # -------------------------
        # Dataset Info as Table
        # -------------------------
        st.markdown("<h3 style='color:#4B0082;'>Dataset Info</h3>", unsafe_allow_html=True)
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue().splitlines()
        
        info_data = []
        for line in info_str[5:]:
            parts = line.split()
            if len(parts) >= 4:
                info_data.append({
                    "Column": parts[0],
                    "Non-Null Count": parts[1],
                    "Dtype": parts[-1]
                })
        
        if info_data:
            df_info_table = pd.DataFrame(info_data)
            st.dataframe(
                df_info_table.style.set_properties(**{'color':'#000', 'background-color':'#f9f9f9', 'text-align':'center'})
                .set_table_styles([{'selector':'th', 'props':[('color','black'), ('font-weight','bold'), ('background-color','#cccccc')]}])
            )

        # -------------------------
        # Numeric Describe + Correlation
        # -------------------------
        numeric_cols = df.select_dtypes(include=['int64','float64']).columns
        if len(numeric_cols) > 0:
            st.markdown("<h3 style='color:#4B0082;'>Numerical Features Summary</h3>", unsafe_allow_html=True)
            st.dataframe(df[numeric_cols].describe().style.set_table_styles([
                {'selector': 'th', 'props': [('color', 'black'), ('font-weight','bold')]},
                {'selector': 'td', 'props': [('color', 'black')]}
            ]))
            
            st.markdown("<h3 style='color:#4B0082;'>Correlation Heatmap</h3>", unsafe_allow_html=True)
            corr_matrix = df[numeric_cols].corr()
            plt.figure(figsize=(10,8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            st.pyplot(plt)

        # -------------------------
        # Categorical Feature Distribution (Side by Side)
        # -------------------------
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            st.markdown("<h3 style='color:#4B0082;'>Categorical Feature Distribution</h3>", unsafe_allow_html=True)
            n_cols = 3  # charts per row
            for i in range(0, len(categorical_cols), n_cols):
                cols = st.columns(n_cols)
                for j, col in enumerate(categorical_cols[i:i+n_cols]):
                    with cols[j]:
                        st.markdown(f"<b style='color:#4B0082;'>{col}</b>", unsafe_allow_html=True)
                        counts = df[col].value_counts()
                        plt.figure(figsize=(4,3))
                        sns.barplot(x=counts.index, y=counts.values, palette='Purples')
                        plt.xticks(rotation=45, ha='right')
                        plt.ylabel('Count')
                        plt.xlabel('')
                        plt.tight_layout()
                        st.pyplot(plt)
        else:
            st.info("No categorical columns available for distribution plots.")

    else:
        st.info("Please upload a dataset to explore.")
# -------------------------
# Tab 3: Model Metrics
# -------------------------
with tab3:
    st.markdown("<h2 style='text-align:center; color:#4B0082;'>Model Performance Metrics</h2>", unsafe_allow_html=True)
    
    if "uploaded_df" in st.session_state:
        df = st.session_state["uploaded_df"]
        target_col = st.selectbox("Select Target Column", df.columns, key="target_col_metrics")
        model_choice_metrics = st.selectbox("Select Model", list(models.keys()), key="model_metrics")
        
        if st.button("Generate Metrics"):
            try:
                # Load model
                with open(models[model_choice_metrics], "rb") as f:
                    model = pickle.load(f)

                # Load features used during training
                feature_file = models[model_choice_metrics].replace(".pkl","_features.pkl")
                with open(feature_file, "rb") as f:
                    master_features = pickle.load(f)

                # Prepare X
                X = pd.get_dummies(df.drop(columns=[target_col]), drop_first=True)
                X = X.reindex(columns=master_features, fill_value=0)
                X = X.fillna(0)

                # Scale numeric features
                numeric_features = ['loan_amount', 'rate_of_interest', 'term', 'property_value',
                                    'income', 'credit_score', 'ltv', 'dtir1', 'loan_limit_ncf',
                                    'submission_of_application_to_inst']
                scaler = StandardScaler()
                for col in numeric_features:
                    if col in X.columns:
                        X[col] = scaler.fit_transform(X[[col]])

                y = df[target_col]
                y_pred = model.predict(X)

                # -------------------------
                # Confusion Matrix + ROC side by side
                # -------------------------
                col1, col2 = st.columns(2)

                # Confusion Matrix
                with col1:
                    st.subheader("Confusion Matrix")
                    fig_cm, ax_cm = plt.subplots(figsize=(4,3))
                    cm = confusion_matrix(y, y_pred)
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                    ax_cm.set_xlabel("Predicted")
                    ax_cm.set_ylabel("Actual")
                    st.pyplot(fig_cm)

                # ROC Curve
                with col2:
                    if hasattr(model, "predict_proba"):
                        st.subheader("ROC Curve / AUC")
                        y_prob = model.predict_proba(X)[:,1]
                        fpr, tpr, _ = roc_curve(y, y_prob)
                        roc_auc = auc(fpr, tpr)
                        fig_roc, ax_roc = plt.subplots(figsize=(4,3))
                        ax_roc.plot(fpr, tpr, color='green', lw=2, label=f'ROC (AUC = {roc_auc:.2f})')  # Green color
                        ax_roc.plot([0,1],[0,1], color='gray', linestyle='--')
                        ax_roc.set_xlabel('False Positive Rate')
                        ax_roc.set_ylabel('True Positive Rate')
                        ax_roc.legend(loc="lower right")
                        st.pyplot(fig_roc)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.info("Upload a dataset first.")

# -------------------------
# Tab 4: Prediction
# -------------------------
with tab4:
    
    # -------------------------
    # Sidebar: Model Selection
    # -------------------------
    st.markdown(
        """
        <style>
        .model-title {
            color: #4B0082;
            font-weight: bold;
            font-size: 20px;
            margin-bottom: -25px;  /* reduces space */
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.sidebar.markdown("<p class='model-title'>Choose Model</p>", unsafe_allow_html=True)
    model_choice_pred = st.sidebar.selectbox(" ", list(models.keys()), key="model_pred")

    st.sidebar.markdown("<h3 style='color:#4B0082; font-weight:bold; margin-bottom:5px; margin-top:5px;'>Enter Loan Details</h3>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr style='border:1px solid lightgrey; margin-top:0px; margin-bottom:5px;'>", unsafe_allow_html=True)


    # -------------------------
    # Sidebar: Numeric Inputs
    # -------------------------
    st.sidebar.markdown("<h3 style='color:#4B0082; font-weight:bold;'>Numeric Inputs</h3>", unsafe_allow_html=True)
    loan_amount = st.sidebar.number_input("Loan Amount", 1000, 10000000, 100000, step=1000)
    term = st.sidebar.number_input("Term (Months)", 1, 1000, 360, step=1)
    submission_of_application_to_inst = st.sidebar.number_input("Submission of Application (days)", 0, 365, 10, step=1)
    loan_limit_ncf = st.sidebar.number_input("Loan Limit NCF", 0, 10000000, 50000, step=1000)
    rate_of_interest = st.sidebar.number_input("Rate of Interest (%)", 0.0, 20.0, 4.0, step=0.1)
    property_value = st.sidebar.number_input("Property Value", 1000.0, 10000000.0, 300000.0, step=1000.0)
    income = st.sidebar.number_input("Income", 0.0, 10000000.0, 50000.0, step=1000.0)
    credit_score = st.sidebar.number_input("Credit Score", 0.0, 1000.0, 700.0, step=1.0)
    ltv = st.sidebar.number_input("Loan to Value (LTV)", 0.0, 200.0, 80.0, step=0.1)
    dtir1 = st.sidebar.number_input("Debt-to-Income Ratio (%)", 0.0, 100.0, 30.0, step=0.1)

    # -------------------------
    # Sidebar: Binary Inputs
    # -------------------------
    st.sidebar.markdown("<h3 style='color:#4B0082; font-weight:bold; margin-top:20px;'>Binary Inputs (0/1)</h3>", unsafe_allow_html=True)
    has_co_applicant = st.sidebar.selectbox("Has Co-Applicant", [0,1])
    approv_in_adv_pre = st.sidebar.selectbox("Approved in Advance", [0,1])
    credit_worthiness_l2 = st.sidebar.selectbox("Credit Worthiness L2", [0,1])
    business_or_commercial_nob_c = st.sidebar.selectbox("Business or Commercial", [0,1])
    neg_ammortization_not_neg = st.sidebar.selectbox("Negative Amortization Not Negative", [0,1])
    interest_only_not_int = st.sidebar.selectbox("Interest Only Not Interest", [0,1])
    lump_sum_payment_not_lpsm = st.sidebar.selectbox("Lump Sum Payment Not LPSM", [0,1])

    # -------------------------
    # Sidebar: Categorical Inputs
    # -------------------------
    st.sidebar.markdown("<h3 style='color:#4B0082; font-weight:bold; margin-top:20px;'>Categorical Inputs</h3>", unsafe_allow_html=True)
    gender = st.sidebar.selectbox("Gender", ["Female","Joint","Male","Not Available"])
    gender_Female = 1 if gender=="Female" else 0
    gender_Joint = 1 if gender=="Joint" else 0
    gender_Male = 1 if gender=="Male" else 0
    gender_Sex_Not_Available = 1 if gender=="Not Available" else 0

    loan_type = st.sidebar.selectbox("Loan Type", ["Type1","Type2","Type3"])
    loan_type_type2 = 1 if loan_type=="Type2" else 0
    loan_type_type3 = 1 if loan_type=="Type3" else 0

    loan_purpose = st.sidebar.selectbox("Loan Purpose", ["Purpose1","Purpose2","Purpose3","Purpose4"])
    loan_purpose_p2 = 1 if loan_purpose=="Purpose2" else 0
    loan_purpose_p3 = 1 if loan_purpose=="Purpose3" else 0
    loan_purpose_p4 = 1 if loan_purpose=="Purpose4" else 0

    occupancy_type = st.sidebar.selectbox("Occupancy Type", ["PR","SR"])
    occupancy_type_pr = 1 if occupancy_type=="PR" else 0
    occupancy_type_sr = 1 if occupancy_type=="SR" else 0

    total_units = st.sidebar.selectbox("Total Units", ["1U","2U","3U","4U"])
    total_units_2U = 1 if total_units=="2U" else 0
    total_units_3U = 1 if total_units=="3U" else 0
    total_units_4U = 1 if total_units=="4U" else 0

    credit_type = st.sidebar.selectbox("Credit Type", ["CRIF","EQUI","EXP"])
    credit_type_CRIF = 1 if credit_type=="CRIF" else 0
    credit_type_EQUI = 1 if credit_type=="EQUI" else 0
    credit_type_EXP = 1 if credit_type=="EXP" else 0

    age_group = st.sidebar.selectbox("Age Group", ["<25","35-44","45-54","55-64","65-74",">74"])
    age__25 = 1 if age_group=="<25" else 0
    age_35_44 = 1 if age_group=="35-44" else 0
    age_45_54 = 1 if age_group=="45-54" else 0
    age_55_64 = 1 if age_group=="55-64" else 0
    age_65_74 = 1 if age_group=="65-74" else 0
    age__74 = 1 if age_group==">74" else 0

    region = st.sidebar.selectbox("Region", ["North East","Central","South"])
    region_North_East = 1 if region=="North East" else 0
    region_central = 1 if region=="Central" else 0
    region_south = 1 if region=="South" else 0

    # -------------------------
    # Input DataFrame & Scaling
    # -------------------------
    input_data = pd.DataFrame([{
        'loan_amount': loan_amount, 'rate_of_interest': rate_of_interest, 'term': term,
        'property_value': property_value, 'income': income, 'credit_score': credit_score,
        'ltv': ltv, 'dtir1': dtir1, 'has_co_applicant': has_co_applicant,
        'loan_limit_ncf': loan_limit_ncf, 'gender_Joint': gender_Joint, 'gender_Male': gender_Male,
        'gender_Sex_Not_Available': gender_Sex_Not_Available, 'approv_in_adv_pre': approv_in_adv_pre,
        'loan_type_type2': loan_type_type2, 'loan_type_type3': loan_type_type3,
        'loan_purpose_p2': loan_purpose_p2, 'loan_purpose_p3': loan_purpose_p3, 'loan_purpose_p4': loan_purpose_p4,
        'credit_worthiness_l2': credit_worthiness_l2, 'business_or_commercial_nob_c': business_or_commercial_nob_c,
        'neg_ammortization_not_neg': neg_ammortization_not_neg, 'interest_only_not_int': interest_only_not_int,
        'lump_sum_payment_not_lpsm': lump_sum_payment_not_lpsm, 'occupancy_type_pr': occupancy_type_pr,
        'occupancy_type_sr': occupancy_type_sr, 'total_units_2U': total_units_2U,
        'total_units_3U': total_units_3U, 'total_units_4U': total_units_4U,
        'credit_type_CRIF': credit_type_CRIF, 'credit_type_EQUI': credit_type_EQUI,
        'credit_type_EXP': credit_type_EXP, 'age_35_44': age_35_44, 'age_45_54': age_45_54,
        'age_55_64': age_55_64, 'age_65_74': age_65_74, 'age__25': age__25, 'age__74': age__74,
        'submission_of_application_to_inst': submission_of_application_to_inst,
        'region_North_East': region_North_East, 'region_central': region_central, 'region_south': region_south
    }])

    # Scale numeric features
    numeric_features = ['loan_amount', 'rate_of_interest', 'term', 'property_value',
                        'income', 'credit_score', 'ltv', 'dtir1', 'loan_limit_ncf',
                        'submission_of_application_to_inst']
    input_data[numeric_features] = StandardScaler().fit_transform(input_data[numeric_features])

    # Initialize session state for prediction
    if "predicted" not in st.session_state:
        st.session_state.predicted = False

    # Predict Button
    if st.sidebar.button("Predict"):
        try:
            with open(models[model_choice_pred], "rb") as f:
                model = pickle.load(f)
            prediction = model.predict(input_data)[0]

            # Set session state to True so info disappears
            st.session_state.predicted = True

            # Display main heading
            st.markdown("<h3 style='color:#4B0082;'>Prediction Results</h3>", unsafe_allow_html=True)

            # Display probabilities
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(input_data)[0]
                col1, col2 = st.columns(2)
                col1.markdown(f"<div style='background-color:#d0e1f9; padding:15px; border-radius:10px; text-align:center;'>"
                            f"<h4>Probability of NOT Default</h4>"
                            f"<p style='font-size:18px; color:#1f77b4;'>{proba[0]*100:.2f}%</p></div>", 
                            unsafe_allow_html=True)
                col2.markdown(f"<div style='background-color:#fde0dc; padding:15px; border-radius:10px; text-align:center;'>"
                            f"<h4>Probability of Default</h4>"
                            f"<p style='font-size:18px; color:#d62728;'>{proba[1]*100:.2f}%</p></div>", 
                            unsafe_allow_html=True)

            # Display final prediction
            if prediction == 1:
                st.markdown(f"<div style='background-color:#ffd6d6; padding:20px; border-radius:10px; text-align:center; margin-top:20px;'>"
                            f"<h2 style='color:#d62728;'>Loan will DEFAULT</h2></div>", 
                            unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:#d6ffd6; padding:20px; border-radius:10px; text-align:center; margin-top:20px;'>"
                            f"<h2 style='color:#2ca02c;'>Loan will NOT Default</h2></div>", 
                            unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

    # Show info only if prediction not done yet
    if not st.session_state.predicted:
        st.info("Enter applicant details in the sidebar and click the Predict button to see results.")

