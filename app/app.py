import streamlit as st
import pandas as pd

from utils import (
    load_model,
    load_preprocessor
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Customer Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

/* ---------- Main ---------- */

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ---------- Headings ---------- */

h1,h2,h3{
    color:white;
    font-weight:700;
}

/* ---------- Cards ---------- */

.card{

    background-color:#1d212b;

    padding:22px;

    border-radius:14px;

    border:1px solid #313744;

    margin-bottom:18px;

    box-shadow:0px 0px 10px rgba(0,0,0,0.18);

}

.metric-card{

    background:#1d212b;

    border-left:6px solid #4CAF50;

    padding:18px;

    border-radius:12px;

}

/* ---------- Sidebar ---------- */

section[data-testid="stSidebar"]{

    background:#1F2430;

}

/* ---------- Button ---------- */

div.stButton > button{

    width:100%;

    border-radius:12px;

    padding:14px;

    font-size:17px;

    font-weight:bold;

    background:#FF4B4B;

    color:white;

    border:none;

}

div.stButton > button:hover{

    background:#ff3030;

}

/* ---------- Progress ---------- */

.stProgress > div > div{

    background:#00C853;

}

/* ---------- Horizontal Line ---------- */

hr{

    border:1px solid #30363d;

}

</style>
""", unsafe_allow_html=True)

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model()

preprocessor = load_preprocessor()

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📊 Churn Predictor")

    st.caption("Machine Learning Dashboard")

    st.markdown("---")

    st.subheader("Model Information")

    c1,c2 = st.columns(2)

    with c1:
        st.metric("Accuracy","77%")

    with c2:
        st.metric("ROC-AUC","0.84")

    st.write("**Algorithm**")

    st.metric(
    label="Algorithm",
    value="Gradient Boosting"
    )

    st.write("**Dataset**")

    st.metric(
    label="Dataset",
    value="IBM Telco"
    )

    st.markdown("---")

    st.subheader("Features")

    st.markdown("""
✅ Customer Churn Prediction

✅ Machine Learning Pipeline

✅ Data Preprocessing

✅ Risk Assessment

✅ Probability Score

✅ Business Recommendation

✅ Interactive Dashboard
""")

    st.markdown("---")

    st.subheader("Developer")

    st.write("**Deepak Kumar Jha**")

    st.caption("AI & Data Science")

# =====================================================
# HEADER
# =====================================================

st.markdown("""
# 📊 Customer Churn Prediction Dashboard

### Predict Telecom Customer Churn using Machine Learning

Analyze customer information, estimate churn probability, identify risk factors,
and receive actionable business recommendations.

""")

col1, col2, col3 = st.columns([2,1,1])

with col1:
    st.info(
        "💡 **Tip:** Complete the customer information below and click "
        "**Predict Customer Churn**."
    )

with col2:
    st.metric(
        "Model Accuracy",
        "77%"
    )

with col3:
    st.metric(
        "ROC-AUC",
        "0.84"
    )

st.divider()

# =====================================================
# HELPER FUNCTION
# =====================================================

def risk_level(prob):

    if prob < 0.30:
        return "Low","green"

    elif prob < 0.70:
        return "Medium","orange"

    else:
        return "High","red"

# =====================================================
# CUSTOMER INPUT SECTION
# =====================================================

with st.container():

    st.markdown("## 📋 Customer Information")

    st.caption(
        "Provide the customer details below to estimate the probability of churn."
    )

    st.write("")
    # =================================================
    # CUSTOMER PROFILE
    # =================================================

    with st.expander("👤 Customer Profile", expanded=False):

        st.caption(
            "Basic demographic information about the customer."
        )

        col1, col2 = st.columns(2)

        with col1:

            gender = st.selectbox(
                "Gender",
                ["Male", "Female"]
            )

            senior_citizen = st.selectbox(
                "Senior Citizen",
                ["No", "Yes"]
            )

            partner = st.selectbox(
                "Partner",
                ["No", "Yes"]
            )

        with col2:

            dependents = st.selectbox(
                "Dependents",
                ["No", "Yes"]
            )

            tenure = st.slider(
                "Customer Tenure (Months)",
                0,
                72,
                24
            )

    # =================================================
    # PHONE SERVICES
    # =================================================

    with st.expander("📞 Phone Services", expanded=False):

        st.caption(
            "Telephone subscription and multi-line services."
        )

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        if phone_service == "Yes":

            multiple_lines = st.checkbox(
                "Customer has Multiple Phone Lines"
            )

            multiple_lines = (
                "Yes"
                if multiple_lines
                else "No"
            )

        else:

            multiple_lines = "No phone service"

            st.info("Phone service is disabled.")

    # =================================================
    # INTERNET SERVICES
    # =================================================

    with st.expander("🌐 Internet Services", expanded=False):

        st.caption(
            "Internet connection type and additional online services."
        )

        internet_service = st.selectbox(

            "Internet Service",

            [

                "DSL",

                "Fiber optic",

                "No"

            ]

        )

        if internet_service != "No":

            c1, c2, c3 = st.columns(3)

            with c1:

                online_security = st.checkbox(
                    "Online Security"
                )

                device_protection = st.checkbox(
                    "Device Protection"
                )

            with c2:

                online_backup = st.checkbox(
                    "Online Backup"
                )

                tech_support = st.checkbox(
                    "Tech Support"
                )

            with c3:

                streaming_tv = st.checkbox(
                    "Streaming TV"
                )

                streaming_movies = st.checkbox(
                    "Streaming Movies"
                )

            online_security = (
                "Yes"
                if online_security
                else "No"
            )

            online_backup = (
                "Yes"
                if online_backup
                else "No"
            )

            device_protection = (
                "Yes"
                if device_protection
                else "No"
            )

            tech_support = (
                "Yes"
                if tech_support
                else "No"
            )

            streaming_tv = (
                "Yes"
                if streaming_tv
                else "No"
            )

            streaming_movies = (
                "Yes"
                if streaming_movies
                else "No"
            )

        else:

            st.info(
                "Internet service is disabled."
            )

            online_security = "No internet service"

            online_backup = "No internet service"

            device_protection = "No internet service"

            tech_support = "No internet service"

            streaming_tv = "No internet service"

            streaming_movies = "No internet service"

    # =================================================
    # BILLING
    # =================================================

    with st.expander("💳 Billing Information", expanded=False):

        st.caption(
            "Billing details, contract type, and payment preferences."
        )

        col1, col2 = st.columns(2)

        with col1:

            contract = st.selectbox(

                "Contract",

                [

                    "Month-to-month",

                    "One year",

                    "Two year"

                ]

            )

            paperless_billing = st.selectbox(

                "Paperless Billing",

                [

                    "Yes",

                    "No"

                ]

            )

        with col2:

            payment_method = st.selectbox(

                "Payment Method",

                [

                    "Electronic check",

                    "Mailed check",

                    "Bank transfer (automatic)",

                    "Credit card (automatic)"

                ]

            )

            monthly_charges = st.number_input(

                "Monthly Charges ($)",

                min_value=18.25,

                max_value=120.00,

                value=70.00,

                step=0.50

            )

        total_charges = round(
            tenure * monthly_charges,
            2
        )

        c1, c2 = st.columns(2)

        with c1:

            st.metric(

                "Estimated Total Charges",

                f"${total_charges:,.2f}"

            )

        with c2:

            st.metric(

                "Customer Tenure",

                f"{tenure} Months"

            )

# =====================================================
# PREDICT BUTTON
# =====================================================

st.markdown("<br>", unsafe_allow_html=True)

predict_button = st.button(

    "🚀 Predict Customer Churn",

    use_container_width=True,

    type="primary"

)

# =====================================================
# PREDICTION
# =====================================================

if predict_button:

    input_data = pd.DataFrame({

        "gender":[gender],
        "SeniorCitizen":[1 if senior_citizen=="Yes" else 0],
        "Partner":[partner],
        "Dependents":[dependents],
        "tenure":[tenure],
        "PhoneService":[phone_service],
        "MultipleLines":[multiple_lines],
        "InternetService":[internet_service],
        "OnlineSecurity":[online_security],
        "OnlineBackup":[online_backup],
        "DeviceProtection":[device_protection],
        "TechSupport":[tech_support],
        "StreamingTV":[streaming_tv],
        "StreamingMovies":[streaming_movies],
        "Contract":[contract],
        "PaperlessBilling":[paperless_billing],
        "PaymentMethod":[payment_method],
        "MonthlyCharges":[monthly_charges],
        "TotalCharges":[total_charges]

    })

    processed = preprocessor.transform(input_data)

    prediction = model.predict(processed)[0]

    probability = model.predict_proba(processed)[0][1]

    risk, color = risk_level(probability)

    st.divider()

    st.header("📊 Prediction Dashboard")

    # ==========================================
    # TOP METRIC CARDS
    # ==========================================
    col1, col2, col3 = st.columns(3)

    risk_probability = probability * 100

    if prediction == 1:
        prediction_text = "Likely to Churn"
        prediction_color = "error"
    else:
        prediction_text = "Likely to Stay"
        prediction_color = "success"

    if risk_probability < 30:
        risk_level = "🟢 Low"
    elif risk_probability < 60:
        risk_level = "🟡 Medium"
    else:
        risk_level = "🔴 High"

    with col1:
        if prediction == 1:
            st.error(f"### 🔴 {prediction_text}")
        else:
            st.success(f"### 🟢 {prediction_text}")

    with col2:
        st.metric(
            "Churn Probability",
            f"{risk_probability:.2f}%"
        )

    with col3:
        st.info(f"### {risk_level} Risk")

    confidence = max(
    risk_probability,
    100-risk_probability
    )

    if prediction == 1:
        st.write("### Churn Probability")
        st.progress(risk_probability / 100)
        st.caption(f"{risk_probability:.2f}% Probability of Churn")
    else:
        st.write("### Customer Retention Probability")
        st.progress((100 - risk_probability) / 100)
        st.caption(f"{100 - risk_probability:.2f}% Probability of Staying")  

    # ==========================================
    # EXECUTIVE SUMMARY
    # ==========================================

    st.markdown("---")

    st.subheader("📋 Executive Summary")

    if prediction==1:

        st.error(f"""
        This customer has a **{risk_probability:.1f}% probability**
        of leaving the company.

        The current subscription profile indicates elevated retention
        risk.

        Early customer engagement is recommended.
        """)

    else:

        st.success(f"""
        This customer has a **{risk_probability:.1f}% probability**
        of churn.

        The overall customer profile appears stable.

        Continue normal engagement strategies.
        """)

    # ==========================================
    # BUSINESS RECOMMENDATION
    # ==========================================

    st.markdown("---")

    st.subheader("💡 Business Recommendations")

    if prediction==1:

        left,right = st.columns(2)

        with left:

            st.info("""
🎁 Offer loyalty discounts

📞 Contact customer personally

💰 Provide retention offer

⭐ Promote annual contract
""")

        with right:

            st.info("""
📡 Review internet service quality

🤝 Improve customer engagement

📈 Track customer feedback

🎯 Monitor account closely
""")

    else:

        left,right = st.columns(2)

        with left:

            st.success("""
✅ Continue quality service

🎁 Offer loyalty rewards

⭐ Suggest premium plans

😊 Maintain satisfaction
""")

        with right:

            st.success("""
📣 Encourage referrals

📦 Promote bundled services

💬 Periodic follow-up

📈 Upsell additional features
""")

    # ==========================================
    # KEY FACTORS
    # ==========================================

    st.markdown("---")

    st.subheader("🔍 Key Factors")

    factors=[]

    if contract=="Month-to-month":

        factors.append("📄 Month-to-month Contract")

    if tenure<12:

        factors.append("📅 Low Customer Tenure")

    if internet_service=="Fiber optic":

        factors.append("🌐 Fiber Optic Internet")

    if monthly_charges>80:

        factors.append("💲 High Monthly Charges")

    if senior_citizen=="Yes":

        factors.append("👤 Senior Citizen")

    if partner=="No":

        factors.append("👥 No Partner")

    if dependents=="No":

        factors.append("🏠 No Dependents")

    if len(factors)==0:

        st.success("No major churn indicators detected.")

    else:

        cols = st.columns(min(3, len(factors)))

        for i, factor in enumerate(factors):

            with cols[i % 3]:

                st.info(factor)

    # ==========================================
    # CUSTOMER SNAPSHOT
    # ==========================================

    st.markdown("---")

    st.subheader("📋 Customer Snapshot")

    snapshot = pd.DataFrame({

        "Feature":[

            "Contract",

            "Internet",

            "Monthly Charges",

            "Tenure",

            "Payment"

        ],

        "Value":[

            contract,

            internet_service,

            f"${monthly_charges:.2f}",

            f"{tenure} Months",

            payment_method

        ]

    })
    st.table(snapshot)

# =====================================================
# FOOTER
# =====================================================

st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.caption("👨‍💻 Developed By")
    st.write("**Deepak Kumar Jha**")

with col2:
    st.caption("🛠 Tech Stack")
    st.write("Python • Scikit-Learn • Streamlit")

with col3:
    st.caption("🤖 ML Model")
    st.write("Gradient Boosting Classifier")