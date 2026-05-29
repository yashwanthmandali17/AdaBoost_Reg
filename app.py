import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AdaBoost Insurance Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

h1, h2, h3 { font-family: 'DM Serif Display', serif; }

.main { background: #0f1117; }

.metric-card {
    background: linear-gradient(135deg, #1a1d2e 0%, #252844 100%);
    border: 1px solid #3a3f6b;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.metric-card .label { color: #8b92c9; font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; }
.metric-card .value { color: #e8eaf6; font-size: 2rem; font-weight: 600; margin-top: 4px; }

.pred-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #0d2137 100%);
    border: 1px solid #2e6da4;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin-top: 1rem;
}
.pred-box .pred-label { color: #63b3ed; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.12em; }
.pred-box .pred-value { color: #90cdf4; font-size: 3.2rem; font-weight: 700; font-family: 'DM Serif Display', serif; margin-top: 6px; }

.section-header {
    border-left: 4px solid #667eea;
    padding-left: 0.8rem;
    margin: 1.5rem 0 1rem;
    color: #c3caf5;
    font-size: 1.1rem;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-size: 1rem;
    font-weight: 600;
    width: 100%;
    transition: opacity 0.2s;
}
.stButton>button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)


# ─── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model  = joblib.load("models/adaboost_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    le_sex    = joblib.load("models/le_sex.pkl")
    le_region = joblib.load("models/le_region.pkl")
    return model, scaler, le_sex, le_region

@st.cache_data
def load_data():
    df = pd.read_csv("data/insurance.csv")
    return df

model, scaler, le_sex, le_region = load_artifacts()
df = load_data()


# ─── Sidebar Navigation ───────────────────────────────────────────────────────
st.sidebar.markdown("## 🏥 AdaBoost\nInsurance Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio("Navigate", ["🔮 Predict", "📊 EDA", "📈 Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption("Model: AdaBoost Regressor\nBase: DecisionTree (depth=4)\nEstimators: 100 | LR: 0.1")


# ═══════════════════════════════════════════════════════════════════
#  PAGE 1 — PREDICT
# ═══════════════════════════════════════════════════════════════════
if page == "🔮 Predict":
    st.title("🏥 Insurance Charge Predictor")
    st.markdown("Enter patient details below to estimate annual insurance charges.")

    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="section-header">👤 Patient Information</div>', unsafe_allow_html=True)
        age      = st.slider("Age", 18, 65, 35)
        sex      = st.selectbox("Sex", ["male", "female"])
        bmi      = st.slider("BMI", 15.0, 50.0, 26.5, step=0.1)
        bp       = st.slider("Blood Pressure (mmHg)", 60, 130, 80)
        children = st.slider("Number of Children", 0, 5, 1)
        smoker   = st.radio("Smoker?", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        region   = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    with col_r:
        st.markdown('<div class="section-header">⚡ Prediction</div>', unsafe_allow_html=True)

        if st.button("Predict Insurance Charges"):
            sex_enc    = le_sex.transform([sex])[0]
            region_enc = le_region.transform([region])[0]
            features   = np.array([[age, bmi, bp, children, smoker, sex_enc, region_enc]])
            features_sc = scaler.transform(features)
            prediction  = model.predict(features_sc)[0]

            st.markdown(f"""
            <div class="pred-box">
                <div class="pred-label">Estimated Annual Charges</div>
                <div class="pred-value">${prediction:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("**Input Summary**")
            summary = pd.DataFrame({
                "Feature": ["Age","Sex","BMI","Blood Pressure","Children","Smoker","Region"],
                "Value"  : [age, sex, bmi, bp, children, "Yes" if smoker else "No", region]
            })
            st.dataframe(summary, use_container_width=True, hide_index=True)

            # Risk indicator
            if prediction > 40000:
                st.error("⚠️ High Risk Profile — Charges above $40,000")
            elif prediction > 20000:
                st.warning("🟡 Moderate Risk — Charges between $20k–$40k")
            else:
                st.success("✅ Low Risk — Charges below $20,000")
        else:
            st.info("👈 Fill in the details and click **Predict** to get started.")


# ═══════════════════════════════════════════════════════════════════
#  PAGE 2 — EDA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis")

    # Overview metrics
    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["Records", "Features", "Avg Charges", "Smoker %"],
        [len(df), df.shape[1], f"${df['charges'].mean():,.0f}", f"{df['smoker'].mean()*100:.1f}%"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Data Preview", "📈 Distributions", "🔗 Correlations", "🔍 Feature vs Target"])

    with tab1:
        st.dataframe(df.head(20), use_container_width=True)
        st.markdown("**Statistical Summary**")
        st.dataframe(df.describe().round(2), use_container_width=True)

    with tab2:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), facecolor='#0f1117')
        plt.rcParams['text.color'] = 'white'
        cols_plot = ['age', 'bmi', 'blood_pressure', 'children', 'charges']
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe', '#43e97b']
        for ax, col, clr in zip(axes.flatten(), cols_plot, colors):
            ax.hist(df[col], bins=30, color=clr, edgecolor='#0f1117', alpha=0.9)
            ax.set_title(col.replace('_', ' ').title(), color='white', fontsize=11)
            ax.set_facecolor('#1a1d2e')
            ax.tick_params(colors='#aaa')
            for spine in ax.spines.values():
                spine.set_edgecolor('#333')
        axes[1][2].set_visible(False)
        fig.suptitle('Numerical Feature Distributions', color='white', fontsize=14, y=1.01)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        df_enc = df.copy()
        df_enc['sex']    = LabelEncoder().fit_transform(df_enc['sex'])
        df_enc['region'] = LabelEncoder().fit_transform(df_enc['region'])
        fig, ax = plt.subplots(figsize=(9, 7), facecolor='#1a1d2e')
        sns.heatmap(df_enc.corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
        ax.set_facecolor('#1a1d2e')
        ax.tick_params(colors='white')
        plt.title('Correlation Matrix', color='white', fontsize=13)
        st.pyplot(fig)
        plt.close()

    with tab4:
        feat = st.selectbox("Select Feature", ['age', 'bmi', 'blood_pressure', 'children'])
        hue  = st.selectbox("Color By", ['smoker', 'sex', 'region'])
        fig, ax = plt.subplots(figsize=(10, 5), facecolor='#1a1d2e')
        ax.set_facecolor('#1a1d2e')
        sns.scatterplot(x=feat, y='charges', hue=hue, data=df, ax=ax,
                        palette='cool', alpha=0.7, s=60)
        ax.tick_params(colors='white')
        ax.set_xlabel(feat, color='white')
        ax.set_ylabel('Charges', color='white')
        ax.set_title(f'{feat} vs Charges', color='white')
        ax.legend(facecolor='#252844', labelcolor='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()


# ═══════════════════════════════════════════════════════════════════
#  PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    # Re-prepare data for evaluation
    @st.cache_data
    def get_evaluation():
        df_ = df.copy()
        df_['sex']    = LabelEncoder().fit_transform(df_['sex'])
        df_['region'] = LabelEncoder().fit_transform(df_['region'])
        X = df_[['age','bmi','blood_pressure','children','smoker','sex','region']]
        y = df_['charges']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        sc = StandardScaler()
        X_test_sc = sc.fit_transform(X_train)   # same fit as training
        X_test_sc = sc.transform(X_test)
        y_pred = model.predict(X_test_sc)
        return y_test.values, y_pred

    y_test, y_pred = get_evaluation()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    c1, c2, c3, c4 = st.columns(4)
    for col, label, val in zip(
        [c1, c2, c3, c4],
        ["R² Score", "RMSE", "MAE", "MAPE"],
        [f"{r2:.4f}", f"${rmse:,.0f}", f"${mae:,.0f}", f"{mape:.2f}%"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="label">{label}</div>
            <div class="value">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["🎯 Actual vs Predicted", "📉 Residuals", "🌟 Feature Importance"])

    dark_bg = '#0f1117'
    card_bg = '#1a1d2e'

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 6), facecolor=dark_bg)
        ax.set_facecolor(card_bg)
        ax.scatter(y_test, y_pred, alpha=0.55, color='#667eea', edgecolors='none', s=55)
        mn, mx = y_test.min(), y_test.max()
        ax.plot([mn, mx], [mn, mx], 'r--', lw=2, label='Perfect Fit')
        ax.set_xlabel('Actual Charges', color='white')
        ax.set_ylabel('Predicted Charges', color='white')
        ax.set_title('Actual vs Predicted', color='white', fontsize=13)
        ax.tick_params(colors='#aaa')
        ax.legend(facecolor=card_bg, labelcolor='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()

    with tab2:
        residuals = y_test - y_pred
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=dark_bg)
        for ax in axes: ax.set_facecolor(card_bg)

        axes[0].scatter(y_pred, residuals, alpha=0.5, color='#f093fb', s=50)
        axes[0].axhline(0, color='red', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted', color='white'); axes[0].set_ylabel('Residuals', color='white')
        axes[0].set_title('Residuals vs Predicted', color='white')
        axes[0].tick_params(colors='#aaa')

        axes[1].hist(residuals, bins=35, color='#4facfe', edgecolor=dark_bg, alpha=0.9)
        axes[1].set_xlabel('Residual', color='white'); axes[1].set_ylabel('Frequency', color='white')
        axes[1].set_title('Residual Distribution', color='white')
        axes[1].tick_params(colors='#aaa')

        for ax in axes:
            for sp in ax.spines.values(): sp.set_edgecolor('#333')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        feat_names = ['age', 'bmi', 'blood_pressure', 'children', 'smoker', 'sex', 'region']
        importances = pd.Series(model.feature_importances_, index=feat_names).sort_values()
        fig, ax = plt.subplots(figsize=(9, 5), facecolor=dark_bg)
        ax.set_facecolor(card_bg)
        colors_ = plt.cm.viridis(np.linspace(0.3, 1, len(importances)))
        importances.plot(kind='barh', ax=ax, color=colors_, edgecolor=dark_bg)
        ax.set_xlabel('Importance', color='white')
        ax.set_title('AdaBoost Feature Importances', color='white', fontsize=13)
        ax.tick_params(colors='white')
        for sp in ax.spines.values(): sp.set_edgecolor('#333')
        st.pyplot(fig)
        plt.close()
