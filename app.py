import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Medical Insurance Cost Dashboard",
    page_icon="💊",
    layout="wide"
)

# -----------------------------
# Custom Dark Theme Styling
# -----------------------------
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(180deg, #0B1020 0%, #111827 100%);
        color: #F9FAFB;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        border: 1px solid rgba(255,255,255,0.08);
        padding: 16px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
    }

    div[data-testid="stMetricLabel"] {
        color: #9CA3AF;
    }

    div[data-testid="stMetricValue"] {
        color: #F9FAFB;
    }

    .chart-card {
        background: linear-gradient(135deg, #111827 0%, #1F2937 100%);
        padding: 18px;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
        margin-bottom: 18px;
    }

    .section-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        color: #F9FAFB;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Plotly Layout Template
# -----------------------------
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.55)",
    font=dict(color="#F9FAFB"),
    margin=dict(l=40, r=20, t=50, b=40)
)

# -----------------------------
# Helper functions
# -----------------------------
@st.cache_data
def load_data():
    return pd.read_csv("insurance.csv")

@st.cache_resource
def train_models(df):
    df_encoded = pd.get_dummies(df, drop_first=True)

    X = df_encoded.drop("charges", axis=1)
    y = df_encoded["charges"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    lr = LinearRegression()
    rf = RandomForestRegressor(random_state=42)

    lr.fit(X_train, y_train)
    rf.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    rf_pred = rf.predict(X_test)

    metrics = {
        "Linear Regression": {
            "MAE": mean_absolute_error(y_test, lr_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, lr_pred)),
            "R2": r2_score(y_test, lr_pred),
        },
        "Random Forest": {
            "MAE": mean_absolute_error(y_test, rf_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, rf_pred)),
            "R2": r2_score(y_test, rf_pred),
        }
    }

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values(by="importance", ascending=False)

    evaluation_data = {
        "y_test": y_test,
        "lr_pred": lr_pred,
        "rf_pred": rf_pred
    }

    return rf, metrics, feature_importance, X.columns, evaluation_data

def prepare_input(age, sex, bmi, children, smoker, region, model_columns):
    input_dict = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }

    input_df = pd.DataFrame([input_dict])

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    return input_df[model_columns]

# -----------------------------
# Load data and models
# -----------------------------
df = load_data()
model, metrics, feature_importance, model_columns, evaluation_data = train_models(df)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Dashboard", "Prediction", "Model Performance"]
)

st.sidebar.markdown("---")
st.sidebar.info("Medical Insurance Cost Analysis & Prediction")

# -----------------------------
# Home Page
# -----------------------------
if page == "Home":
    st.title("💊 Medical Insurance Cost Analysis and Prediction")
    st.markdown("""
    This interactive dashboard explores the key factors affecting medical insurance charges
    and provides a machine learning model to predict insurance cost based on user inputs.
    """)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Average Charge", f"${df['charges'].mean():,.0f}")
    col4.metric("Max Charge", f"${df['charges'].max():,.0f}")

    st.markdown("### Key Insights")
    st.markdown("""
    - Smoking has the strongest impact on medical insurance charges.
    - Higher BMI is generally associated with higher costs.
    - Insurance charges tend to increase with age.
    - Random Forest performs better than Linear Regression on this dataset.
    """)

    st.markdown("### Top 10 Highest Insurance Charges")
    st.dataframe(
        df.sort_values("charges", ascending=False).head(10),
        use_container_width=True
    )

# -----------------------------
# Dashboard Page
# -----------------------------
elif page == "Dashboard":
    st.title("📊 Interactive Dashboard")
    st.caption("Explore the dataset using dynamic filters and professional visualizations.")

    st.sidebar.markdown("---")
    st.sidebar.header("Dashboard Filters")

    sex_filter = st.sidebar.selectbox("Gender", ["All", "male", "female"])
    smoker_filter = st.sidebar.selectbox("Smoking Status", ["All", "yes", "no"])

    filtered_df = df.copy()

    if sex_filter != "All":
        filtered_df = filtered_df[filtered_df["sex"] == sex_filter]

    if smoker_filter != "All":
        filtered_df = filtered_df[filtered_df["smoker"] == smoker_filter]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        # KPI cards
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Filtered Rows", len(filtered_df))
        k2.metric("Average Charge", f"${filtered_df['charges'].mean():,.0f}")
        k3.metric("Median Charge", f"${filtered_df['charges'].median():,.0f}")
        k4.metric("Max Charge", f"${filtered_df['charges'].max():,.0f}")

        # Row 1
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Distribution of Charges</div>', unsafe_allow_html=True)
            fig = px.histogram(
                filtered_df,
                x="charges",
                nbins=30,
                color_discrete_sequence=["#06B6D4"]
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Charges by Smoking Status (Box Plot)</div>', unsafe_allow_html=True)
            fig = px.box(
                filtered_df,
                x="smoker",
                y="charges",
                color="smoker",
                color_discrete_map={"yes": "#EF4444", "no": "#10B981"},
                points="outliers"
            )
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig.update_xaxes(title="Smoking Status")
            fig.update_yaxes(title="Insurance Charges")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 2
        col3, col4 = st.columns(2)

        with col3:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Age vs Charges</div>', unsafe_allow_html=True)
            fig = px.scatter(
                filtered_df,
                x="age",
                y="charges",
                color="smoker",
                color_discrete_map={"yes": "#F59E0B", "no": "#3B82F6"},
                opacity=0.7
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">BMI vs Charges</div>', unsafe_allow_html=True)
            fig = px.scatter(
                filtered_df,
                x="bmi",
                y="charges",
                color="sex",
                color_discrete_map={"male": "#8B5CF6", "female": "#EC4899"},
                opacity=0.7
            )
            fig.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Row 3
        col5, col6 = st.columns(2)

        with col5:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Average Charges by Region</div>', unsafe_allow_html=True)
            region_avg = (
                filtered_df.groupby("region", as_index=False)["charges"]
                .mean()
                .sort_values("charges", ascending=False)
            )
            fig = px.bar(
                region_avg,
                x="region",
                y="charges",
                color="region",
                color_discrete_sequence=["#14B8A6", "#6366F1", "#F97316", "#EAB308"]
            )
            fig.update_layout(**PLOTLY_LAYOUT, showlegend=False)
            fig.update_xaxes(title="Region")
            fig.update_yaxes(title="Average Charges")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col6:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            st.markdown('<div class="section-title">Top 10 Highest Insurance Charges</div>', unsafe_allow_html=True)
            top_10 = filtered_df.sort_values("charges", ascending=False).head(10).copy()
            top_10["label"] = top_10.index.astype(str)

            fig = px.bar(
                top_10,
                x="charges",
                y="label",
                orientation="h",
                color="charges",
                color_continuous_scale="Turbo",
                hover_data=["age", "sex", "bmi", "children", "smoker", "region"]
            )
            fig.update_layout(**PLOTLY_LAYOUT, yaxis_title="Record Index")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Correlation Heatmap
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-title">Correlation Matrix</div>', unsafe_allow_html=True)

        df_corr = filtered_df.copy()
        df_corr["sex"] = df_corr["sex"].map({"male": 1, "female": 0})
        df_corr["smoker"] = df_corr["smoker"].map({"yes": 1, "no": 0})
        df_corr["region"] = df_corr["region"].astype("category").cat.codes

        corr = df_corr.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            aspect="auto"
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Prediction Page
# -----------------------------
elif page == "Prediction":
    st.title("🤖 Insurance Cost Prediction")
    st.markdown("Enter the information below to predict medical insurance charges.")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 64, 30)
        sex = st.selectbox("Sex", ["male", "female"])

    with col2:
        bmi = st.slider("BMI", 15.0, 50.0, 27.0)
        children = st.slider("Children", 0, 5, 0)

    with col3:
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    if st.button("Predict Insurance Cost"):
        input_df = prepare_input(age, sex, bmi, children, smoker, region, model_columns)
        prediction = model.predict(input_df)[0]

        st.success(f"Predicted Insurance Cost: ${prediction:,.2f}")

        st.markdown("### Input Summary")
        summary_df = pd.DataFrame({
            "Feature": ["Age", "Sex", "BMI", "Children", "Smoker", "Region"],
            "Value": [age, sex, bmi, children, smoker, region]
        })
        st.dataframe(summary_df, use_container_width=True)

# -----------------------------
# Model Performance Page
# -----------------------------
elif page == "Model Performance":
    st.title("📈 Model Performance")

    results_df = pd.DataFrame(metrics).T.reset_index()
    results_df.columns = ["Model", "MAE", "RMSE", "R2"]
    st.dataframe(results_df, use_container_width=True)

    y_test = evaluation_data["y_test"]
    lr_pred = evaluation_data["lr_pred"]
    rf_pred = evaluation_data["rf_pred"]

    st.subheader("Actual vs Predicted — Linear Regression")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, lr_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    st.pyplot(fig)

    st.subheader("Actual vs Predicted — Random Forest")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(y_test, rf_pred, alpha=0.6)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    ax.set_xlabel("Actual Charges")
    ax.set_ylabel("Predicted Charges")
    st.pyplot(fig)

    st.subheader("Residual Plot — Linear Regression")
    lr_residuals = y_test - lr_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(lr_pred, lr_residuals, alpha=0.6)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Predicted Charges")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

    st.subheader("Residual Plot — Random Forest")
    rf_residuals = y_test - rf_pred
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(rf_pred, rf_residuals, alpha=0.6)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Predicted Charges")
    ax.set_ylabel("Residuals")
    st.pyplot(fig)

    st.subheader("Feature Importance (Random Forest)")
    fig, ax = plt.subplots(figsize=(10, 5))
    top_features = feature_importance.head(10)
    ax.barh(top_features["feature"], top_features["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)

    best_model = results_df.sort_values("R2", ascending=False).iloc[0]["Model"]
    st.info(f"Best model based on R² score: {best_model}")