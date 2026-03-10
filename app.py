import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
# Helper functions
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("insurance.csv")
    return df

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

    return rf, metrics, feature_importance, X.columns

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

    input_df = input_df[model_columns]
    return input_df

# -----------------------------
# Load data and models
# -----------------------------
df = load_data()
model, metrics, feature_importance, model_columns = train_models(df)

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
    st.markdown(
        """
        This dashboard explores the key factors affecting medical insurance charges
        and provides a machine learning model to predict insurance cost based on user inputs.
        """
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Average Charge", f"${df['charges'].mean():,.0f}")
    col4.metric("Max Charge", f"${df['charges'].max():,.0f}")

    st.markdown("### Key Insights")
    st.markdown(
        """
        - Smoking has the strongest impact on medical insurance charges.
        - Higher BMI is generally associated with higher costs.
        - Insurance charges tend to increase with age.
        - Random Forest performs better than Linear Regression on this dataset.
        """
    )

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

# -----------------------------
# Dashboard Page
# -----------------------------
elif page == "Dashboard":
    st.title("📊 Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Distribution of Charges")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(df["charges"], bins=30)
        ax.set_xlabel("Charges")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    with col2:
        st.subheader("Charges by Smoking Status")
        smoker_avg = df.groupby("smoker")["charges"].mean()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(smoker_avg.index, smoker_avg.values)
        ax.set_xlabel("Smoker")
        ax.set_ylabel("Average Charges")
        st.pyplot(fig)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Age vs Charges")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["age"], df["charges"], alpha=0.6)
        ax.set_xlabel("Age")
        ax.set_ylabel("Charges")
        st.pyplot(fig)

    with col4:
        st.subheader("BMI vs Charges")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(df["bmi"], df["charges"], alpha=0.6)
        ax.set_xlabel("BMI")
        ax.set_ylabel("Charges")
        st.pyplot(fig)

    st.subheader("Average Charges by Region")
    region_avg = df.groupby("region")["charges"].mean().sort_values()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(region_avg.index, region_avg.values)
    ax.set_xlabel("Region")
    ax.set_ylabel("Average Charges")
    st.pyplot(fig)

    st.subheader("Correlation Matrix")
    df_corr = df.copy()
    df_corr["sex"] = df_corr["sex"].map({"male": 1, "female": 0})
    df_corr["smoker"] = df_corr["smoker"].map({"yes": 1, "no": 0})
    df_corr["region"] = df_corr["region"].astype("category").cat.codes
    corr = df_corr.corr(numeric_only=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im)
    st.pyplot(fig)

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