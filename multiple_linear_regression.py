import streamlit as st
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# ---------------- Page Configuration ----------------
st.set_page_config(
    page_title="Multiple Linear Regression",
    layout="centered"
)

# ---------------- Load CSS ----------------
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("styles.css")

# ---------------- Title ----------------
st.markdown("""
<div class="card">
    <h1>Multiple Linear Regression</h1>
    <p>
        Predict <b>Tip Amount</b> using multiple independent variables from the
        <b>Seaborn Tips Dataset</b>
    </p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ---------------- Dataset Preview ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# ---------------- Feature Selection ----------------
features = ["total_bill", "size", "day", "time", "smoker"]
target = "tip"

X = df[features]
y = df[target]

num_features = ["total_bill", "size"]
cat_features = ["day", "time", "smoker"]

# ---------------- Preprocessing ----------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(drop="first"), cat_features)
    ]
)

# ---------------- Model Pipeline ----------------
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

# ---------------- Train-Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- Train Model ----------------
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ---------------- Metrics ----------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# ---------------- Model Performance ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance")

c1, c2, c3, c4 = st.columns(4)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("MSE", f"{mse:.2f}")
c3.metric("RMSE", f"{rmse:.2f}")
c4.metric("R² Score", f"{r2:.2f}")

st.markdown('</div>', unsafe_allow_html=True)
# ---------------- Visualization ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Multiple Regression Lines (All Features Combined)")

fig, ax = plt.subplots(figsize=(8, 5))

# Scatter (make lighter)
ax.scatter(
    df["total_bill"],
    df["tip"],
    alpha=0.4,
    s=40
)

# Bill range
bill_range = np.linspace(df["total_bill"].min(), df["total_bill"].max(), 100)

# Different regression scenarios
scenarios = [
    {"size": 2, "day": "Thur", "time": "Lunch", "smoker": "No"},
    {"size": 4, "day": "Fri", "time": "Dinner", "smoker": "No"},
    {"size": 2, "day": "Sat", "time": "Dinner", "smoker": "Yes"},
    {"size": 3, "day": "Sun", "time": "Dinner", "smoker": "No"},
]

# Plot multiple regression lines (VISIBLE)
for scenario in scenarios:
    scenario_df = pd.DataFrame({
        "total_bill": bill_range,
        "size": scenario["size"],
        "day": scenario["day"],
        "time": scenario["time"],
        "smoker": scenario["smoker"]
    })

    tip_pred = model.predict(scenario_df)

    ax.plot(
        bill_range,
        tip_pred,
        color="red",
        linewidth=2.5,      # 🔥 thicker
        alpha=0.9           # 🔥 less transparent
    )

ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Predicted Tip ($)")
ax.set_title("Multiple Linear Regression Lines (All Features Combined)")

st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)


# ---------------- Prediction Section ----------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Predict Tip Amount")

total_bill = st.slider("Total Bill ($)", 5.0, 60.0, 25.0)
size = st.slider("Party Size", 1, 6, 2)
day = st.selectbox("Day", df["day"].unique())
time = st.selectbox("Time", df["time"].unique())
smoker = st.selectbox("Smoker", df["smoker"].unique())

input_df = pd.DataFrame([{
    "total_bill": total_bill,
    "size": size,
    "day": day,
    "time": time,
    "smoker": smoker
}])

prediction = model.predict(input_df)[0]

st.markdown(
    f'<div class="prediction-box">Predicted Tip: ${prediction:.2f}</div>',
    unsafe_allow_html=True
)

st.markdown('</div>', unsafe_allow_html=True)
