import streamlit as st
import numpy as np

from src.loader import DatasetLoader
from src.scaler import DataScaler
from src.analyzer import DataAnalyzer
from src.statistics import StatisticsEngine


st.title("NumPy Data Analytics Toolkit")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    # Load dataset
    loader = DatasetLoader(uploaded_file)
    data = loader.load_dataset()

    st.subheader("Dataset Preview")
    st.table(data)

    # ==============================
    # Statistical Summary
    # ==============================
    st.subheader("Statistical Summary")

    stats = StatisticsEngine(data)
    summary = stats.summary()

    # Convert dictionary to table format
    table_data = []
    headers = ["Statistic"] + [f"Feature {i+1}" for i in range(data.shape[1])]

    for key, values in summary.items():
        row = [key] + list(values)
        table_data.append(row)

    st.table([headers] + table_data)

    # ==============================
    # Scaling Section
    # ==============================
    st.subheader("Scaling Options")

    scaler = DataScaler(data)

    scaling_method = st.selectbox(
        "Select Scaling Method",
        ["None", "Min-Max Scaling", "Standard Scaling"]
    )

    if scaling_method == "Min-Max Scaling":
        data = scaler.min_max_scale()
        st.write("Min-Max Scaled Data")
        st.table(data)

    elif scaling_method == "Standard Scaling":
        data = scaler.standard_scale()
        st.write("Standard Scaled Data")
        st.table(data)

    # ==============================
    # Analysis Section
    # ==============================
    st.subheader("Data Analysis")

    analyzer = DataAnalyzer(data)

    if st.button("Show Correlation Matrix"):
        corr = analyzer.correlation_matrix()
        st.write("Correlation Matrix")
        st.table(corr)

    if st.button("Detect Outliers"):
        outliers = analyzer.detect_outlier()

        if outliers is None:
            st.success("No Outliers Detected")
        else:
            st.write("Outlier Indices (row, column)")
            st.write(outliers)