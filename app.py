import streamlit as st
import pandas as pd

st.title("Water Station Data Viewer")

# Load the latest daily snapshot CSV
csv_path = "data/AB_WS_R_Flows_2025-07-15.csv"  # Replace with your actual latest filename

try:
    df = pd.read_csv(csv_path)
    st.write(f"Showing data from {csv_path}:")
    st.dataframe(df)
except FileNotFoundError:
    st.write("Daily snapshot CSV file not found.")
