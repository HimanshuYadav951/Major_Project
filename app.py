
import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Set Streamlit page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load models and data
kmeans_model = pickle.load(open("kmeans_model.pkl", "rb"))
final_model = joblib.load(open("final_model.sav", "rb"))
df_clustered = pd.read_csv("Clustered_Customer_Data.csv")

# Sidebar
st.sidebar.title("Customer Segmentation App")
st.sidebar.image("https://miro.medium.com/v2/resize:fit:1400/1*evL_9eEbPUKn3_vIugSJUg.png", use_column_width=True)

# Tabs
tab1, tab2 = st.tabs(["📊 Predict Segment", "📈 Cluster Insights"])

with tab1:
    st.title("🔍 Predict Customer Segment")
    st.markdown("Enter customer data to predict their cluster.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            balance = st.number_input("Balance", 0.0, 100000.0, 500.0)
            purchases = st.number_input("Purchases", 0.0, 100000.0, 100.0)
            cash_advance = st.number_input("Cash Advance", 0.0, 100000.0, 0.0)
            credit_limit = st.number_input("Credit Limit", 0.0, 100000.0, 1500.0)
            payments = st.number_input("Payments", 0.0, 100000.0, 200.0)
            min_payments = st.number_input("Minimum Payments", 0.0, 100000.0, 150.0)

        with col2:
            balance_freq = st.slider("Balance Frequency", 0.0, 1.0, 0.9)
            purchase_freq = st.slider("Purchase Frequency", 0.0, 1.0, 0.3)
            one_off_freq = st.slider("One-off Purchase Frequency", 0.0, 1.0, 0.1)
            installment_freq = st.slider("Installment Purchase Frequency", 0.0, 1.0, 0.1)
            cash_freq = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.0)
            prc_full = st.slider("PRC Full Payment", 0.0, 1.0, 0.0)

        with col3:
            one_off = st.number_input("One-off Purchases", 0.0, 100000.0, 0.0)
            installment = st.number_input("Installment Purchases", 0.0, 100000.0, 100.0)
            cash_trx = st.number_input("Cash Advance TRX", 0, 100, 0)
            purchase_trx = st.number_input("Purchase TRX", 0, 100, 2)
            tenure = st.slider("Tenure", 0, 12, 12)

        submitted = st.form_submit_button("Predict Cluster")

        if submitted:
            input_data = np.array([[
                balance, balance_freq, purchases, one_off, installment, cash_advance,
                purchase_freq, one_off_freq, installment_freq, cash_freq, cash_trx,
                purchase_trx, credit_limit, payments, min_payments, prc_full, tenure
            ]])
            cluster = final_model.predict(input_data)[0]
            st.success(f"The customer belongs to **Cluster {cluster}**.")

with tab2:
    st.title("📈 Cluster Analysis & t-SNE Visualization")
    st.markdown("Explore the existing clusters using 2D t-SNE reduction.")

    numeric_cols = df_clustered.select_dtypes(include=[np.number]).drop(columns=["Cluster"])
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
    tsne_results = tsne.fit_transform(numeric_cols)

    df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    df_tsne['Cluster'] = df_clustered['Cluster']

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue='Cluster', palette='Set1', s=60, alpha=0.7)
    plt.title("t-SNE Visualization of Clusters")
    st.pyplot(fig)
