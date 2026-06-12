import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Invoice Payment Prediction",
    page_icon="📊",
    layout="wide"
)

# ── Load predictions ──────────────────────────────────────────
@st.cache_data
def load_data():
    predictions = pd.read_csv('data/open_invoice_predictions.csv')
    high_risk = pd.read_csv('data/high_risk_with_days.csv')
    bucket3 = pd.read_csv('data/bucket3_predictions.csv')
    bucket1 = pd.read_csv('data/bucket1_features.csv')
    return predictions, high_risk, bucket3, bucket1

predictions, high_risk, bucket3, bucket1 = load_data()

# ── Sidebar navigation ────────────────────────────────────────
st.sidebar.title("📊 Invoice Prediction")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Open Invoices", "High Risk Detail", "Partial Payments"]
)

# ══════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
if page == "Overview":
    st.title("Invoice Payment Prediction Dashboard")
    st.markdown("ML-powered risk scoring for accounts receivable")
    st.markdown("---")

    # ── KPI cards ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Open Invoices", len(predictions))

    with col2:
        high_count = len(predictions[predictions['risk_level'] == 'High'])
        st.metric("High Risk", high_count, delta="Needs action")

    with col3:
        medium_count = len(predictions[predictions['risk_level'] == 'Medium'])
        st.metric("Medium Risk", medium_count)

    with col4:
        low_count = len(predictions[predictions['risk_level'] == 'Low'])
        st.metric("Low Risk", low_count)

    st.markdown("---")

    # ── Two columns layout ────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Level Distribution")
        risk_counts = predictions['risk_level'].value_counts()
        colors = {'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#27AE60'}
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(
            risk_counts.values,
            labels=risk_counts.index,
            colors=[colors.get(x, '#95A5A6') for x in risk_counts.index],
            autopct='%1.1f%%',
            startangle=90
        )
        ax.set_title("Open Invoice Risk Breakdown")
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.subheader("Training Data Summary")
        late_rate = bucket1['is_late'].mean() * 100
        on_time = len(bucket1[bucket1['is_late'] == 0])
        late = len(bucket1[bucket1['is_late'] == 1])

        st.info(f"Total historical invoices: **{len(bucket1)}**")
        st.success(f"Paid on time: **{on_time}**")
        st.error(f"Paid late: **{late}**")
        st.warning(f"Late rate: **{late_rate:.1f}%**")

        st.markdown("---")
        st.subheader("Model Performance")
        st.metric("Classifier ROC-AUC", "0.977")
        st.metric("Regressor MAE", "4.5 days")
        st.metric("Overall Accuracy", "91%")

# ══════════════════════════════════════════════════════════════
# PAGE 2 — OPEN INVOICES
# ══════════════════════════════════════════════════════════════
elif page == "Open Invoices":
    st.title("Open Invoice Risk Scores")
    st.markdown("All open invoices scored by late payment probability")
    st.markdown("---")

    # ── Filters ───────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        risk_filter = st.multiselect(
            "Filter by risk level",
            options=['High', 'Medium', 'Low'],
            default=['High', 'Medium', 'Low']
        )
    with col2:
        sort_by = st.selectbox(
            "Sort by",
            options=['late_probability', 'invoice_amount', 'days_until_due']
        )

    filtered = predictions[predictions['risk_level'].isin(risk_filter)]
    filtered = filtered.sort_values(sort_by, ascending=False)

    # ── Colour code risk level ────────────────────────────────
    def colour_risk(val):
        if val == 'High':
            return 'color: #E74C3C; font-weight: bold'
        elif val == 'Medium':
            return 'color: #F39C12; font-weight: bold'
        else:
            return 'color: #27AE60; font-weight: bold'

    st.dataframe(
        filtered[['invoice_id', 'customer_id', 'invoice_amount',
                  'due_date', 'late_probability', 'risk_level',
                  'days_until_due']].style.map(
            colour_risk, subset=['risk_level']
        ),
        use_container_width=True,
        height=500
    )

    st.markdown(f"Showing **{len(filtered)}** invoices")

# ══════════════════════════════════════════════════════════════
# PAGE 3 — HIGH RISK DETAIL
# ══════════════════════════════════════════════════════════════
elif page == "High Risk Detail":
    st.title("High Risk Invoices — Action Required")
    st.markdown("Invoices flagged as high risk with estimated payment dates")
    st.markdown("---")

    st.error(f"⚠️ {len(high_risk)} invoices require immediate attention")

    for _, row in high_risk.iterrows():
        with st.expander(
            f"📄 {row['invoice_id']} — Customer {int(row['customer_id'])} "
            f"— ₹{int(row['invoice_amount']):,} — {row['late_probability']}% late risk"
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Late Probability", f"{row['late_probability']}%")
            with col2:
                st.metric("Est. Days Late", f"{int(row['estimated_days_late'])} days")
            with col3:
                st.metric("Est. Payment Date", str(row['estimated_payment_date'])[:10])

            if row['late_probability'] >= 80:
                st.error("🔴 Action: Call customer today")
            else:
                st.warning("🟡 Action: Send payment reminder")

# ══════════════════════════════════════════════════════════════
# PAGE 4 — PARTIAL PAYMENTS
# ══════════════════════════════════════════════════════════════
elif page == "Partial Payments":
    st.title("Partial Payment Analysis")
    st.markdown("Predicting which partial invoices will settle vs become bad debt")
    st.markdown("---")

    col1, col2 = st.columns(2)
    will_settle = bucket3[bucket3['prediction'] == 'Will Settle']
    bad_debt = bucket3[bucket3['prediction'] == 'Bad Debt']

    with col1:
        st.success(f"✅ Will Settle: **{len(will_settle)}** invoices")
    with col2:
        st.error(f"❌ Bad Debt Risk: **{len(bad_debt)}** invoices")

    st.markdown("---")
    st.subheader("All Partial Payment Invoices")

    st.dataframe(
        bucket3[['invoice_id', 'customer_id', 'invoice_amount',
                 'pct_paid_so_far', 'remaining_balance',
                 'settle_probability', 'prediction', 'status']].round(2),
        use_container_width=True,
        height=400
    )