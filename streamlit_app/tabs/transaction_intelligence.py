import streamlit as st

from streamlit_app import db


def render():
    st.header("Transaction Risk Intelligence")
    st.caption("Search and filter transactions; inspect risk score components and rule evidence for any transaction.")

    col1, col2, col3, col4 = st.columns(4)
    risk_tier = col1.selectbox("Risk tier", ["Any", "LOW", "MEDIUM", "HIGH", "CRITICAL"])
    channel = col2.selectbox("Channel", ["Any", "CARD_PRESENT", "ECOM", "POS", "ATM", "WALLET"])
    min_amount = col3.number_input("Min amount ($)", min_value=0.0, value=0.0, step=10.0)
    customer_id = col4.number_input("Customer ID", min_value=0, value=0, step=1, help="0 = any customer")

    limit = st.slider("Rows to show", 10, 200, 50, step=10)

    df = db.transactions_page(
        limit=limit,
        risk_tier=None if risk_tier == "Any" else risk_tier,
        channel=None if channel == "Any" else channel,
        min_amount=None if min_amount == 0 else min_amount,
        customer_id=None if customer_id == 0 else int(customer_id),
    )

    if df.empty:
        st.warning("No transactions match these filters.")
        return

    st.dataframe(
        df.rename(columns={
            "transaction_id": "Txn ID", "transaction_ts": "Time", "customer_id": "Customer",
            "merchant_id": "Merchant", "amount": "Amount", "channel": "Channel", "status": "Status",
            "risk_tier": "Risk Tier", "combined_score": "Score",
        }),
        column_config={
            "Amount": st.column_config.NumberColumn(format="$%.2f"),
            "Score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
        },
        width="stretch", hide_index=True, height=400,
    )

    st.divider()
    st.subheader("Inspect a transaction")
    txn_id = st.number_input("Transaction ID", min_value=1, step=1, value=int(df.iloc[0]["transaction_id"]))

    matching = df[df["transaction_id"] == txn_id]
    if matching.empty:
        st.caption("Enter a transaction ID from the table above (or any known ID) to see its score breakdown.")
        return

    row = matching.iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined score", f"{row['combined_score']:.3f}" if row["combined_score"] is not None else "n/a")
    c2.metric("ML component", f"{row['ml_component']:.3f}" if row["ml_component"] is not None else "n/a")
    c3.metric("Rules component", f"{row['rules_component']:.3f}" if row["rules_component"] is not None else "n/a")
    c4.metric("Behavioral component", f"{row['behavioral_component']:.3f}" if row["behavioral_component"] is not None else "n/a")

    rules_df = db.transaction_rules(int(txn_id))
    if rules_df.empty:
        st.caption("No rules fired on this transaction.")
    else:
        st.markdown("**Rules triggered:**")
        for _, r in rules_df.iterrows():
            with st.expander(f"{r['rule_id']} — {r['severity']} — {r['rule_description']}"):
                st.json(r["evidence"])
