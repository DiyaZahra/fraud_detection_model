import streamlit as st
import pandas as pd
import joblib

model = joblib.load("rf_model.joblib")
features = list(model.feature_names_in_)
THRESHOLD = 0.5  # agar threshold tune kiya tha to wo value likho

st.title("💳 Fraud Detection (PaySim)")

t = st.selectbox("Type", ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"])
amount = st.number_input("Amount", min_value=0.0, value=1000.0)
old_o = st.number_input("Sender balance before", min_value=0.0)
new_o = st.number_input("Sender balance after", min_value=0.0)
old_d = st.number_input("Receiver balance before", min_value=0.0)
new_d = st.number_input("Receiver balance after", min_value=0.0)
hour = st.slider("Hour of day", 0, 23, 12)
merchant = st.checkbox("Receiver is a merchant")

if st.button("Check"):
    row = {
        "amount": amount,
        "oldbalanceOrg": old_o,
        "newbalanceOrig": new_o,
        "oldbalanceDest": old_d,
        "newbalanceDest": new_d,
        "hour": hour,
        "errorBalanceOrig": new_o + amount - old_o,
        "errorBalanceDest": old_d + amount - new_d,
        "drainedAll": int(amount == old_o),
        "destMerchant": int(merchant),
    }
    for ty in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
        row[f"type_{ty}"] = int(t == ty)

    X = pd.DataFrame([row]).reindex(columns=features, fill_value=0)
    p = model.predict_proba(X)[0, 1]

    st.metric("Fraud probability", f"{p:.1%}")
    if p >= THRESHOLD:
        st.error("🚨 Fraud")
    else:
        st.success("✅ Legit")
