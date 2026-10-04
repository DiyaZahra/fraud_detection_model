import streamlit as st
import pandas as pd
from xgboost import XGBClassifier

# Same column order as used in training
COLUMNS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
           'newbalanceDest', 'hour', 'errorBalanceOrig', 'errorBalanceDest',
           'drainedAll', 'destMerchant', 'type_CASH_IN', 'type_CASH_OUT',
           'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


@st.cache_resource
def load_model():
    m = XGBClassifier()
    m.load_model('fraud_model.json')
    return m


model = load_model()

st.title('Mobile Money Fraud Detection')
st.write('Enter the transaction details and the model will predict whether it looks fraudulent.')

txn_type = st.selectbox('Transaction type',
                        ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount = st.number_input('Amount', min_value=0.0, value=1000.0, step=100.0)
oldbalanceOrg = st.number_input('Sender balance before', min_value=0.0, value=5000.0, step=100.0)
newbalanceOrig = st.number_input('Sender balance after', min_value=0.0, value=4000.0, step=100.0)
oldbalanceDest = st.number_input('Receiver balance before', min_value=0.0, value=0.0, step=100.0)
newbalanceDest = st.number_input('Receiver balance after', min_value=0.0, value=0.0, step=100.0)
hour = st.slider('Hour of day', 0, 23, 12)
receiver = st.radio('Receiver type', ['Customer (C)', 'Merchant (M)'])
destMerchant = 1 if receiver == 'Merchant (M)' else 0

if st.button('Predict'):
    row = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'hour': hour,
        'errorBalanceOrig': newbalanceOrig + amount - oldbalanceOrg,
        'errorBalanceDest': oldbalanceDest + amount - newbalanceDest,
        'drainedAll': int(amount == oldbalanceOrg),
        'destMerchant': destMerchant,
        'type_CASH_IN': int(txn_type == 'CASH_IN'),
        'type_CASH_OUT': int(txn_type == 'CASH_OUT'),
        'type_DEBIT': int(txn_type == 'DEBIT'),
        'type_PAYMENT': int(txn_type == 'PAYMENT'),
        'type_TRANSFER': int(txn_type == 'TRANSFER'),
    }
    data = pd.DataFrame([row])[COLUMNS]
    prob = model.predict_proba(data)[0, 1]

    if prob >= 0.5:
        st.error(f'Likely FRAUD. Fraud probability: {prob:.2%}')
    else:
        st.success(f'Looks normal. Fraud probability: {prob:.2%}')
