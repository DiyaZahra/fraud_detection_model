import streamlit as st
import pandas as pd
import joblib

# Same column order as used in training
COLUMNS = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest',
           'newbalanceDest', 'hour', 'errorBalanceOrig', 'errorBalanceDest',
           'drainedAll', 'destMerchant', 'type_CASH_IN', 'type_CASH_OUT',
           'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']


@st.cache_resource
def load_model():
    return joblib.load('rf_model.joblib')


model = load_model()

st.title('Mobile Money Fraud Detection')
st.write('Enter the transaction details and the model will predict whether it looks fraudulent.')

txn_type = st.selectbox('Transaction type',
                        ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
amount_text = st.text_input('Amount', placeholder='e.g. 181000')
old_org_text = st.text_input('Sender balance before', placeholder='e.g. 181000')
new_org_text = st.text_input('Sender balance after', placeholder='e.g. 0')
old_dest_text = st.text_input('Receiver balance before', placeholder='e.g. 0')
new_dest_text = st.text_input('Receiver balance after', placeholder='e.g. 0')
hour = st.slider('Hour of day', 0, 23, 12)
receiver = st.radio('Receiver type', ['Customer (C)', 'Merchant (M)'])
destMerchant = 1 if receiver == 'Merchant (M)' else 0

if st.button('Predict'):
    try:
        amount = float(amount_text.replace(',', ''))
        oldbalanceOrg = float(old_org_text.replace(',', ''))
        newbalanceOrig = float(new_org_text.replace(',', ''))
        oldbalanceDest = float(old_dest_text.replace(',', ''))
        newbalanceDest = float(new_dest_text.replace(',', ''))
    except ValueError:
        st.error('Please fill in all five number fields with valid numbers.')
        st.stop()

    if min(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest) < 0:
        st.error('Numbers cannot be negative.')
        st.stop()

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

st.caption('Demo only: the model was trained on PaySim synthetic data and is not meant for real transactions.')
