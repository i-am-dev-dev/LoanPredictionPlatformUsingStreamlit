import streamlit as st
import pandas as pd
import hashlib
import json
from time import time
import random
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# --- Block and Blockchain Code
# ------------------------------
class Block:
    def __init__(self, index, transactions, timestamp, previous_hash, creator, creator_stake):
        self.index = index
        self.transactions = transactions
        self.timestamp = timestamp
        self.previous_hash = previous_hash
        self.creator = creator
        self.creator_stake = creator_stake
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self, stakeholder_stakes):
        self.chain = []
        self.stakeholder_stakes = stakeholder_stakes
        self.create_genesis_block()

    def create_genesis_block(self):
        genesis_block = Block(0, [], str(time()), "0", "Genesis", 0)
        self.chain.append(genesis_block)

    def add_block(self, transactions):
        last_block = self.chain[-1]
        creator = random.choice(list(self.stakeholder_stakes.keys()))
        creator_stake = self.stakeholder_stakes[creator]
        new_block = Block(
            index=last_block.index + 1,
            transactions=transactions,
            timestamp=str(time()),
            previous_hash=last_block.hash,
            creator=creator,
            creator_stake=creator_stake
        )
        self.chain.append(new_block)

    def is_valid(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            if current_block.hash != current_block.compute_hash():
                return False
            if current_block.previous_hash != previous_block.hash:
                return False
            if current_block.creator_stake < 100:
                return False
        return True


# ------------------------------
# --- Loan Prediction Helpers
# ------------------------------
DATA_PATH = "C:\\GENERAL\\lp\\loan_prediction_dataset_shuffled.csv"

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        if 'Interest Rate' in df.columns:
            df['Interest Rate'] = df['Interest Rate'].astype(str).str.replace('%', '').astype(float) / 100
        df['Liquidity to Funds Ratio'] = df['Liquidity'] / (df['Company Current Funds'] + 1)
        df['Credit-Profit Interaction'] = df['Credit History'] * df['Profit Percentage']
        df['Liquidity-Funds Interaction'] = df['Liquidity'] * df['Company Current Funds']
        df['Credit*Frequency'] = df['Credit History'] * df['Frequency of taking loans']
        df['Profit^2'] = df['Profit Percentage'] ** 2
        return df
    return pd.DataFrame()

def calculate_loan_amount(liquidity, funds, profit):
    if profit < 0:
        base = max(100000, (1000000 - liquidity) * 1.5)
    else:
        base = max(100000, (1000000 - liquidity) * (1 + profit / 100))
    return min(base, 10000000)

# ------------------------------
# --- Streamlit Interface
# ------------------------------
st.title("📊 Multi-Function App with Buttons")

# Buttons to switch between apps
option = st.sidebar.radio("Choose a Feature", ["Delivery Blockchain", "Loan Repayment Ratio", "Loan Prediction"])

# ------------------------------
# --- 1. Loan Repayment Ratio
# ------------------------------
if option == "Loan Repayment Ratio":
    st.header("📈 Loan Repayment Ratio Calculator")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx","csv"], key="repayment_file")
    loan_id = st.text_input("Enter LOAN ID")

    def calculate_repayment_ratio(df, loan_id):
        if 'LOAN ID' not in df.columns or 'LOAN AMOUNT' not in df.columns or 'LOAN AMOUNT REFUNDED' not in df.columns:
            return "Dataset is missing required columns."

        row = df[df['LOAN ID'] == loan_id]
        if row.empty:
            return "LOAN ID not found."

        try:
            loan_amount = float(row['LOAN AMOUNT'].values[0])
            refunded = float(row['LOAN AMOUNT REFUNDED'].values[0])
            if loan_amount == 0:
                return "Loan amount is zero. Cannot compute ratio."
            repayment_ratio = refunded / loan_amount
            return f"Repayment Ratio for {loan_id}: **{repayment_ratio:.2f}**"
        except Exception as e:
            return f"Error in processing data: {str(e)}"

    if uploaded_file and loan_id:
        try:
            df = pd.read_excel(uploaded_file)
            result = calculate_repayment_ratio(df, loan_id)
            st.markdown(result)
        except Exception as e:
            st.error(f"Failed to read file: {str(e)}")

# ------------------------------
# --- 2. Loan Prediction
# ------------------------------
elif option == "Loan Prediction":
    st.header("🏦 Loan Prediction System")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv","xlsx"], key="loan_pred_file")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)

            # Basic check
            if df.empty or 'Company Current Funds' not in df.columns:
                st.warning("CSV does not have expected structure. Make sure 'Company Current Funds' column exists.")
            else:
                st.write("Sample data from dataset:")
                st.dataframe(df.head())

                st.subheader("Predict Loan Requirement and Amount")
                liquidity = st.number_input("Liquidity", min_value=0)
                credit = st.number_input("Credit Score (300 - 850)", min_value=300, max_value=850)
                frequency = st.slider("Frequency of taking loans", min_value=0, max_value=10)
                profit = st.number_input("Profit Percentage")

                if st.button("Predict Loan Requirement"):
                    needs_loan = credit < 700 or liquidity < 1000000
                    if needs_loan:
                        st.success("✅ This company **requires** a loan.")
                        amount = calculate_loan_amount(liquidity, df['Company Current Funds'].mean(), profit)
                        st.markdown(f"💰 **Estimated Loan Amount Required**: ₹{amount:,.2f}")
                    else:
                        st.info("❌ This company likely **does not need** a loan.")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    else:
        st.info("Please upload a CSV file to begin.")


# ------------------------------
# --- 3. Delivery Blockchain
# ------------------------------
elif option == "Delivery Blockchain":
    st.header("🚛 Delivery Blockchain with Stakeholders")

    uploaded_file = st.file_uploader("Upload Excel File", type=["csv","xlsx"], key="blockchain_file")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)

            required_cols = ['BookingID', 'Origin_Location', 'Destination_Location']
            if not all(col in df.columns for col in required_cols):
                st.error("File must contain BookingID, Origin_Location, Destination_Location")
            else:
                stakeholders = ["Stakeholder1", "Stakeholder2", "Stakeholder3"]
                stakeholder_stakes = {s: random.randint(50, 500) for s in stakeholders}
                blockchain = Blockchain(stakeholder_stakes)

                for index, row in df.iterrows():
                    trip_data = {
                        'BookingID': row['BookingID'],
                        'Origin_Location': row['Origin_Location'],
                        'Destination_Location': row['Destination_Location']
                    }
                    blockchain.add_block(trip_data)

                if blockchain.is_valid():
                    st.success("✅ Blockchain is valid!")

                st.subheader("📦 Blockchain Details")
                for block in blockchain.chain:
                    with st.expander(f"Block {block.index}"):
                        st.json({
                            "Index": block.index,
                            "Transactions": block.transactions,
                            "Timestamp": block.timestamp,
                            "Previous Hash": block.previous_hash,
                            "Creator": block.creator,
                            "Creator Stake": block.creator_stake,
                            "Hash": block.hash
                        })

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")