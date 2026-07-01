"""
================================================================================
 Credit Score Classification — Streamlit Web App (Deployment)
================================================================================
Web app untuk melakukan inferencing model credit score secara interaktif.
Mendukung dua mode input:
    1. Form manual  -> isi data satu nasabah langsung di browser.
    2. Upload CSV    -> prediksi batch untuk banyak nasabah sekaligus
                        (format kolom sama seperti data_D.csv).

Jalankan dengan:
    streamlit run app.py
================================================================================
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from c_Inference import CreditScoreInferenceService

MODEL_PATH = Path("models/credit_score_model.pkl")

CLASS_COLOR = {"Poor": "#e74c3c", "Standard": "#f1c40f", "Good": "#2ecc71"}


# ==============================================================================
# Caching model load (hanya di-load sekali per sesi server, bukan per klik)
# ==============================================================================
@st.cache_resource(show_spinner="Memuat model...")
def load_service() -> CreditScoreInferenceService:
    return CreditScoreInferenceService.from_pickle(MODEL_PATH)


def render_prediction(result) -> None:
    color = CLASS_COLOR.get(result.predicted_class, "#3498db")
    st.markdown(
        f"""
        <div style="padding:16px;border-radius:10px;background-color:{color}22;
                     border:1px solid {color};">
            <span style="font-size:14px;color:#888;">Prediksi Credit Score</span><br>
            <span style="font-size:28px;font-weight:700;color:{color};">{result.predicted_class}</span>
            <span style="font-size:14px;color:#888;"> &nbsp;(keyakinan {result.confidence:.1%})</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("Probabilitas tiap kelas:")
    proba_df = pd.DataFrame(
        {"Kelas": list(result.class_probabilities.keys()),
         "Probabilitas": list(result.class_probabilities.values())}
    ).sort_values("Probabilitas", ascending=False)
    st.bar_chart(proba_df.set_index("Kelas"))


def render_manual_form(service: CreditScoreInferenceService) -> None:
    st.subheader("Input Data Nasabah")
    with st.form("manual_input_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Age", min_value=14, max_value=100, value=35)
            occupation = st.text_input("Occupation", value="Engineer")
            annual_income = st.number_input("Annual Income", min_value=0.0, value=50000.0, step=1000.0)
            monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, value=4000.0, step=100.0)
            num_bank_accounts = st.number_input("Num Bank Accounts", min_value=0, max_value=20, value=4)
            num_credit_card = st.number_input("Num Credit Card", min_value=0, max_value=20, value=4)
            interest_rate = st.number_input("Interest Rate (%)", min_value=0, max_value=40, value=12)
            num_of_loan = st.number_input("Num of Loan", min_value=0, max_value=15, value=2)
            type_of_loan = st.text_input("Type of Loan", value="Auto Loan, and Home Equity Loan")

        with col2:
            delay_from_due_date = st.number_input("Delay from Due Date (days)", min_value=-10, max_value=100, value=10)
            num_delayed_payment = st.number_input("Num of Delayed Payment", min_value=0, max_value=60, value=5)
            changed_credit_limit = st.number_input("Changed Credit Limit", value=5.0)
            num_credit_inquiries = st.number_input("Num Credit Inquiries", min_value=0, max_value=50, value=3)
            credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Bad"])
            outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, value=1200.0, step=50.0)
            credit_util_ratio = st.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, value=30.0)
            credit_history_age = st.text_input("Credit History Age", value="15 Years and 6 Months")

        with col3:
            payment_of_min_amount = st.selectbox("Payment of Min Amount", ["Yes", "No", "NM"])
            total_emi = st.number_input("Total EMI per Month", min_value=0.0, value=150.0, step=10.0)
            amount_invested = st.number_input("Amount Invested Monthly", min_value=0.0, value=200.0, step=10.0)
            payment_behaviour = st.selectbox(
                "Payment Behaviour",
                [
                    "High_spent_Small_value_payments", "High_spent_Medium_value_payments",
                    "High_spent_Large_value_payments", "Low_spent_Small_value_payments",
                    "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
                ],
            )
            monthly_balance = st.number_input("Monthly Balance", value=350.0)

        submitted = st.form_submit_button("Prediksi", use_container_width=True)

    if submitted:
        raw_input = {
            "Age": str(age), "Occupation": occupation, "Annual_Income": str(annual_income),
            "Monthly_Inhand_Salary": monthly_inhand_salary, "Num_Bank_Accounts": num_bank_accounts,
            "Num_Credit_Card": num_credit_card, "Interest_Rate": interest_rate,
            "Num_of_Loan": str(num_of_loan), "Type_of_Loan": type_of_loan,
            "Delay_from_due_date": delay_from_due_date, "Num_of_Delayed_Payment": str(num_delayed_payment),
            "Changed_Credit_Limit": str(changed_credit_limit), "Num_Credit_Inquiries": num_credit_inquiries,
            "Credit_Mix": credit_mix, "Outstanding_Debt": str(outstanding_debt),
            "Credit_Utilization_Ratio": credit_util_ratio, "Credit_History_Age": credit_history_age,
            "Payment_of_Min_Amount": payment_of_min_amount, "Total_EMI_per_month": total_emi,
            "Amount_invested_monthly": str(amount_invested), "Payment_Behaviour": payment_behaviour,
            "Monthly_Balance": monthly_balance,
        }
        try:
            result = service.predict_single(raw_input)
            render_prediction(result)
        except Exception as exc:
            st.error(f"Gagal melakukan prediksi: {exc}")


def render_batch_upload(service: CreditScoreInferenceService) -> None:
    st.subheader("Prediksi Batch dari File CSV")
    st.caption("Format kolom harus sama seperti `data_D.csv` (boleh mengandung kolom kotor, akan dibersihkan otomatis oleh pipeline).")
    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            if df.columns[0].lower() in ("unnamed: 0", ""):
                df = df.drop(columns=[df.columns[0]])

            with st.spinner("Menjalankan prediksi..."):
                results = service.predict_batch(df)

            output_df = df.copy()
            output_df["Predicted_Credit_Score"] = [r.predicted_class for r in results]
            output_df["Confidence"] = [round(r.confidence, 4) for r in results]

            st.success(f"Berhasil memprediksi {len(output_df)} baris.")
            st.dataframe(output_df, use_container_width=True)

            st.download_button(
                "Unduh hasil (CSV)",
                data=output_df.to_csv(index=False).encode("utf-8"),
                file_name="credit_score_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(f"Gagal memproses file: {exc}")


def main() -> None:
    st.set_page_config(page_title="Credit Score Classifier", layout="wide")
    st.title("Credit Score Classification")
    st.caption("Prediksi kategori risiko kredit nasabah (Poor / Standard / Good) menggunakan model Random Forest yang dilatih via pipeline MLflow.")

    if not MODEL_PATH.exists():
        st.error(
            f"File model tidak ditemukan di `{MODEL_PATH}`. "
            "Jalankan `python run_training.py` terlebih dahulu untuk menghasilkan model."
        )
        st.stop()

    service = load_service()

    tab_manual, tab_batch = st.tabs(["Input Manual", "Upload CSV (Batch)"])
    with tab_manual:
        render_manual_form(service)
    with tab_batch:
        render_batch_upload(service)

    with st.sidebar:
        st.header("Info Model")
        st.write("**Kelas target:**", ", ".join(service.target_classes))
        st.write("**Jumlah kolom input mentah:**", len(service.required_columns))
        with st.expander("Lihat daftar kolom input"):
            st.write(service.required_columns)


if __name__ == "__main__":
    main()
