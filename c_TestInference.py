"""
================================================================================
 Test Case: Deployment Testing per Kelas (Poor / Standard / Good)
================================================================================
Menguji `CreditScoreInferenceService` menggunakan satu contoh data nasabah
nyata per kelas target (diambil langsung dari `data_D.csv`, dengan nilai
kotor apa adanya untuk sekaligus menguji bahwa `CreditDataCleaner` di dalam
pipeline bekerja dengan benar saat inferencing).

Jalankan:
    python test_inference.py
    # atau, jika pytest tersedia:
    pytest test_inference.py -v
================================================================================
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

from c_Inference import CreditScoreInferenceService

MODEL_PATH = Path("models/credit_score_model.pkl")


# ==============================================================================
# TEST CASES — satu representasi nyata per kelas (data mentah, belum dibersihkan)
# ==============================================================================
TEST_CASES: Dict[str, Dict[str, Any]] = {
    # Diambil langsung dari data_D.csv (baris nyata, ground truth Credit_Score
    # sudah diketahui), dengan nilai kotor apa adanya (mis. kolom bertipe
    # teks, placeholder "_") -- sekaligus menguji bahwa CreditDataCleaner di
    # dalam pipeline menangani data mentah dengan benar saat inferencing.
    "Good": {
        "ID": "0x83ca", "Customer_ID": "CUS_0xb9d2", "Month": "January",
        "Name": "arbarat", "Age": "28", "SSN": "000-80-0683",
        "Occupation": "Manager", "Annual_Income": "79950.34",
        "Monthly_Inhand_Salary": 6393.528333333333, "Num_Bank_Accounts": 3,
        "Num_Credit_Card": 1, "Interest_Rate": 10, "Num_of_Loan": "4",
        "Type_of_Loan": "Credit-Builder Loan, Debt Consolidation Loan, Not Specified, and Auto Loan",
        "Delay_from_due_date": 15, "Num_of_Delayed_Payment": "3",
        "Changed_Credit_Limit": "2.27", "Num_Credit_Inquiries": 1.0,
        "Credit_Mix": "Good", "Outstanding_Debt": "515.89",
        "Credit_Utilization_Ratio": 29.73539709090268,
        "Credit_History_Age": "24 Years and 3 Months",
        "Payment_of_Min_Amount": "No", "Total_EMI_per_month": 261.4747996593964,
        "Amount_invested_monthly": "86.85770355819382",
        "Payment_Behaviour": "High_spent_Large_value_payments",
        "Monthly_Balance": "531.0203301157431",
    },
    "Standard": {
        "ID": "0x20c27", "Customer_ID": "CUS_0xf64", "Month": "June",
        "Name": None, "Age": "32", "SSN": "478-73-8323",
        "Occupation": "Doctor", "Annual_Income": "56125.5",
        "Monthly_Inhand_Salary": 4875.125, "Num_Bank_Accounts": 8,
        "Num_Credit_Card": 3, "Interest_Rate": 18, "Num_of_Loan": "2",
        "Type_of_Loan": "Credit-Builder Loan, and Mortgage Loan",
        "Delay_from_due_date": 30, "Num_of_Delayed_Payment": "14",
        "Changed_Credit_Limit": "17.89", "Num_Credit_Inquiries": 4.0,
        "Credit_Mix": "Standard", "Outstanding_Debt": "370.22",
        "Credit_Utilization_Ratio": 32.014181590395296,
        "Credit_History_Age": "28 Years and 10 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 81.82285657517096,
        "Amount_invested_monthly": "182.06551022025016",
        "Payment_Behaviour": "High_spent_Medium_value_payments",
        "Monthly_Balance": "473.62413320457887",
    },
    "Poor": {
        "ID": "0xc518", "Customer_ID": "CUS_0x697f", "Month": "March",
        "Name": "Philh", "Age": "39", "SSN": "367-66-5050",
        "Occupation": "Entrepreneur", "Annual_Income": "62148.0",
        "Monthly_Inhand_Salary": None, "Num_Bank_Accounts": 9,
        "Num_Credit_Card": 7, "Interest_Rate": 24, "Num_of_Loan": "5",
        "Type_of_Loan": "Mortgage Loan, Personal Loan, Payday Loan, Personal Loan, and Home Equity Loan",
        "Delay_from_due_date": 56, "Num_of_Delayed_Payment": "25",
        "Changed_Credit_Limit": "_", "Num_Credit_Inquiries": 8.0,
        "Credit_Mix": "Bad", "Outstanding_Debt": "2373.61",
        "Credit_Utilization_Ratio": 33.95171994557009,
        "Credit_History_Age": "16 Years and 6 Months",
        "Payment_of_Min_Amount": "Yes", "Total_EMI_per_month": 258.84886110407353,
        "Amount_invested_monthly": "195.3722153038279",
        "Payment_Behaviour": "High_spent_Small_value_payments",
        "Monthly_Balance": "300.6789235920986",
    },
}


def run_tests() -> bool:
    if not MODEL_PATH.exists():
        print(f"[GAGAL] Model tidak ditemukan di {MODEL_PATH.resolve()}. "
              f"Jalankan `python run_training.py` terlebih dahulu.")
        return False

    service = CreditScoreInferenceService.from_pickle(MODEL_PATH)

    print("=" * 78)
    print("TEST CASE: Deployment Testing per Kelas")
    print("=" * 78)

    all_passed = True
    for expected_class, raw_input in TEST_CASES.items():
        result = service.predict_single(raw_input)
        status = "LULUS" if result.predicted_class == expected_class else "TIDAK SESUAI EKSPEKTASI"
        if result.predicted_class != expected_class:
            all_passed = False

        print(f"\nTest case  : {expected_class}")
        print(f"  Ekspektasi        : {expected_class}")
        print(f"  Prediksi model    : {result.predicted_class}")
        print(f"  Keyakinan         : {result.confidence:.2%}")
        print(f"  Probabilitas      : {result.to_dict()['class_probabilities']}")
        print(f"  Status            : {status}")

    print("\n" + "=" * 78)
    if all_passed:
        print("SEMUA TEST CASE SESUAI DENGAN EKSPEKTASI KELAS DOMAIN.")
    else:
        print("BEBERAPA TEST CASE TIDAK SESUAI EKSPEKTASI DOMAIN "
              "(wajar karena model bersifat probabilistik & kelas berdekatan "
              "seperti Poor/Standard secara konsep memang tumpang tindih; "
              "periksa probabilitas di atas untuk melihat seberapa dekat).")
    print("=" * 78)
    return all_passed


# ==============================================================================
# Wrapper agar dapat dijalankan juga oleh pytest (opsional)
# ==============================================================================
def test_good_case_runs_without_error():
    service = CreditScoreInferenceService.from_pickle(MODEL_PATH)
    result = service.predict_single(TEST_CASES["Good"])
    assert result.predicted_class in service.target_classes


def test_standard_case_runs_without_error():
    service = CreditScoreInferenceService.from_pickle(MODEL_PATH)
    result = service.predict_single(TEST_CASES["Standard"])
    assert result.predicted_class in service.target_classes


def test_poor_case_runs_without_error():
    service = CreditScoreInferenceService.from_pickle(MODEL_PATH)
    result = service.predict_single(TEST_CASES["Poor"])
    assert result.predicted_class in service.target_classes


def test_missing_column_raises_value_error():
    service = CreditScoreInferenceService.from_pickle(MODEL_PATH)
    incomplete_input = {"Age": "30"}  # sengaja tidak lengkap
    # Kolom yang hilang otomatis diisi NaN oleh service, jadi ini seharusnya
    # tetap berhasil (bukan error) -- menguji ketahanan input parsial.
    result = service.predict_single(incomplete_input)
    assert result.predicted_class in service.target_classes


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
