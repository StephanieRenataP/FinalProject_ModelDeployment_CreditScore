"""
================================================================================
 Credit Score — Inference Module
================================================================================
Modul ini bertanggung jawab HANYA untuk memuat artefak model (.pkl) hasil
training (`training_pipeline.py`) dan menjalankan prediksi pada data mentah
baru — tanpa menulis ulang satu baris pun logika cleaning/preprocessing,
karena semua itu sudah terbungkus di dalam `pipeline` (lihat `CreditDataCleaner`
di `training_pipeline.py`).

Dipakai oleh:
    - `app.py` (Streamlit web app)
    - `test_inference.py` (test case per kelas)
    - skrip/batch job lain yang butuh prediksi credit score

PENTING: modul ini meng-import `training_pipeline` agar class custom
(`CreditDataCleaner`, dst.) yang dibutuhkan `pickle.load` untuk merekonstruksi
objek pipeline tersedia di memory. Jangan dihapus meskipun terlihat tidak
dipakai langsung.
================================================================================
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

import b_TrainingPipeline  # noqa: F401  (wajib: registrasi CreditDataCleaner utk unpickle)


# ==============================================================================
# 1. HASIL PREDIKSI (value object)
# ==============================================================================
@dataclass(frozen=True)
class PredictionResult:
    """Satu hasil prediksi untuk satu baris/nasabah."""

    predicted_class: str
    class_probabilities: Dict[str, float]

    @property
    def confidence(self) -> float:
        return self.class_probabilities[self.predicted_class]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in self.class_probabilities.items()},
        }


# ==============================================================================
# 2. VALIDATOR INPUT
# ==============================================================================
class InputValidator:
    """Memastikan data mentah yang dikirim pengguna memiliki kolom yang
    dibutuhkan pipeline sebelum diproses, dan memberi pesan kesalahan yang
    jelas jika tidak — daripada membiarkan scikit-learn melempar traceback
    teknis yang membingungkan pengguna akhir web app.
    """

    def __init__(self, required_columns: List[str]):
        self._required_columns = required_columns

    def validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in self._required_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"Kolom input berikut wajib ada namun tidak ditemukan: {missing}. "
                f"Kolom yang dibutuhkan: {self._required_columns}"
            )
        if df.empty:
            raise ValueError("Data input kosong (0 baris).")


# ==============================================================================
# 3. SERVICE INFERENCING UTAMA
# ==============================================================================
class CreditScoreInferenceService:
    """Membungkus artefak model (.pkl) dan menyediakan API prediksi yang bersih.

    Contoh pemakaian:
        service = CreditScoreInferenceService.from_pickle("models/credit_score_model.pkl")
        result = service.predict_single({"Age": "35", "Annual_Income": "50000", ...})
        results = service.predict_batch(dataframe_mentah)
    """

    def __init__(self, artifact: Dict[str, Any]):
        self._pipeline = artifact["pipeline"]
        self._raw_feature_columns: List[str] = artifact["raw_feature_columns"]
        self._target_classes: List[str] = artifact["target_classes"]
        self._validator = InputValidator(self._raw_feature_columns)

    # -- factory method (constructor alternatif) --
    @classmethod
    def from_pickle(cls, model_path: Union[str, Path]) -> "CreditScoreInferenceService":
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"File model tidak ditemukan: {model_path.resolve()}")
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)
        return cls(artifact)

    # -- properti publik (read-only, encapsulation) --
    @property
    def required_columns(self) -> List[str]:
        return list(self._raw_feature_columns)

    @property
    def target_classes(self) -> List[str]:
        return list(self._target_classes)

    # -- API prediksi --
    def _prepare_frame(self, data: Union[Dict[str, Any], pd.DataFrame]) -> pd.DataFrame:
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Lengkapi kolom yang tidak dikirim (mis. ID/SSN/Name yang bersifat
        # identitas dan tidak wajib diisi pengguna web) dengan NaN, karena
        # kolom-kolom tsb pada akhirnya dibuang oleh CreditDataCleaner dan
        # tidak memengaruhi prediksi.
        for col in self._raw_feature_columns:
            if col not in df.columns:
                df[col] = np.nan

        return df[self._raw_feature_columns]

    def predict_batch(self, data: Union[Dict[str, Any], pd.DataFrame]) -> List[PredictionResult]:
        df = self._prepare_frame(data)
        self._validator.validate(df)

        predictions = self._pipeline.predict(df)
        probabilities = self._pipeline.predict_proba(df)
        classifier_classes = list(self._pipeline.named_steps["classifier"].classes_)

        results: List[PredictionResult] = []
        for pred, proba_row in zip(predictions, probabilities):
            proba_dict = {cls: float(p) for cls, p in zip(classifier_classes, proba_row)}
            results.append(PredictionResult(predicted_class=str(pred), class_probabilities=proba_dict))
        return results

    def predict_single(self, data: Dict[str, Any]) -> PredictionResult:
        return self.predict_batch(data)[0]


# ==============================================================================
# 4. CLI CEPAT UNTUK CEK MANUAL (opsional)
# ==============================================================================
def _demo() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Uji cepat inference dari CLI.")
    parser.add_argument("--model-path", default="models/credit_score_model.pkl")
    parser.add_argument("--csv-path", required=True, help="Path CSV data mentah (format sama seperti data_D.csv)")
    args = parser.parse_args()

    service = CreditScoreInferenceService.from_pickle(args.model_path)
    df = pd.read_csv(args.csv_path)
    results = service.predict_batch(df)
    for i, r in enumerate(results):
        print(f"Baris {i}: {r.to_dict()}")


if __name__ == "__main__":
    _demo()
