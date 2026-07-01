"""
================================================================================
 Credit Score Classification — Training Pipeline
================================================================================
Institusi keuangan: pipeline training model machine learning untuk menilai
performa kredit nasabah (Poor / Standard / Good).

Tujuan modul ini (poin b dari tugas):
    1. Membungkus seluruh proses (cleaning -> preprocessing -> training ->
       tuning -> evaluation) menjadi satu pipeline yang dapat dijalankan ulang
       (re-trainable) secara lokal, dengan satu titik masuk (entry point).
    2. Mencatat (track) setiap eksperimen — parameter, metrik, dan artefak
       model — menggunakan MLflow, sehingga proses retraining selalu
       termonitor dan dapat dibandingkan antar run.
    3. Disusun berbasis OOP murni: setiap tanggung jawab (preprocessing,
       training, tuning, evaluation, tracking, persistence) dipisah menjadi
       kelasnya masing-masing, menerapkan encapsulation, abstraction,
       inheritance, dan polymorphism — bukan kumpulan fungsi prosedural.

Cara pakai (CLI):
    python training_pipeline.py --data-path data_D.csv \
        --mlflow-tracking-uri ./mlruns \
        --experiment-name credit_score_experiment

Output:
    - models/credit_score_model.pkl  -> artefak lengkap siap deploy
    - run-run MLflow di bawah tracking URI yang ditentukan
================================================================================
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedGroupKFold,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("credit_score_pipeline")


# ==============================================================================
# 1. KONFIGURASI
# ==============================================================================
# Seluruh "angka ajaib" (magic numbers/strings) domain dikumpulkan di satu
# tempat sebagai dataclass immutable, agar pipeline mudah dikonfigurasi ulang
# tanpa menyentuh logika di kelas-kelas lain (single source of truth).
@dataclass(frozen=True)
class FeatureConfig:
    """Definisi kolom dan aturan validasi domain (hasil EDA pada notebook)."""

    target_col: str = "Credit_Score"
    group_col: str = "Customer_ID"

    # Kolom identitas yang harus dibuang sebelum masuk ke model.
    id_cols: Tuple[str, ...] = ("ID", "Customer_ID", "SSN", "Name", "Month")

    # Kolom numerik yang tersimpan sebagai teks akibat karakter sampah ("_").
    numeric_text_cols: Tuple[str, ...] = (
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance",
    )

    # Kolom numerik final yang diimputasi berbasis struktur panel nasabah.
    numeric_impute_cols: Tuple[str, ...] = (
        "Age", "Annual_Income", "Monthly_Inhand_Salary", "Num_Bank_Accounts",
        "Num_Credit_Card", "Interest_Rate", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Num_Credit_Inquiries", "Outstanding_Debt",
        "Total_EMI_per_month", "Amount_invested_monthly", "Monthly_Balance",
        "Credit_History_Age_Months",
    )

    categorical_impute_cols: Tuple[str, ...] = ("Occupation", "Credit_Mix", "Payment_Behaviour")

    # Rentang nilai valid secara domain (di luar rentang -> dianggap rusak -> NaN).
    valid_ranges: Dict[str, Tuple[float, float]] = field(default_factory=lambda: {
        "Age": (14, 100),
        "Annual_Income": (0, 2_000_000),
        "Num_Bank_Accounts": (0, 20),
        "Num_Credit_Card": (0, 20),
        "Interest_Rate": (0, 40),
        "Num_of_Loan": (0, 15),
        "Num_of_Delayed_Payment": (0, 60),
        "Num_Credit_Inquiries": (0, 50),
        "Delay_from_due_date": (-10, 100),
        "Total_EMI_per_month": (0, 10000),
        "Amount_invested_monthly": (0, 9999),
    })

    categorical_placeholder_map: Dict[str, str] = field(default_factory=lambda: {
        "Occupation": "_______",
        "Credit_Mix": "_",
        "Payment_Behaviour": "!@9#%8",
    })

    class_order: Tuple[str, ...] = ("Poor", "Standard", "Good")

    @property
    def numeric_features(self) -> List[str]:
        return list(self.numeric_impute_cols) + ["Num_Loan_Types", "Credit_Utilization_Ratio"]

    @property
    def categorical_features(self) -> List[str]:
        return list(self.categorical_impute_cols) + ["Payment_of_Min_Amount"]


@dataclass(frozen=True)
class TrainingConfig:
    """Konfigurasi proses training/eksperimen (bukan domain data)."""

    random_state: int = 42
    test_size: float = 0.2
    n_cv_splits: int = 3
    n_iter_search: int = 5
    primary_metric: str = "f1_macro"


@dataclass(frozen=True)
class MLflowConfig:
    """Konfigurasi MLflow tracking & registry."""

    # MLflow >= 3.x menonaktifkan backend filesystem murni ("./mlruns") secara
    # default (maintenance mode). SQLite lokal dipakai sebagai default yang
    # tetap 100% lokal/offline namun didukung penuh oleh versi MLflow terbaru.
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "credit_score_experiment"
    registered_model_name: str = "credit_score_classifier"


# ==============================================================================
# 2. UTILITAS PEMBERSIHAN DATA (stateless, mudah diuji unit-per-unit)
# ==============================================================================
class DataCleaningUtils:
    """Kumpulan fungsi pembersihan data murni (stateless static methods).

    Dipisahkan dari `CreditDataCleaner` agar setiap fungsi pembersihan dapat
    diuji secara terisolasi (unit testing) tanpa perlu membangun objek
    transformer scikit-learn secara penuh. Ini adalah penerapan prinsip
    Single Responsibility: kelas ini hanya tahu cara "membersihkan satu
    kolom", bukan bagaimana keseluruhan pipeline bekerja.
    """

    @staticmethod
    def to_numeric_clean(series: pd.Series) -> pd.Series:
        """Hilangkan karakter sampah (mis. '_') lalu konversi ke numerik."""
        return pd.to_numeric(
            series.astype(str).str.replace("_", "", regex=False).str.strip(),
            errors="coerce",
        )

    @staticmethod
    def parse_credit_history_age(series: pd.Series) -> pd.Series:
        """Ubah teks 'X Years and Y Months' menjadi total bulan (numerik)."""
        extracted = series.astype(str).str.extract(r"(\d+)\s*Years?\s*and\s*(\d+)\s*Months?")
        years = pd.to_numeric(extracted[0], errors="coerce")
        months = pd.to_numeric(extracted[1], errors="coerce")
        return years * 12 + months

    @staticmethod
    def count_loan_types(series: pd.Series) -> pd.Series:
        """Hitung jumlah jenis pinjaman dari kolom teks 'Type_of_Loan'."""
        s = series.astype(str)
        is_specified = series.notna() & (s.str.strip().str.lower() != "not specified")
        counts = s.str.count(",") + 1
        return pd.Series(np.where(is_specified, counts, 0), index=series.index)

    @staticmethod
    def mode_or_nan(s: pd.Series) -> Any:
        m = s.mode(dropna=True)
        return m.iloc[0] if len(m) else np.nan


# ==============================================================================
# 3. CUSTOM TRANSFORMER: CreditDataCleaner
# ==============================================================================
class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Transformer scikit-learn yang membungkus seluruh logika pembersihan.

    Dengan menjadikan ini langkah pertama di dalam `Pipeline`, logika
    pembersihan HANYA ditulis sekali dan otomatis ikut tersimpan di dalam
    artefak model (.pkl). Skrip inferencing saat deployment tidak perlu
    menduplikasi logika ini sama sekali — cukup memanggil `pipeline.predict()`
    dengan data mentah.

    Strategi imputasi (konsisten antara training & inferencing):
        - `fit`   : mempelajari median/modus GLOBAL dari data latih.
        - `transform`: jika batch membawa `Customer_ID` dengan >1 baris untuk
          nasabah yang sama, isi dahulu dengan median/modus nasabah tsb dalam
          batch yang sedang diproses (memanfaatkan struktur data panel).
          Sisanya (termasuk nasabah baru bersifat single-row) diisi dengan
          median/modus global hasil `fit`.

    Parameters
    ----------
    config : FeatureConfig
        Konfigurasi kolom & rentang nilai domain (dependency injection,
        bukan konstanta global, agar transformer ini reusable & testable).
    """

    def __init__(self, config: Optional[FeatureConfig] = None):
        # Disimpan apa adanya (scikit-learn convention: __init__ tidak boleh
        # melakukan validasi/transformasi terhadap parameter).
        self.config = config

    # -- internal helpers (encapsulation: detail privat, prefix underscore) --
    def _resolve_config(self) -> FeatureConfig:
        return self.config if self.config is not None else FeatureConfig()

    def _basic_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._resolve_config()
        df = X.copy()

        for col in cfg.numeric_text_cols:
            df[col] = DataCleaningUtils.to_numeric_clean(df[col])

        for col, (lo, hi) in cfg.valid_ranges.items():
            df.loc[~df[col].between(lo, hi), col] = np.nan

        for col, placeholder in cfg.categorical_placeholder_map.items():
            df[col] = df[col].replace(placeholder, np.nan)

        df["Credit_History_Age_Months"] = DataCleaningUtils.parse_credit_history_age(
            df["Credit_History_Age"]
        )
        df["Num_Loan_Types"] = DataCleaningUtils.count_loan_types(df["Type_of_Loan"])
        df = df.drop(columns=["Credit_History_Age", "Type_of_Loan"])
        return df

    # -- scikit-learn API (fit / transform) --
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "CreditDataCleaner":
        cfg = self._resolve_config()
        df = self._basic_clean(X)

        self.global_medians_ = {c: df[c].median() for c in cfg.numeric_impute_cols}
        self.global_modes_ = {c: DataCleaningUtils.mode_or_nan(df[c]) for c in cfg.categorical_impute_cols}
        self.feature_columns_ = cfg.numeric_features + cfg.categorical_features
        self.n_features_seen_ = len(self.feature_columns_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        cfg = self._resolve_config()
        df = self._basic_clean(X)
        has_group = cfg.group_col in df.columns

        for col in cfg.numeric_impute_cols:
            if has_group:
                grp_median = df.groupby(cfg.group_col)[col].transform("median")
                df[col] = df[col].fillna(grp_median)
            df[col] = df[col].fillna(self.global_medians_[col])

        for col in cfg.categorical_impute_cols:
            if has_group:
                df[col] = df.groupby(cfg.group_col)[col].transform(
                    lambda s: s.fillna(DataCleaningUtils.mode_or_nan(s))
                )
            df[col] = df[col].fillna(self.global_modes_[col])

        drop_cols = [c for c in cfg.id_cols if c in df.columns]
        df = df.drop(columns=drop_cols)
        return df[self.feature_columns_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.feature_columns_)


# ==============================================================================
# 4. PREPROCESSOR FACTORY
# ==============================================================================
class PreprocessorFactory:
    """Membangun tahap kedua pipeline: imputasi pengaman, scaling, encoding.

    Dipisah dari `CreditDataCleaner` karena tanggung jawabnya berbeda:
    `CreditDataCleaner` menangani "kebersihan & kebenaran nilai" (domain
    logic), sementara kelas ini menangani "representasi numerik yang siap
    dikonsumsi algoritma ML" (statistical preprocessing). Memisahkan
    keduanya membuat masing-masing lebih mudah diuji dan diganti.
    """

    @staticmethod
    def build(config: FeatureConfig) -> ColumnTransformer:
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        return ColumnTransformer(transformers=[
            ("num", numeric_transformer, config.numeric_features),
            ("cat", categorical_transformer, config.categorical_features),
        ])


# ==============================================================================
# 5. SPESIFIKASI MODEL (Abstraction + Inheritance + Polymorphism)
# ==============================================================================
class BaseModelSpec(ABC):
    """Kontrak abstrak yang harus dipenuhi setiap kandidat algoritma.

    Setiap algoritma machine learning yang ingin dibandingkan dalam
    eksperimen WAJIB mengimplementasikan kelas turunan ini. Ini adalah
    penerapan eksplisit dari *abstraction* (kontrak `build_estimator` &
    `param_distributions`) dan *polymorphism* (setiap subclass memberi
    implementasi berbeda yang dipanggil secara seragam oleh `ModelExperimentRunner`).
    """

    def __init__(self, random_state: int):
        self._random_state = random_state

    @property
    @abstractmethod
    def name(self) -> str:
        """Nama tampilan algoritma, dipakai sebagai label run di MLflow."""

    @abstractmethod
    def build_estimator(self) -> BaseEstimator:
        """Kembalikan instance estimator scikit-learn dengan parameter default
        yang masuk akal (hasil pertimbangan domain), siap dibungkus pipeline."""

    @abstractmethod
    def param_distributions(self) -> Dict[str, List[Any]]:
        """Ruang hyperparameter untuk RandomizedSearchCV.

        Dict kosong berarti algoritma ini tidak ikut tahap tuning (mis. KNN
        yang bukan kandidat realistis untuk model final pada data ini).
        """


class LogisticRegressionSpec(BaseModelSpec):
    name = "Logistic Regression"  # type: ignore[assignment]

    def build_estimator(self) -> BaseEstimator:
        return LogisticRegression(max_iter=2000, class_weight="balanced", random_state=self._random_state)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class KNearestNeighborsSpec(BaseModelSpec):
    name = "K-Nearest Neighbors"  # type: ignore[assignment]

    def build_estimator(self) -> BaseEstimator:
        return KNeighborsClassifier(n_neighbors=15)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class DecisionTreeSpec(BaseModelSpec):
    name = "Decision Tree"  # type: ignore[assignment]

    def build_estimator(self) -> BaseEstimator:
        return DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=self._random_state)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {}


class RandomForestSpec(BaseModelSpec):
    name = "Random Forest"  # type: ignore[assignment]

    def build_estimator(self) -> BaseEstimator:
        return RandomForestClassifier(
            n_estimators=80, max_depth=12, class_weight="balanced",
            n_jobs=1, random_state=self._random_state,
        )

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {
            "classifier__n_estimators": [60, 90, 120],
            "classifier__max_depth": [10, 16, 22, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": ["sqrt", "log2"],
        }


class GradientBoostingSpec(BaseModelSpec):
    name = "Gradient Boosting"  # type: ignore[assignment]

    def build_estimator(self) -> BaseEstimator:
        return GradientBoostingClassifier(n_estimators=60, max_depth=3, random_state=self._random_state)

    def param_distributions(self) -> Dict[str, List[Any]]:
        return {
            "classifier__n_estimators": [40, 60, 90],
            "classifier__max_depth": [2, 3, 4],
            "classifier__learning_rate": [0.03, 0.05, 0.1],
            "classifier__subsample": [0.8, 1.0],
        }


class ModelRegistry:
    """Daftar pusat seluruh kandidat algoritma yang akan dieksperimenkan.

    Menambah algoritma baru cukup menambahkan satu subclass `BaseModelSpec`
    dan mendaftarkannya di sini — tidak ada bagian lain dari pipeline yang
    perlu diubah (Open/Closed Principle).
    """

    _SPEC_CLASSES = (
        LogisticRegressionSpec,
        KNearestNeighborsSpec,
        DecisionTreeSpec,
        RandomForestSpec,
        GradientBoostingSpec,
    )

    def __init__(self, random_state: int):
        self._specs = [cls(random_state) for cls in self._SPEC_CLASSES]

    def all(self) -> List[BaseModelSpec]:
        return list(self._specs)

    def get(self, name: str) -> BaseModelSpec:
        for spec in self._specs:
            if spec.name == name:
                return spec
        raise KeyError(f"Model '{name}' tidak terdaftar di ModelRegistry.")


# ==============================================================================
# 6. PIPELINE FACTORY (merangkai cleaner + preprocessor + classifier)
# ==============================================================================
class CreditPipelineFactory:
    """Merangkai tiga tahap menjadi satu objek `Pipeline` scikit-learn utuh."""

    def __init__(self, feature_config: FeatureConfig):
        self._feature_config = feature_config

    def build(self, estimator: BaseEstimator) -> Pipeline:
        return Pipeline(steps=[
            ("cleaner", CreditDataCleaner(config=self._feature_config)),
            ("preprocessor", PreprocessorFactory.build(self._feature_config)),
            ("classifier", estimator),
        ])


# ==============================================================================
# 7. PENCATATAN EKSPERIMEN (MLflow wrapper)
# ==============================================================================
class MLflowExperimentTracker:
    """Encapsulation seluruh interaksi dengan MLflow.

    Bagian lain dari pipeline (runner, tuner, evaluator) tidak pernah
    memanggil `mlflow.*` secara langsung — mereka hanya berbicara dengan
    kelas ini. Jika suatu saat backend tracking diganti (mis. ke server
    remote, atau ke tool lain), hanya kelas ini yang perlu disesuaikan.
    """

    def __init__(self, config: MLflowConfig):
        self._config = config
        mlflow.set_tracking_uri(config.tracking_uri)
        mlflow.set_experiment(config.experiment_name)

    def start_run(self, run_name: str, nested: bool = False, tags: Optional[Dict[str, str]] = None):
        return mlflow.start_run(run_name=run_name, nested=nested, tags=tags or {})

    @staticmethod
    def log_params(params: Dict[str, Any]) -> None:
        # MLflow membatasi panjang string parameter; potong jika perlu.
        safe_params = {k: str(v)[:250] for k, v in params.items()}
        mlflow.log_params(safe_params)

    @staticmethod
    def log_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)

    @staticmethod
    def log_text(text: str, artifact_file: str) -> None:
        mlflow.log_text(text, artifact_file)

    @staticmethod
    def log_dict(data: Dict[str, Any], artifact_file: str) -> None:
        mlflow.log_dict(data, artifact_file)

    @staticmethod
    def log_sklearn_model(
        model: Pipeline,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        input_example: Optional[pd.DataFrame] = None,
    ) -> None:
        # `serialization_format="cloudpickle"` dipakai (bukan default "skops")
        # karena pipeline ini memuat custom transformer (`CreditDataCleaner`)
        # yang didefinisikan di modul ini sendiri — skops menolak meng-load
        # tipe custom yang tidak ada di daftar tipe terpercaya bawaannya.
        mlflow.sklearn.log_model(
            sk_model=model,
            name=artifact_path,
            registered_model_name=registered_model_name,
            input_example=input_example,
            serialization_format="cloudpickle",
        )

    def end_run(self) -> None:
        mlflow.end_run()


# ==============================================================================
# 8. DATA SPLITTING (group-aware, mencegah data leakage panel)
# ==============================================================================
class CustomerAwareSplitter:
    """Membagi data latih/uji pada level nasabah (`Customer_ID`), bukan baris.

    Karena dataset bersifat panel (satu nasabah -> banyak baris bulanan),
    pembagian acak per baris akan membocorkan informasi nasabah yang sama
    ke data latih maupun data uji sekaligus. Kelas ini mengisolasi logika
    tersebut agar konsisten dipakai di seluruh pipeline.
    """

    def __init__(self, feature_config: FeatureConfig, training_config: TrainingConfig):
        self._fc = feature_config
        self._tc = training_config

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        customer_target = df.groupby(self._fc.group_col)[self._fc.target_col].agg(
            lambda s: s.mode().iloc[0]
        )
        train_customers, _ = train_test_split(
            customer_target.index,
            test_size=self._tc.test_size,
            stratify=customer_target.values,
            random_state=self._tc.random_state,
        )
        train_mask = df[self._fc.group_col].isin(train_customers)

        X = df.drop(columns=[self._fc.target_col])
        y = df[self._fc.target_col]

        X_train, X_test = X[train_mask.values], X[~train_mask.values]
        y_train, y_test = y[train_mask.values], y[~train_mask.values]
        groups_train = X_train[self._fc.group_col]

        logger.info(
            "Split selesai | latih: %s baris / %s nasabah | uji: %s baris / %s nasabah",
            X_train.shape[0], train_customers.nunique(),
            X_test.shape[0], customer_target.index.nunique() - train_customers.nunique(),
        )
        return X_train, X_test, y_train, y_test, groups_train


# ==============================================================================
# 9. EKSPERIMEN PERBANDINGAN ALGORITMA
# ==============================================================================
class ModelExperimentRunner:
    """Menjalankan cross-validation group-aware untuk setiap `BaseModelSpec`
    dan mencatat hasilnya sebagai nested run di MLflow.
    """

    SCORING = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
    }

    def __init__(
        self,
        pipeline_factory: CreditPipelineFactory,
        training_config: TrainingConfig,
        tracker: MLflowExperimentTracker,
    ):
        self._pipeline_factory = pipeline_factory
        self._tc = training_config
        self._tracker = tracker

    def run(
        self,
        specs: List[BaseModelSpec],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
    ) -> pd.DataFrame:
        cv = StratifiedGroupKFold(
            n_splits=self._tc.n_cv_splits, shuffle=True, random_state=self._tc.random_state
        )

        results: List[Dict[str, Any]] = []
        for spec in specs:
            logger.info("Menjalankan cross-validation untuk model: %s", spec.name)
            pipe = self._pipeline_factory.build(spec.build_estimator())

            with self._tracker.start_run(run_name=f"cv_{spec.name}", nested=True, tags={"stage": "model_comparison"}):
                cv_result = cross_validate(
                    pipe, X_train, y_train, cv=cv, groups=groups_train,
                    scoring=self.SCORING, n_jobs=1,
                )
                metrics = {
                    "cv_accuracy_mean": cv_result["test_accuracy"].mean(),
                    "cv_f1_macro_mean": cv_result["test_f1_macro"].mean(),
                    "cv_f1_macro_std": cv_result["test_f1_macro"].std(),
                    "cv_precision_macro_mean": cv_result["test_precision_macro"].mean(),
                    "cv_recall_macro_mean": cv_result["test_recall_macro"].mean(),
                }
                self._tracker.log_params({
                    "model_name": spec.name,
                    "cv_strategy": "StratifiedGroupKFold",
                    "n_splits": self._tc.n_cv_splits,
                })
                self._tracker.log_metrics(metrics)

            results.append({"Model": spec.name, **metrics})
            logger.info("%s -> F1-macro CV: %.4f", spec.name, metrics["cv_f1_macro_mean"])

        results_df = pd.DataFrame(results).sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)
        return results_df


# ==============================================================================
# 10. HYPERPARAMETER TUNING
# ==============================================================================
class HyperparameterTuner:
    """Membungkus `RandomizedSearchCV` untuk model dengan F1-macro CV tertinggi."""

    def __init__(
        self,
        pipeline_factory: CreditPipelineFactory,
        training_config: TrainingConfig,
        tracker: MLflowExperimentTracker,
    ):
        self._pipeline_factory = pipeline_factory
        self._tc = training_config
        self._tracker = tracker

    def tune(
        self,
        spec: BaseModelSpec,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        groups_train: pd.Series,
    ) -> Pipeline:
        param_dist = spec.param_distributions()
        pipe = self._pipeline_factory.build(spec.build_estimator())

        if not param_dist:
            logger.info("%s tidak memiliki ruang hyperparameter terdaftar; fit langsung tanpa tuning.", spec.name)
            pipe.fit(X_train, y_train)
            return pipe

        cv = StratifiedGroupKFold(
            n_splits=self._tc.n_cv_splits, shuffle=True, random_state=self._tc.random_state
        )
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_dist,
            n_iter=self._tc.n_iter_search,
            cv=cv,
            scoring=self._tc.primary_metric,
            n_jobs=1,
            random_state=self._tc.random_state,
            verbose=0,
        )

        with self._tracker.start_run(run_name=f"tuning_{spec.name}", nested=True, tags={"stage": "hyperparameter_tuning"}):
            search.fit(X_train, y_train, groups=groups_train)
            self._tracker.log_params({"model_name": spec.name, **search.best_params_})
            self._tracker.log_metrics({"best_cv_f1_macro": search.best_score_})
            logger.info("Hyperparameter terbaik (%s): %s", spec.name, search.best_params_)
            logger.info("F1-macro CV terbaik: %.4f", search.best_score_)

        return search.best_estimator_


# ==============================================================================
# 11. EVALUASI MODEL AKHIR
# ==============================================================================
class ModelEvaluator:
    """Mengevaluasi model akhir pada data uji holdout (nasabah belum pernah dilihat)."""

    def __init__(self, feature_config: FeatureConfig, tracker: MLflowExperimentTracker):
        self._fc = feature_config
        self._tracker = tracker

    def evaluate(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        y_pred = model.predict(X_test)
        order = list(self._fc.class_order)

        metrics = {
            "test_accuracy": accuracy_score(y_test, y_pred),
            "test_f1_macro": f1_score(y_test, y_pred, average="macro"),
            "test_precision_macro": precision_score(y_test, y_pred, average="macro"),
            "test_recall_macro": recall_score(y_test, y_pred, average="macro"),
        }

        report_text = classification_report(y_test, y_pred, labels=order, target_names=order)
        cm = confusion_matrix(y_test, y_pred, labels=order)
        cm_df = pd.DataFrame(cm, index=[f"true_{c}" for c in order], columns=[f"pred_{c}" for c in order])

        self._tracker.log_metrics(metrics)
        self._tracker.log_text(report_text, "classification_report.txt")
        self._tracker.log_text(cm_df.to_csv(), "confusion_matrix.csv")

        logger.info("Evaluasi data uji selesai:")
        for k, v in metrics.items():
            logger.info("  %-22s : %.4f", k, v)
        logger.info("\n%s", report_text)

        return {"metrics": metrics, "classification_report": report_text, "confusion_matrix": cm_df}


# ==============================================================================
# 12. PERSISTENSI ARTEFAK (pickle siap deploy)
# ==============================================================================
class ArtifactManager:
    """Menyusun dan menyimpan artefak `.pkl` yang dibutuhkan skrip inferencing."""

    def __init__(self, feature_config: FeatureConfig):
        self._fc = feature_config

    def build_artifact(self, model: Pipeline, raw_feature_columns: List[str]) -> Dict[str, Any]:
        return {
            "pipeline": model,
            "raw_feature_columns": raw_feature_columns,
            "numeric_features": self._fc.numeric_features,
            "categorical_features": self._fc.categorical_features,
            "target_classes": list(self._fc.class_order),
            "valid_ranges": dict(self._fc.valid_ranges),
        }

    def save(self, artifact: Dict[str, Any], output_path: Path) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            pickle.dump(artifact, f)
        logger.info("Artefak model disimpan di: %s", output_path.resolve())
        return output_path


# ==============================================================================
# 13. ORKESTRATOR UTAMA
# ==============================================================================
class CreditScoreTrainingPipeline:
    """Titik masuk tunggal yang merangkai seluruh tahap end-to-end.

    Kelas ini sengaja "tipis" — ia tidak berisi logika domain apa pun,
    hanya mengorkestrasi kelas-kelas khusus di atas sesuai urutan alur
    kerja data science standar. Ini memudahkan pengujian setiap komponen
    secara terpisah (unit test) maupun pengujian end-to-end (integration test).
    """

    def __init__(
        self,
        data_path: Path,
        feature_config: Optional[FeatureConfig] = None,
        training_config: Optional[TrainingConfig] = None,
        mlflow_config: Optional[MLflowConfig] = None,
        output_model_path: Path = Path("models/credit_score_model.pkl"),
    ):
        self.data_path = Path(data_path)
        self.feature_config = feature_config or FeatureConfig()
        self.training_config = training_config or TrainingConfig()
        self.mlflow_config = mlflow_config or MLflowConfig()
        self.output_model_path = Path(output_model_path)

        self.tracker = MLflowExperimentTracker(self.mlflow_config)
        self.pipeline_factory = CreditPipelineFactory(self.feature_config)
        self.splitter = CustomerAwareSplitter(self.feature_config, self.training_config)
        self.experiment_runner = ModelExperimentRunner(self.pipeline_factory, self.training_config, self.tracker)
        self.tuner = HyperparameterTuner(self.pipeline_factory, self.training_config, self.tracker)
        self.evaluator = ModelEvaluator(self.feature_config, self.tracker)
        self.artifact_manager = ArtifactManager(self.feature_config)

    # -- tahap-tahap individual (publik agar dapat dipanggil/diuji terpisah) --
    def load_data(self) -> pd.DataFrame:
        logger.info("Memuat data dari: %s", self.data_path)
        df = pd.read_csv(self.data_path, index_col=0)
        logger.info("Ukuran data mentah: %s", df.shape)
        return df

    def run(self) -> Dict[str, Any]:
        df_raw = self.load_data()
        X_train, X_test, y_train, y_test, groups_train = self.splitter.split(df_raw)

        registry = ModelRegistry(self.training_config.random_state)

        with self.tracker.start_run(run_name="credit_score_training_pipeline", tags={"project": "credit_scoring"}):
            self.tracker.log_params({
                "n_train_rows": X_train.shape[0],
                "n_test_rows": X_test.shape[0],
                "test_size": self.training_config.test_size,
                "cv_splits": self.training_config.n_cv_splits,
                "random_state": self.training_config.random_state,
                "candidate_models": ", ".join(s.name for s in registry.all()),
            })

            # --- Tahap 1: perbandingan algoritma ---
            comparison_df = self.experiment_runner.run(registry.all(), X_train, y_train, groups_train)
            self.tracker.log_text(comparison_df.to_csv(index=False), "model_comparison.csv")
            logger.info("\n%s", comparison_df.to_string(index=False))

            best_model_name = comparison_df.iloc[0]["Model"]
            best_spec = registry.get(best_model_name)
            logger.info("Model terbaik dari tahap perbandingan: %s", best_model_name)
            self.tracker.log_params({"selected_model": best_model_name})

            # --- Tahap 2: hyperparameter tuning pada model terbaik ---
            final_model = self.tuner.tune(best_spec, X_train, y_train, groups_train)

            # --- Tahap 3: evaluasi akhir pada data uji holdout ---
            evaluation = self.evaluator.evaluate(final_model, X_test, y_test)

            # --- Tahap 4: logging model ke MLflow + registry ---
            self.tracker.log_sklearn_model(
                model=final_model,
                artifact_path="model",
                registered_model_name=self.mlflow_config.registered_model_name,
                input_example=X_train.head(3),
            )

            # --- Tahap 5: simpan artefak pickle lengkap untuk deployment ---
            artifact = self.artifact_manager.build_artifact(
                model=final_model,
                raw_feature_columns=list(X_train.columns),
            )
            saved_path = self.artifact_manager.save(artifact, self.output_model_path)

        return {
            "best_model_name": best_model_name,
            "comparison_df": comparison_df,
            "final_model": final_model,
            "evaluation": evaluation,
            "artifact_path": saved_path,
        }


# ==============================================================================
# 14. CLI ENTRY POINT
# ==============================================================================
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Training pipeline model Credit Score dengan MLflow tracking.")
    parser.add_argument("--data-path", type=str, default="data_D.csv", help="Path ke file data_D.csv")
    parser.add_argument("--output-model-path", type=str, default="models/credit_score_model.pkl",
                         help="Path output artefak pickle")
    parser.add_argument("--mlflow-tracking-uri", type=str, default="sqlite:///mlflow.db", help="MLflow tracking URI")
    parser.add_argument("--experiment-name", type=str, default="credit_score_experiment", help="Nama MLflow experiment")
    parser.add_argument("--registered-model-name", type=str, default="credit_score_classifier",
                         help="Nama model pada MLflow Model Registry")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proporsi nasabah untuk data uji")
    parser.add_argument("--cv-splits", type=int, default=3, help="Jumlah fold StratifiedGroupKFold")
    parser.add_argument("--n-iter-search", type=int, default=5, help="Jumlah iterasi RandomizedSearchCV")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    training_config = TrainingConfig(
        random_state=args.random_state,
        test_size=args.test_size,
        n_cv_splits=args.cv_splits,
        n_iter_search=args.n_iter_search,
    )
    mlflow_config = MLflowConfig(
        tracking_uri=args.mlflow_tracking_uri,
        experiment_name=args.experiment_name,
        registered_model_name=args.registered_model_name,
    )

    pipeline = CreditScoreTrainingPipeline(
        data_path=Path(args.data_path),
        training_config=training_config,
        mlflow_config=mlflow_config,
        output_model_path=Path(args.output_model_path),
    )

    try:
        result = pipeline.run()
    except Exception:
        logger.exception("Training pipeline gagal dijalankan.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("TRAINING SELESAI")
    logger.info("Model terpilih       : %s", result["best_model_name"])
    logger.info("F1-macro (data uji)  : %.4f", result["evaluation"]["metrics"]["test_f1_macro"])
    logger.info("Artefak tersimpan di : %s", result["artifact_path"].resolve())
    logger.info("=" * 70)


if __name__ == "__main__":
    # PENTING (kompatibilitas pickle): jika file ini dijalankan langsung
    # (`python training_pipeline.py`), Python akan mengikat seluruh kelas
    # yang didefinisikan di sini ke modul `__main__`, sehingga artefak
    # `.pkl` yang dihasilkan hanya bisa di-load ulang dari skrip yang juga
    # menjalankan file ini sebagai `__main__` — TIDAK bisa di-load dari
    # skrip inferencing terpisah (mis. `inference.py`) yang melakukan
    # `import training_pipeline`.
    #
    # Untuk menghindari jebakan ini, jalankan pipeline melalui salah satu:
    #   1) entry point terpisah: `python run_training.py ...`
    #      (lihat run_training.py, file ini TIDAK dieksekusi sebagai __main__)
    #   2) `python -m training_pipeline ...` TETAP bermasalah dengan alasan
    #      yang sama (module run sebagai __main__), jadi gunakan opsi (1).
    main()
