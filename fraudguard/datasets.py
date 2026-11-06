"""Dataset adapters: loading, leakage-safe feature engineering and temporal ordering.

Three public benchmarks of financial cybercrime, one adapter each:

* paysim  — mobile-money transfers / cash-outs (Lopez-Rojas et al., 2016)
* ulb     — European credit-card transactions (Dal Pozzolo et al., 2015)
* samld   — anti-money-laundering transfers, SAML-D (Oztas et al., 2023)

Every adapter returns ONE frame sorted by event time with:
  features (NUMERIC + CATEGORICAL), `y` (0/1 label) and `t` (event time, sortable).
Behavioural (velocity / history) features use only transactions strictly BEFORE
the current one, so they are available at decision time and leak nothing from
the future. The same `engineer()` is used in training and in the API batch
endpoint.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache" / "kagglehub" / "datasets"


@dataclass
class Dataset:
    key: str
    title_ar: str
    crime_ar: str
    source: str
    citation: str
    kaggle: str
    raw_file: str
    numeric: list[str]
    categorical: list[str]
    labels_ar: dict[str, str] = field(default_factory=dict)

    @property
    def features(self) -> list[str]:
        return self.numeric + self.categorical

    def raw_path(self) -> Path:
        hits = sorted(CACHE.glob(f"{self.kaggle}/versions/*/{self.raw_file}"))
        if not hits:
            raise FileNotFoundError(f"{self.raw_file} not found; run scripts/download_data.py")
        return hits[-1]


# ---------------------------------------------------------------------------
# helpers (all causal: use only rows before the current one within a group)
# ---------------------------------------------------------------------------
def _prior_count(df: pd.DataFrame, key: str) -> pd.Series:
    return df.groupby(key, sort=False).cumcount().astype("float32")


def _prior_sum(df: pd.DataFrame, key: str, col: str) -> pd.Series:
    return (df.groupby(key, sort=False)[col].cumsum() - df[col]).astype("float32")


def _since_prev(df: pd.DataFrame, key: str, tcol: str) -> pd.Series:
    return df.groupby(key, sort=False)[tcol].diff().astype("float32")  # NaN for first event


def _distinct_so_far(df: pd.DataFrame, key: str, other: str) -> pd.Series:
    first_pair = ~df.duplicated([key, other])
    # distinct counterparties seen BEFORE this row
    return (first_pair.groupby(df[key], sort=False).cumsum() - first_pair).astype("float32")


# ---------------------------------------------------------------------------
# PaySim
# ---------------------------------------------------------------------------
PAYSIM = Dataset(
    key="paysim", title_ar="تحويلات وسحب الأموال عبر الهاتف (PaySim)",
    crime_ar="احتيال التحويل والسحب النقدي (الاستيلاء على الحسابات)",
    source="https://www.kaggle.com/datasets/ealaxi/paysim1",
    citation="Lopez-Rojas, E., Elmir, A., & Axelsson, S. (2016). PaySim: A financial mobile money simulator for fraud detection. EMSS 2016.",
    kaggle="ealaxi/paysim1", raw_file="PS_20174392719_1491204439457_log.csv",
    numeric=["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "hour",
             "err_balance_orig", "err_balance_dest", "amount_to_balance", "orig_emptied", "dest_zero_both",
             "dest_prior_count", "dest_prior_amount", "dest_hours_since_prev"],
    categorical=["type"],
    labels_ar={"amount": "المبلغ", "oldbalanceOrg": "رصيد المرسل قبل", "newbalanceOrig": "رصيد المرسل بعد",
               "oldbalanceDest": "رصيد المستلم قبل", "newbalanceDest": "رصيد المستلم بعد", "hour": "ساعة العملية",
               "err_balance_orig": "خطأ توازن رصيد المرسل", "err_balance_dest": "خطأ توازن رصيد المستلم",
               "amount_to_balance": "المبلغ ÷ رصيد المرسل", "orig_emptied": "تفريغ حساب المرسل",
               "dest_zero_both": "رصيد المستلم صفر قبل وبعد", "dest_prior_count": "عمليات سابقة للمستلم",
               "dest_prior_amount": "مجموع ما استلمه سابقاً", "dest_hours_since_prev": "ساعات منذ آخر استلام",
               "type": "نوع العملية"},
)


def paysim_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw[raw["type"].isin(["TRANSFER", "CASH_OUT"])].copy()
    df = df.sort_values("step", kind="stable").reset_index(drop=True)
    out = pd.DataFrame({c: df[c].astype("float32") for c in
                        ["amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]})
    out["hour"] = (df["step"] % 24).astype("float32")
    out["err_balance_orig"] = (df.oldbalanceOrg - df.amount - df.newbalanceOrig).astype("float32")
    out["err_balance_dest"] = (df.oldbalanceDest + df.amount - df.newbalanceDest).astype("float32")
    out["amount_to_balance"] = (df.amount / (df.oldbalanceOrg + 1)).astype("float32")
    out["orig_emptied"] = ((df.newbalanceOrig == 0) & (df.oldbalanceOrg > 0)).astype("float32")
    out["dest_zero_both"] = ((df.oldbalanceDest == 0) & (df.newbalanceDest == 0)).astype("float32")
    out["dest_prior_count"] = _prior_count(df, "nameDest")
    out["dest_prior_amount"] = _prior_sum(df, "nameDest", "amount")
    out["dest_hours_since_prev"] = _since_prev(df, "nameDest", "step")
    out["type"] = df["type"].astype(str)
    out["y"] = df["isFraud"].astype("int8") if "isFraud" in df else 0
    out["t"] = df["step"].astype("int32")
    return out


# ---------------------------------------------------------------------------
# ULB credit-card
# ---------------------------------------------------------------------------
V = [f"V{i}" for i in range(1, 29)]
ULB = Dataset(
    key="ulb", title_ar="معاملات بطاقات الائتمان (ULB)", crime_ar="احتيال بطاقات الائتمان",
    source="https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
    citation="Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. IEEE SSCI.",
    kaggle="mlg-ulb/creditcardfraud", raw_file="creditcard.csv",
    numeric=V + ["Amount", "log_amount", "hour"], categorical=[],
    labels_ar={**{v: f"مكوّن مخفي {v} (PCA)" for v in V}, "Amount": "المبلغ", "log_amount": "لوغاريتم المبلغ",
               "hour": "ساعة المعاملة"},
)


def ulb_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.sort_values("Time", kind="stable").reset_index(drop=True)
    out = df[V + ["Amount"]].astype("float32")
    out["log_amount"] = np.log1p(df.Amount).astype("float32")
    out["hour"] = ((df.Time // 3600) % 24).astype("float32")
    out["y"] = df["Class"].astype("int8") if "Class" in df else 0
    out["t"] = df["Time"].astype("float64")
    return out


# ---------------------------------------------------------------------------
# SAML-D anti-money-laundering
# ---------------------------------------------------------------------------
SAMLD = Dataset(
    key="samld", title_ar="مراقبة التحويلات لمكافحة غسل الأموال (SAML-D)", crime_ar="غسل الأموال",
    source="https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml",
    citation="Oztas, B., Cetinkaya, D., Adedoyin, F., Budka, M., Dogan, H., & Aksu, G. (2023). Enhancing anti-money laundering: Development of a synthetic transaction monitoring dataset. IEEE ICEBE 2023.",
    kaggle="berkanoztas/synthetic-transaction-monitoring-dataset-aml", raw_file="SAML-D.csv",
    numeric=["Amount", "log_amount", "hour", "weekday", "currency_mismatch", "cross_border",
             "sender_prior_count", "sender_prior_mean", "amount_to_sender_mean", "sender_secs_since_prev",
             "sender_distinct_receivers", "receiver_prior_count", "receiver_distinct_senders",
             "receiver_secs_since_prev", "pair_prior_count"],
    categorical=["Payment_type", "Payment_currency", "Received_currency", "Sender_bank_location",
                 "Receiver_bank_location"],
    labels_ar={"Amount": "المبلغ", "log_amount": "لوغاريتم المبلغ", "hour": "ساعة التحويل", "weekday": "يوم الأسبوع",
               "currency_mismatch": "اختلاف عملة الدفع والاستلام", "cross_border": "تحويل عابر للحدود",
               "sender_prior_count": "تحويلات سابقة للمرسل", "sender_prior_mean": "متوسط مبالغ المرسل السابقة",
               "amount_to_sender_mean": "المبلغ ÷ متوسط المرسل", "sender_secs_since_prev": "ثوانٍ منذ آخر تحويل للمرسل",
               "sender_distinct_receivers": "عدد المستلمين المختلفين للمرسل (Fan-out)",
               "receiver_prior_count": "تحويلات سابقة للمستلم",
               "receiver_distinct_senders": "عدد المرسلين المختلفين للمستلم (Fan-in)",
               "receiver_secs_since_prev": "ثوانٍ منذ آخر استلام", "pair_prior_count": "تحويلات سابقة بين الطرفين",
               "Payment_type": "نوع الدفع", "Payment_currency": "عملة الدفع", "Received_currency": "عملة الاستلام",
               "Sender_bank_location": "دولة بنك المرسل", "Receiver_bank_location": "دولة بنك المستلم"},
)


def samld_engineer(raw: pd.DataFrame) -> pd.DataFrame:
    # Laundering_type is a post-hoc annotation that encodes the label -> never used.
    df = raw.drop(columns=[c for c in ["Laundering_type"] if c in raw.columns])
    ts = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.assign(ts=ts.astype("int64") // 10**9)
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    tsd = pd.to_datetime(df.ts, unit="s")
    out = pd.DataFrame({"Amount": df.Amount.astype("float32")})
    out["log_amount"] = np.log1p(df.Amount).astype("float32")
    out["hour"] = tsd.dt.hour.astype("float32")
    out["weekday"] = tsd.dt.weekday.astype("float32")
    out["currency_mismatch"] = (df.Payment_currency != df.Received_currency).astype("float32")
    out["cross_border"] = (df.Sender_bank_location != df.Receiver_bank_location).astype("float32")
    out["sender_prior_count"] = _prior_count(df, "Sender_account")
    prior_sum = _prior_sum(df, "Sender_account", "Amount")
    out["sender_prior_mean"] = (prior_sum / out.sender_prior_count.replace(0, np.nan)).astype("float32")
    out["amount_to_sender_mean"] = (df.Amount / out.sender_prior_mean).astype("float32")
    out["sender_secs_since_prev"] = _since_prev(df, "Sender_account", "ts")
    out["sender_distinct_receivers"] = _distinct_so_far(df, "Sender_account", "Receiver_account")
    out["receiver_prior_count"] = _prior_count(df, "Receiver_account")
    out["receiver_distinct_senders"] = _distinct_so_far(df, "Receiver_account", "Sender_account")
    out["receiver_secs_since_prev"] = _since_prev(df, "Receiver_account", "ts")
    pair = df.Sender_account.astype(str) + ">" + df.Receiver_account.astype(str)
    out["pair_prior_count"] = df.groupby(pair, sort=False).cumcount().astype("float32")
    for c in SAMLD.categorical:
        out[c] = df[c].astype(str)
    out["y"] = df["Is_laundering"].astype("int8") if "Is_laundering" in df else 0
    out["t"] = df["ts"].astype("int64")
    return out


DATASETS = {d.key: d for d in (PAYSIM, ULB, SAMLD)}
ENGINEER = {"paysim": paysim_engineer, "ulb": ulb_engineer, "samld": samld_engineer}
RAW_COLUMNS = {
    "paysim": ["step", "type", "amount", "nameOrig", "oldbalanceOrg", "newbalanceOrig", "nameDest",
               "oldbalanceDest", "newbalanceDest"],
    "ulb": ["Time"] + V + ["Amount"],
    "samld": ["Time", "Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency",
              "Received_currency", "Sender_bank_location", "Receiver_bank_location", "Payment_type"],
}
LABEL_COLUMN = {"paysim": "isFraud", "ulb": "Class", "samld": "Is_laundering"}


def label_ar(ds: Dataset, name: str) -> str:
    if name in ds.labels_ar:
        return ds.labels_ar[name]
    for c in ds.categorical:
        if name.startswith(c + "_"):
            return f"{ds.labels_ar.get(c, c)} = {name[len(c) + 1:]}"
    return name


def base_feature(ds: Dataset, name: str) -> str:
    if name in ds.numeric:
        return name
    for c in ds.categorical:
        if name.startswith(c + "_"):
            return c
    return name
