import numpy as np
import pandas as pd

from fraudguard.datasets import paysim_engineer, samld_engineer
from fraudguard.prepare import temporal_split


def _samld(rows):
    cols = ["Time", "Date", "Sender_account", "Receiver_account", "Amount", "Payment_currency", "Received_currency",
            "Sender_bank_location", "Receiver_bank_location", "Payment_type", "Is_laundering", "Laundering_type"]
    return pd.DataFrame(rows, columns=cols)


def test_samld_behaviour_uses_only_the_past_and_drops_label_leak():
    raw = _samld([
        ["10:00:00", "2023-01-01", 1, 9, 100.0, "UK pounds", "UK pounds", "UK", "UK", "Cheque", 0, "Normal_Fan_Out"],
        ["10:00:05", "2023-01-01", 1, 8, 50.0, "UK pounds", "Euro", "UK", "Spain", "Cross-border", 1, "Smurfing"],
        ["10:00:09", "2023-01-01", 1, 9, 70.0, "UK pounds", "UK pounds", "UK", "UK", "Cheque", 0, "Normal_Fan_Out"],
    ])
    out = samld_engineer(raw)
    assert "Laundering_type" not in out.columns
    assert out.sender_prior_count.tolist() == [0, 1, 2]
    assert out.sender_distinct_receivers.tolist() == [0, 1, 2]
    assert out.pair_prior_count.tolist() == [0, 0, 1]
    assert np.isnan(out.sender_secs_since_prev.iloc[0]) and out.sender_secs_since_prev.iloc[1] == 5
    assert out.sender_prior_mean.iloc[2] == 75 and out.currency_mismatch.tolist() == [0, 1, 0]
    # changing a FUTURE row must not change features of earlier rows
    raw2 = raw.copy(); raw2.loc[2, "Amount"] = 1e6
    pd.testing.assert_frame_equal(samld_engineer(raw2).iloc[:2], out.iloc[:2])


def test_paysim_keeps_only_fraud_relevant_types_and_balance_errors():
    raw = pd.DataFrame({"step": [1, 1, 2], "type": ["PAYMENT", "TRANSFER", "CASH_OUT"], "amount": [10, 100, 100],
                        "nameOrig": ["a", "b", "c"], "oldbalanceOrg": [50, 100, 300], "newbalanceOrig": [40, 0, 200],
                        "nameDest": ["m", "d", "d"], "oldbalanceDest": [0, 0, 100], "newbalanceDest": [0, 0, 200],
                        "isFraud": [0, 1, 0]})
    out = paysim_engineer(raw)
    assert len(out) == 2 and out.orig_emptied.tolist() == [1, 0]
    assert out.dest_prior_count.tolist() == [0, 1] and out.dest_hours_since_prev.iloc[1] == 1


def test_temporal_split_never_puts_one_timestamp_in_two_splits():
    t = pd.Series(np.repeat(np.arange(100), 3))
    s = pd.Series(temporal_split(t))
    assert s.groupby(t).nunique().max() == 1
    assert list(pd.unique(s)) == ["train", "validation", "test"]
