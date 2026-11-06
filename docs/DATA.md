# Data dictionary

FraudGuard uses three public benchmarks of financial cybercrime. Each is turned into one
feature table by `fraudguard/datasets.py` (one `*_engineer()` function per dataset) and split
by `fraudguard/prepare.py`. The resulting tables are published as the
[`data-v4.0` release](https://github.com/amolood/fraudguard-ml/releases/tag/data-v4.0).

## Common columns

| Column | Meaning |
|---|---|
| `y` | Label: 1 = fraud / money laundering, 0 = legitimate |
| `t` | Event time used for ordering (PaySim: hour step · ULB: seconds from start · SAML-D: Unix seconds) |
| `split` | `train` / `validation` / `test`, cut at the 60th and 80th percentiles of `t` |

Rows are sorted by `t`. The split cuts on time values, so one timestamp never falls in two splits.

## Leakage rules

1. **Behavioural features look backwards only.** Every count, sum, mean or "time since previous" uses
   transactions strictly before the current one for the same account (`groupby().cumcount()`,
   cumulative sums minus the current row, `diff()`). `tests/test_datasets.py` checks that editing a
   future row leaves earlier rows unchanged.
2. **Columns that reveal the label are dropped.** SAML-D's `Laundering_type` names the laundering
   typology and is never used.
3. **Simulator artefacts are excluded from the main models.** PaySim's balance-error features are kept
   in the table for the ablation study only (see below).

## PaySim — mobile-money transfer and cash-out fraud

Source: [Kaggle `ealaxi/paysim1`](https://www.kaggle.com/datasets/ealaxi/paysim1) ·
Lopez-Rojas, Elmir & Axelsson (2016) · Licence **CC BY-SA 4.0**.

Only `TRANSFER` and `CASH_OUT` transactions are kept: fraud never occurs in the other three types.
That leaves 2,770,409 rows, 8,213 of them fraud, over 743 hourly steps.

| Feature | Definition |
|---|---|
| `type` | `TRANSFER` or `CASH_OUT` |
| `amount` | Transaction amount |
| `oldbalanceOrg`, `newbalanceOrig` | Sender balance before / after |
| `oldbalanceDest`, `newbalanceDest` | Receiver balance before / after |
| `hour` | `step mod 24` |
| `amount_to_balance` | `amount / (oldbalanceOrg + 1)` |
| `orig_emptied` | 1 if the sender's balance went from > 0 to 0 |
| `dest_zero_both` | 1 if the receiver's balance is 0 before and after |
| `dest_prior_count` | Earlier transactions received by the same account |
| `dest_prior_amount` | Total amount the same account received earlier |
| `dest_hours_since_prev` | Hours since that account last received money (empty for the first) |
| `err_balance_orig`, `err_balance_dest` | `old − amount − new` balance errors. **Excluded from the main model:** a simulator artefact |

**Caveat.** The simulator always generates fraud as "drain the whole balance". The balance fields
alone therefore separate fraud almost perfectly (test PR-AUC = 1.0). Without them, PR-AUC falls to
0.38. PaySim results say little about real-world difficulty.

## ULB — European credit-card fraud

Source: [Kaggle `mlg-ulb/creditcardfraud`](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) ·
Dal Pozzolo et al. (2015) · Licence **Open Database License (ODbL) / Database Contents License**.

284,807 real card transactions over 48 hours, 492 of them fraud.

| Feature | Definition |
|---|---|
| `V1` … `V28` | PCA components released by the data owner. The original fields are confidential |
| `Amount`, `log_amount` | Transaction amount and `log(1 + amount)` |
| `hour` | `(Time // 3600) mod 24` |

**Caveat.** The test split holds only 75 frauds and the validation split only 57, so confidence
intervals are wide. The data spans two days, so the `hour` feature drifts by construction (PSI 4.4).

## SAML-D — anti-money-laundering transaction monitoring

Source: [Kaggle `berkanoztas/synthetic-transaction-monitoring-dataset-aml`](https://www.kaggle.com/datasets/berkanoztas/synthetic-transaction-monitoring-dataset-aml) ·
Oztas et al. (2023) · Licence **CC BY-NC-SA 4.0** (non-commercial).

9,504,852 synthetic transfers from October 2022 to August 2023, 9,873 of them laundering.
The data covers 28 typologies (11 normal, 17 suspicious).

| Feature | Definition |
|---|---|
| `Amount`, `log_amount` | Amount and `log(1 + amount)` |
| `hour`, `weekday` | From the transaction timestamp |
| `Payment_type`, `Payment_currency`, `Received_currency` | Categorical |
| `Sender_bank_location`, `Receiver_bank_location` | Categorical |
| `currency_mismatch` | 1 if payment and received currency differ |
| `cross_border` | 1 if the two bank locations differ |
| `sender_prior_count` | Earlier transfers by the sender |
| `sender_prior_mean` | Mean amount of the sender's earlier transfers |
| `amount_to_sender_mean` | `Amount / sender_prior_mean` |
| `sender_secs_since_prev` | Seconds since the sender's previous transfer |
| `sender_distinct_receivers` | Distinct receivers the sender used before (fan-out) |
| `receiver_prior_count` | Earlier transfers received |
| `receiver_distinct_senders` | Distinct senders the receiver had before (fan-in) |
| `receiver_secs_since_prev` | Seconds since the receiver's previous incoming transfer |
| `pair_prior_count` | Earlier transfers between the same sender and receiver |

**Caveat.** With the behavioural features, PR-AUC is 0.978. Without them it is 0.077, so the
account-history features carry almost all of the signal. Real laundering is likely harder to separate
than this synthetic data.

## Split sizes

| Dataset | Train rows (positives) | Validation rows (positives) | Test rows (positives) |
|---|---|---|---|
| PaySim | 1,672,907 (3,193) | 544,998 (762) | 552,504 (4,258) |
| ULB | 170,888 (360) | 56,957 (57) | 56,962 (75) |
| SAML-D | 5,702,911 (5,742) | 1,900,970 (1,988) | 1,900,971 (2,143) |

Training uses all positives plus at most 1,000,000 randomly sampled negatives (seed 42). Every
algorithm gets the same sample.

## Licences of the derived tables

The processed tables are derivative works and keep the licence of their source:
`paysim.parquet` is CC BY-SA 4.0, `ulb.parquet` is ODbL, and `samld.parquet` is CC BY-NC-SA 4.0.
Cite the original authors when you use them (see the README's references).
