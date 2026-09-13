# Inter-Stock Dependency and Behavioural Co-Movement in Conviction-Based Stock Selection for Indian Equities

Code accompanying an MSc dissertation in Machine Learning and Artificial Intelligence.

The study asks a narrow, testable question: **when a model that selects Indian equities is allowed to see how a stock behaves relative to the other stocks it moves with, does it select better?** Almost every published stock-selection model reads one security's own history in isolation. This pipeline builds a dependency structure from the market itself, derives features from it, and then measures what those features are worth by removing them and re-running everything unchanged.

---

## What this repository contains

The full pipeline, from raw price download to backtested results, as a sequence of Google Colab notebooks.

The study was run over four equity universes. **This repository holds the code for one of them.** Every notebook up to `Code_7` is universe-independent and identical across all four; only the final modelling stage is universe-specific, and those files carry the universe in their filename.

Throughout this document, `<universe>` stands for whichever segment this copy of the repository covers — for example `midcap150`, `nifty100`, `smallcap250` or `allliquid`.

---

## The idea in one paragraph

Stocks do not move independently. Public-sector banks move together; metal producers move with the aluminium price; IT services move with the rupee. If those relationships are real and reasonably stable, then knowing what a stock's peer group is doing is information about the stock — information a single-series model throws away. The pipeline recovers those groups from return co-movement alone (no sector labels are ever supplied), builds a family of features describing where a stock sits relative to its group, and tests whether a gradient-boosted model given those features predicts return conviction better than an identical model without them.

Two design choices distinguish this from a conventional setup:

**The prediction target is defined by an exit rule, not a calendar.** Rather than forecasting the return over the next *n* days — a horizon no investor actually trades — the pipeline simulates buying on each date and holding until the price falls a fixed multiple of its own Average True Range below the running peak. The realised, annualised return of that simulated holding is what the model learns to predict.

**The model is trained and evaluated on economic consequence, not accuracy.** Committing capital to a losing position and declining a winning one are not errors of equal size. A utility matrix encodes that asymmetry and governs the training objective, the prediction rule (expected utility, not maximum probability) and hyperparameter selection alike.

---

## Pipeline

```
Code_1a ─┐
Code_1b ─┤  acquisition and screening      → data_raw.csv
Code_1c ─┤                                   indices_all.csv
Code_1d ─┤                                   commodities_all.csv
Code_1e ─┘                                   stock_index_mapping.csv
              │
Code_2        │  cleaning and integrity     → data_clean.csv
              │
Code_3        │  feature engineering        → data_features.parquet
              │
Code_4        │  dependency network         → clusters.csv
              │
Code_5        │  cluster-level series       → cluster_returns.csv
              │                               commodity_returns.csv
Code_6        │  conviction labelling       → data_exit_all.csv
              │
Code_7        │  assembly                   → dataset_all.parquet
              │
Code_8a       │  model-ready preparation
Code_8b       │  training and tuning        → best_model.pkl
Code_8c       │  backtest and evaluation    → backtest_comprehensive_report.xlsx
Code_8d       │  SHAP attribution           → feature_importance.csv
```

### Stage by stage

| Notebook | Purpose | Key inputs | Key outputs |
|---|---|---|---|
| `Code_1a_Fetch_Raw_Data` | Downloads adjusted and unadjusted OHLCV for the NSE universe | `unique_tickers.csv` | `data_raw_all.csv`, `metadata_raw_all.csv`, `tickers_not_found.csv` |
| `Code_1b_Filter_Liquid_Securities` | Applies the liquidity and tradability screen | `data_raw_all.csv` | `data_raw.csv`, `liquidity_analysis.csv`, `filtered_out_tickers.csv` |
| `Code_1c_Fetch_Indices_Data` | Downloads index reference series | — | `indices_raw_all.csv`, `indices_all.csv` |
| `Code_1d_Fetch_Commodities_Data` | Downloads commodity and FX reference series | — | `commodities_raw_all.csv`, `commodities_all.csv` |
| `Code_1e_Map_Stocks_to_Indices` | Maps each security to its index membership | `data_raw.csv` | `stock_index_mapping.csv` |
| `Code_2_Clean_Preprocess` | Treats missing values by cause; audits price-series integrity | `data_raw.csv` | `data_clean.csv`, `cleaning_report.csv`, `ticker_level_cleaning.csv` |
| `Code_3_Add_Features` | Builds the feature families and the feature reference | `data_clean.csv`, `indices_all.csv` | `data_features.parquet`, `feature_dictionary.csv`, `feature_health_report.csv` |
| `Code_4` | Correlation graph, filtering, clustering, commodity anchoring | `data_features.parquet`, `commodities_all.csv` | `clusters.csv`, `silhouette_grid.csv` |
| `Code_5` | Cluster and commodity return, volatility and drawdown series | `clusters.csv`, `data_features.parquet` | `cluster_returns.csv`, `commodity_returns.csv` |
| `Code_6_Conviction_Labeling` | Simulates the adaptive exit and assigns conviction tiers | `data_features.parquet` | `data_exit_all.csv`, `failed_stocks_code6.csv` |
| `Code_7_Combine_Final_Dataset` | Joins features, clusters, cluster returns and labels | all of the above | `dataset_all.parquet`, `merge_summary.csv` |
| `Code_8a` | Universe subsetting, label audit, model-ready shaping | `dataset_all.parquet`, `feature_reference.csv` | `data_preparation_summary.csv`, `label_nan_audit.csv` |
| `Code_8b_<universe>_{with,without}_cluster` | Grid search, walk-forward validation, training | prepared dataset | `best_model.pkl`, `cv_fold_metrics.csv`, `feature_importance.csv` |
| `Code_8c_<universe>_{with,without}_cluster` | Out-of-sample backtest over both test periods | trained model | `backtest_comprehensive_report.xlsx`, `trades_*.csv`, `cashflows_*.csv` |
| `Code_8d_SHAP_<universe>_{with,without}_cluster` | Exact SHAP attribution over the training partition | trained model | `feature_ranking_full.csv`, `importance_comparison.csv` |

---

## The ablation: `with_cluster` and `without_cluster`

Stages `8b`, `8c` and `8d` each exist in two variants. This is the study's central experiment and the reason the file naming matters.

Every feature produced by `Code_3` and `Code_5` carries a flag in `feature_reference.csv` recording whether it requires information from beyond the individual security. The `without_cluster` variant is produced by applying that single flag as a filter. Nothing else changes — same data, same algorithm, same grid, same folds, same seeds.

That matters because it means the two models cannot differ in any respect other than the one under test. Hand-curating a second feature list would leave room for the two configurations to diverge in ways nobody intended.

Run both. The comparison between them is the result.

---

## Controls against look-ahead bias

Financial backtests fail quietly. A single leak inflates measured performance more than any modelling improvement could, and leaves no visible trace in aggregate diagnostics. The pipeline applies these controls, and they are worth preserving in any modification:

- The **liquidity screen** is computed over an early window only. A security qualifies on the liquidity it had before the evaluation period, never on liquidity it acquired later.
- **Missing values are filled forward in the direction of time only.** Interpolation is prohibited throughout, because interpolating uses a later price to fill an earlier gap.
- **Every feature is computed from a backward-looking window** terminating at the observation date.
- **Cluster membership is re-estimated annually from strictly prior data.** The structure applied during a given year is derived from the years before it.
- **Labels are aligned one day forward** of the features that predict them.
- **Feature selection and pruning happen inside training folds**, never on the full sample.
- **Both test periods are touched exactly once**, at evaluation.

No transformation anywhere in the pipeline is fitted on the full sample. Scaling is avoided altogether rather than managed, which removes an entire class of leakage rather than attempting to control it.

---

## Running it

Everything was developed and run on **Google Colab, free tier**. No paid compute, no proprietary data, no licensed feeds.

### Setup

1. Create a folder in Google Drive for the project and place `unique_tickers.csv` in it.
2. Open the notebooks in Colab in numerical order.
3. Set the Drive folder path in the configuration cell at the top of each notebook.
4. Run each notebook to completion before starting the next — each stage consumes the persisted output of its predecessor.

### Order

```
1a → 1b → 1c → 1d → 1e → 2 → 3 → 4 → 5 → 6 → 7 → 8a → 8b → 8c → 8d
```

Stages `8b` through `8d` are run twice, once for each feature configuration.

### Practical notes

- **`Code_6` is the long one.** It simulates an exit from every possible entry date for every security. Expect it to run in pieces across several sessions on free-tier compute. It writes progress as it goes and can be resumed.
- **`Code_4` has a tunable threshold.** The clustering threshold controls how correlated two securities must be before they are grouped. Raising it produces tighter clusters covering less of the universe; lowering it does the reverse. `silhouette_grid.csv` records the trade-off across the range so the choice can be inspected rather than assumed. The study adopted **0.40**.
- **Intermediate files are large.** `data_features.parquet` and `dataset_all.parquet` run to millions of rows. Keep them in Drive rather than Colab's local disk, which is cleared between sessions.

---

## Configuration

The main parameters, all set in the configuration cell of the notebook that uses them:

| Parameter | Notebook | Study value |
|---|---|---|
| Price history start | `1a` | 2009-07-01 |
| Index / commodity history start | `1c`, `1d` | 2011-01-01 |
| Liquidity screen window | `1b` | early window, pre-study |
| Correlation threshold | `4` | 0.40 |
| Correlation window | `4` | 60 trading days |
| Cluster refresh frequency | `4` | annual, from prior data |
| ATR exit multiple | `6` | 4.0 |
| Conviction tiers | `6` | High ≥ 40%, Medium 20–40%, Low 10–20%, Ignore < 10% (annualised) |
| Training window | `8b` | 2011–2019 |
| Test period 1 | `8c` | 2020–2022 |
| Test period 2 | `8c` | 2023–2025 |
| Walk-forward folds | `8b` | 3 |
| Grid size | `8b` | 108 configurations |
| Transaction cost | `8c` | 25 bps per round trip |

Tier boundaries are **economic reference points, not quantiles** — roughly the return available without equity risk, the long-run return of the broad market, and a substantial premium over it. This is deliberate: a quantile-based label would mean something different in a bull market than in a bear market, and the tiers would stop being comparable across periods.

---

## Requirements

```
pandas
numpy
scipy
scikit-learn
xgboost
shap
imbalanced-learn
ta
matplotlib
xlsxwriter
pyarrow
yfinance
```

All available on Colab, most pre-installed. `ta`, `shap` and `yfinance` generally need installing.

---

## Data

Price and volume data is downloaded from Yahoo Finance via `yfinance`. Index and commodity reference series come from the same source. No paid data is used anywhere in the pipeline.

Two consequences worth stating plainly. Yahoo's coverage of Indian equities is good but not perfect — `tickers_not_found.csv` and `filtered_out_tickers.csv` record what was dropped and why. And because the data is re-downloaded rather than versioned, **exact reproduction depends on the vendor's history being stable.** Corporate-action adjustments are occasionally revised retrospectively.

---

## Outputs

The final evaluation writes a multi-sheet workbook, `backtest_comprehensive_report.xlsx`, containing:

- Pooled internal rate of return by predicted conviction tier
- The same, broken down by year
- Positions entered, by tier and entry year
- Exit-mechanism decomposition — how positions actually closed
- Excess return over the matched total-return benchmark
- Trade-level and cashflow-level extracts

Alongside it, `feature_importance.csv` and `feature_ranking_full.csv` carry the SHAP attribution, flagged by whether each feature requires inter-stock information.

Benchmarks are **matched to the universe**. A mid-capitalisation strategy is measured against a mid-capitalisation total-return index, with dividends. Comparing a smaller-capitalisation universe against a large-capitalisation index would report the size premium as if it were selection skill.

---

## Scope and limitations

Stated plainly, because they bound what the code can support:

- **Information set.** Price, volume and structure only. No fundamentals, no news, no sentiment, and no derivative-market data — no open interest, implied volatility or futures basis.
- **Frequency.** Daily bars throughout. Intraday behaviour, including the path by which a stop level is reached within a session, is not modelled.
- **Instruments.** Cash-segment equities. Nothing here addresses derivative expression or hedged construction.
- **Position sizing.** Every position is opened at the same notional weight. This isolates signal quality, which is what the study measures, but it means the pipeline says nothing about capital allocation across simultaneous holdings, and nothing about market impact or capacity.
- **Exit parameter.** The ATR multiple is fixed for every security and every market state. It is not tuned and not conditioned on regime. A trailing stop performs worst in flat, oscillating markets.
- **Market.** Indian equities only.

---

## Repository conventions

- Notebooks are numbered in execution order. The number is the contract — `Code_5` will not run until `Code_4` has written `clusters.csv`.
- Files named `*_report.csv`, `*_audit.csv` and `*_summary.csv` are diagnostics, not pipeline inputs. They exist so that each stage can be checked before the next one is run, and they are worth reading.
- `feature_reference.csv` is the schema of record for the feature set. If features are added, they must be registered there with their dependency flag, or the ablation will silently mis-classify them.

---

## Citation

If this code or method is useful in your own work, please cite the dissertation:

> Saxena, P. (2026) *Inter-Stock Dependency and Behavioural Co-Movement in Conviction-Based Stock Selection for Indian Equities*. MSc dissertation, MSc Machine Learning and Artificial Intelligence.

---

## Disclaimer

This is academic research. It is not investment advice, and nothing here is a recommendation to buy or sell any security. Backtested results are simulated, rest on assumptions documented in the dissertation, and are not a reliable indicator of future performance. Anyone deploying this would need to satisfy themselves independently about execution, capacity and regulatory obligations.
