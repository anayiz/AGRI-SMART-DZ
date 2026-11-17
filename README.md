# AGRI-SMART DZ

AGRI-SMART DZ is a Streamlit agricultural market-intelligence prototype for Algeria. It combines a multi-product catalog, sourced market observations, price-history analysis, experimental forecasts, explainable early-warning signals, regional visualization, hypothetical scenarios, and downloadable reports. The original potato demonstration remains available as a separate classic view.

> **Data integrity:** Bundled and generated sample observations are synthetic demonstrations, not verified market data. The application does not currently connect to Algerian ministry feeds, official statistics APIs, or live wholesale/retail markets. Manual or imported observations retain their source and quality status; users must verify their provenance.

> **Operational limitation:** Forecasts and risk alerts are analytical signals, not guarantees, official shortage declarations, or substitutes for field verification. The included wilaya GeoJSON has no documented provenance or license in this repository; validate its boundaries and usage rights before operational use.

## Features

- **Multi-product monitoring:** Initial configurable catalog across vegetables, oils, fruits, cereals, and essential foods, including potato, tomato, onion, garlic, cooking oil, milk, sugar, rice, wheat, flour, and semolina. Additional products and units can be added in the data-management page.
- **Observation management:** Manual entry and CSV/Excel import for date, product, wilaya/region, market, price, quantity, incoming supply, unit, source, collection method, quality, and notes. Import validation reports invalid records and duplicate rows before saving.
- **Source and quality labels:** `OFFICIAL`, `VERIFIED`, `UNVERIFIED`, and `SYNTHETIC_DEMO` are stored with observations. These labels record supplied metadata; the app does not independently verify sources.
- **Market dashboard:** Filterable product/category/region/date/quality views, historical trends, latest regional price comparisons, stock indicators, and recent explainable risk alerts. Averages across distinct products or units are intentionally not combined.
- **Algeria map:** Plotly renders the bundled 58-feature GeoJSON, colors matching observed wilayas by the app's rule-based risk level, and leaves unmatched areas as no data. Selecting a wilaya shows available latest observations and history. Boundary provenance and licensing are not documented.
- **Forecast experiments:** Separate 7- and 30-day forecasts by product, unit, and region. Linear Regression, Random Forest, and Gradient Boosting are evaluated against a persistence baseline on a chronological holdout. The page reports MAE, RMSE, and R² and distinguishes observed holdout values from forecasts. Empirical holdout residual ranges are indicative, not guaranteed calibrated confidence intervals.
- **Early warnings:** Rule-based price-change and reported-stock comparisons produce explainable anomaly/watch signals and suggested verification actions. A price anomaly alone is not labeled a confirmed shortage; the system does not autonomously confirm shortages.
- **Hypothetical scenarios:** Production, demand, distribution delay, stock, and transport-cost adjustments are used in a transparent sensitivity calculation. Results are hypothetical, not causal estimates or market predictions.
- **Reports:** Filtered market observations and alerts can be downloaded as CSV, Excel, and PDF. Observation dates, sources, quality, units, and alert indicators are included where available.
- **Languages and visual design:** Arabic (RTL), French, and English with the green AGRI-SMART DZ palette.
- **Classic potato experience:** The pre-existing potato demo views remain accessible from navigation.

## Requirements and local setup

Python 3.10+ is recommended. Runtime dependencies are pinned in `requirements.txt`.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux or macOS

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The app initializes a local SQLite database (`agri_smart.db`) and seeds synthetic demonstration observations when needed. The classic potato module also retains its original demo data flow. Do not treat either dataset as observed market data.

Run the automated tests from the project root:

```bash
python -m unittest discover -s tests -v
```

## Persistence and deployment

SQLite is the zero-configuration local/demo backend. Streamlit Community Cloud does not guarantee persistence of files written to the application container, so SQLite there is not suitable for durable production records.

The SQLAlchemy persistence layer supports PostgreSQL through `DATABASE_URL`. Set it in the deployment platform's secret/environment-variable manager (never commit credentials), for example:

```text
postgresql://<user>:<password>@<host>:5432/<database>
```

The app initializes its schema on startup. Configure a managed PostgreSQL service, access controls, backups, and a secure connection string before relying on hosted records. This repository does not provision that database or provide real-data integrations.

## Repository structure

```text
app.py                         Streamlit UI and feature integration
src/database.py                SQLAlchemy schema, SQLite/PostgreSQL persistence
src/data_import.py             CSV/Excel parsing, validation, duplicate handling
src/market_intelligence.py     Forecasting, risk signals, scenario sensitivity
src/reporting.py               CSV, Excel, and PDF report generation
src/data_processing.py         Existing data cleanup and feature preparation
src/price_prediction.py        Existing classic potato model workflow
src/generate_data.py           Synthetic demonstration data generation
src/translations.py            Arabic/French translations with English fallback
data/                           Demo data and wilaya GeoJSON
tests/                          Unit tests for import, intelligence, and reports
```

## Known limitations

- No live or authorized official Algerian market/statistics data source is connected.
- Quality values such as `OFFICIAL` and `VERIFIED` are user-supplied labels, not proof of verification.
- The GeoJSON's source, boundary accuracy, and license are undocumented; do not make operational coverage claims until these are verified.
- Forecast suitability depends on sufficient clean historical observations. Metrics are computed from the available chronological holdout and should not be generalized beyond it.
- Shortage alerts are threshold-based indicators using reported price and stock, not confirmed shortages; supply, production, demand, and seasonality coverage is currently limited by available observations.
- Scenario outputs are simple sensitivity calculations and do not estimate causal market effects.
- Hosted durable storage requires an externally provisioned PostgreSQL database and correctly configured secrets.

## Optional competition dossier

The dossier generator is in `tools/generate_gew2026_dossier.py`; its optional dependencies are listed in `tools/requirements-gew2026.txt`. PDF conversion requires LibreOffice (`soffice`) on `PATH`.

## License

No license has been selected for this repository. Public visibility alone does not grant permission to reuse or redistribute the code.
