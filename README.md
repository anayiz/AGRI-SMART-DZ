# AGRI-SMART DZ

**AI-Powered Agricultural Market Intelligence and Early Warning Platform for Food Security in Algeria.**

AGRI-SMART DZ is a Streamlit prototype for exploring agricultural market signals. The current MVP focuses on potato prices and estimated stock across eight demo regions. It combines price forecasts, stock monitoring, rule-based shortage alerts, anomaly review, regional comparisons, and illustrative supply scenarios.

> **Data and model limitation:** The included market observations and trained model artifacts are synthetic demonstration data, not verified Algerian market data. The application does not provide real-time market access or legal findings. Alerts and anomalies are signals for verification only.

## Application screenshots

Screenshots are not included yet. Add reviewed Arabic and French desktop/mobile captures here after they have been produced and approved.

## Features

- Arabic and French interface, with RTL layout for Arabic.
- Potato price forecasts for 7- and 30-day horizons.
- Regional price and stock comparisons, plus an approximate city-level map.
- GREEN / YELLOW / RED shortage-risk indicators and rule-based alerts.
- Isolation Forest anomaly signals labeled as requiring verification.
- What-if supply adjustments, presented as illustrative model sensitivity rather than causal estimates.
- Manual price and stock entry, and CSV report exports.
- Synthetic data labeling throughout the demo workflow.

## Technology

| Area | Technology |
| --- | --- |
| Application | Python 3.12, Streamlit |
| Data and charts | pandas, NumPy, Plotly |
| Machine learning | scikit-learn (Random Forest, Linear Regression, Isolation Forest) |
| Local storage | SQLite |
| Model serialization | joblib |
| Demo inputs | CSV files in `data/` |

## Quick start

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### Linux or macOS

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app.py
```

The app initializes a local SQLite database and imports the included synthetic CSV data when the database has no potato market observations. The database is local and is intentionally excluded from Git. If model files are missing or incompatible, the prediction workflow trains replacement models from the included demo data; the first forecast request may therefore take longer.

To run the focused tests:

```bash
python -m unittest discover -s tests -v
```

## Project layout

```text
app.py                  Streamlit entry point
requirements.txt        Runtime dependencies
.streamlit/config.toml  Public UI theme settings
data/                   Synthetic demonstration CSV datasets
models/                 Locally trained model artifacts (Git-ignored)
src/database.py         SQLite schema and data access
src/data_processing.py  Feature engineering and chronological splitting
src/price_prediction.py Model training, evaluation, and forecasts
src/generate_data.py    Synthetic demo-data generator
src/translations.py     Arabic and French interface strings
tests/                  Focused unit tests
tools/                  Optional GEW 2026 dossier generator
```

## Dependencies

The application runtime dependencies and tested versions are listed in [`requirements.txt`](requirements.txt). The separate `tools/requirements-gew2026.txt` file is only needed to generate the optional competition dossier.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub after reviewing the files and confirming the push.
2. In Streamlit Community Cloud, create an app from the repository's `main` branch.
3. Set the main file path to `app.py` and select Python 3.12.
4. Deploy; Community Cloud installs dependencies from `requirements.txt`.

### Cloud storage limitation

The application currently stores manual entries in a local SQLite file. Streamlit Community Cloud does not guarantee persistence of files written to the app container across restarts or redeployments. Demo CSVs are part of the repository and can be re-imported, but manual updates should be treated as temporary. Persistent multi-user storage requires implementing and configuring an external database (for example, PostgreSQL); this project does not currently include that integration.

## Competition dossier

The optional dossier generator is in `tools/generate_gew2026_dossier.py`. Its document-generation dependencies are listed separately in `tools/requirements-gew2026.txt`. PDF conversion requires LibreOffice (`soffice`) on `PATH`; see the generator's help for available options.

## License

No license has been selected for this repository. Until a license is added, public visibility does not grant permission to reuse or redistribute the code.
