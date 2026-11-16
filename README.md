# AGRI-SMART DZ
AI-powered agricultural market intelligence for Algeria.

> This MVP focuses on potatoes and uses only SYNTHETIC_DEMO data. It is not a real Algerian market dataset.

## Features
- Price prediction with Random Forest + Linear Regression baselines
- Stock monitoring and regional comparison
- Shortage-risk classification (GREEN / YELLOW / RED)
- Isolation Forest anomaly detection over the recent 90-day window
- Smart alert generation and manual data entry for stock/price updates
- Interactive Streamlit dashboard for presentation and demo use
- Ten sidebar sections for the dashboard, forecasts, stock, shortage risk, anomalies, regional analysis, scenarios, alerts, reports, and project information
- Dashboard snapshots for the July 2026 demo scenario and latest available data
- Arabic/French dashboard with right-to-left layout for Arabic
- What-if supply scenario estimates for 7- and 30-day prices
- Approximate city-level market map for the demo regions
- CSV exports for the market dataset and current rule-based alerts
- Manual price and stock entry under Stock Monitoring

## Setup (Windows PowerShell)
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python src\generate_data.py
python src\database.py
python src\price_prediction.py
streamlit run app.py
```

The dashboard opens in Arabic on the July 2026 synthetic scenario. Use the
sidebar to switch to French, select the latest available snapshot, or focus on
a region. The July scenario highlights price pressure in Alger, a stock decline
in Oran, and combined pressure in Constantine. Forecasts use the latest date in
the dataset, even when viewing the historical scenario.

Use the sidebar navigation to switch between the ten sections. The Reports
section exports CSV files only; forecast target dates are anchored to the last
available dataset date and may be stale if the dataset is not updated.

The what-if panel changes the latest observed stock input and runs the trained
price models again. Treat its output as an illustrative sensitivity test, not a
causal estimate. Map points are approximate city locations, not wilaya
boundaries. Supply-chain discrepancy analysis is not enabled because the demo
does not contain verified production, warehouse-release, and wholesale-market
volume data.

## Project structure
- app.py: interactive dashboard
- src/generate_data.py: synthetic dataset generator
- src/data_processing.py: feature engineering and chronology-aware splits
- src/price_prediction.py: model training and prediction logic
- src/database.py: SQLite storage for prices, stock, alerts, and predictions
- src/translations.py: Arabic and French interface strings
- data/: demo CSV datasets
- models/: trained prediction bundles
- tests/: focused tests for chronological splitting and database behavior
- agri_smart.db: SQLite database

## Notes
- The system is designed for a local Windows MVP demo.
- It clearly labels synthetic data and confirms that detected anomalies require verification rather than accusing any party of illegal activity.
- Model artifacts are automatically retrained when their scikit-learn version does not match the pinned environment.
- Re-running `python src\database.py` safely refreshes demo rows while preserving manual entries.

## GEW 2026 dossier generator

Install the document-generation dependencies:
```powershell
python -m pip install -r tools\requirements-gew2026.txt
```

Run `python tools\generate_gew2026_dossier.py` to create six DOCX reports, seven PDFs (including the presentation PDF), an eight-slide PPTX, and a verified ZIP archive. PDF export requires LibreOffice (`soffice`) on `PATH`. Use `--no-pdf` to create the DOCX/PPTX files and archive without PDF conversion.

The generator reads the dataset inventory and model metrics from the project files. All current market data and saved model metrics are synthetic; they are not evidence of real-market accuracy or impact. It refuses to delete or overwrite an existing output directory or archive.
