import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

from src import data_processing as dp
from src import database as db
from src import data_import, market_intelligence, reporting
from src.price_prediction import model_is_current, predict_latest, train_and_evaluate
from src.translations import translate, translate_region

ROOT = Path(__file__).resolve().parent
SCENARIO_DATE = pd.Timestamp("2026-07-24")
REGION_COORDINATES = {
    "Alger": (36.7538, 3.0588),
    "Oran": (35.6971, -0.6308),
    "Constantine": (36.3650, 6.6147),
    "Mostaganem": (35.9311, 0.0892),
    "Ain Defla": (36.2641, 1.9679),
    "El Oued": (33.3561, 6.8672),
    "Setif": (36.1900, 5.4100),
    "Tizi Ouzou": (36.7118, 4.0459),
}
ALERT_COLUMNS = [
    "region",
    "date",
    "alert_level",
    "reason",
    "suggested_action",
    "price_change_pct",
    "stock_ratio",
]
NAVIGATION_SECTIONS = (
    ("dashboard", "Dashboard"),
    ("price_prediction", "Price Prediction"),
    ("stock_monitoring", "Stock Monitoring"),
    ("shortage_detection", "Shortage Detection"),
    ("anomaly_detection", "Anomaly Detection"),
    ("regional_analysis", "Regional Market Analysis"),
    ("what_if", "What-if Simulation"),
    ("smart_alerts", "Smart Alerts"),
    ("reports", "Reports"),
    ("about", "About AGRI-SMART DZ"),
)


st.set_page_config(
    page_title="AGRI-SMART DZ",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)


def tr(text, **values):
    return translate(st.session_state.get("language", "ar"), text, **values)


def apply_visual_theme():
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp, [data-testid="stAppViewContainer"] {
            background: #F5F7F5 !important;
            color: #164B35 !important;
        }
        [data-testid="stHeader"] { background: #F5F7F5 !important; }
        [data-testid="stSidebar"] { background: #DCEFE3 !important; }
        [data-testid="stSidebar"] * { color: #164B35; }
        h1, h2, h3 { color: #164B35; }
        [data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #DCEFE3;
            border-radius: 12px;
            padding: 14px;
        }
        [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {
            color: #164B35 !important;
            opacity: 1;
        }
        [data-testid="stMetricValue"] {
            color: #164B35;
            font-size: 1.35rem;
            white-space: normal;
            overflow-wrap: anywhere;
        }
        [data-baseweb="select"] > div, input, textarea {
            background: #FFFFFF !important;
            color: #164B35 !important;
        }
        input[type="radio"], input[type="checkbox"] { accent-color: #2E8B57; }
        .stButton > button {
            background: #2E8B57;
            border: 1px solid #2E8B57;
            border-radius: 8px;
            color: #FFFFFF;
        }
        .stButton > button:hover { background: #164B35; border-color: #164B35; color: #FFFFFF; }
        [data-testid="stAlert"] { border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def style_chart(figure):
    figure.update_layout(
        template="plotly_white",
        paper_bgcolor="#F5F7F5",
        plot_bgcolor="#FFFFFF",
        font={"color": "#164B35"},
        colorway=["#164B35", "#2E8B57", "#78A98B", "#D49B26", "#9B5555"],
    )
    figure.update_xaxes(gridcolor="#E5EEE7", zerolinecolor="#DCEFE3")
    figure.update_yaxes(gridcolor="#E5EEE7", zerolinecolor="#DCEFE3")
    return figure


def select_language():
    selection = st.sidebar.selectbox(
        translate("ar", "Language / Langue / اللغة"),
        ["العربية", "Français", "English"],
        index=0,
    )
    language = {"العربية": "ar", "Français": "fr", "English": "en"}[selection]
    st.session_state["language"] = language
    if language == "ar":
        st.markdown(
            "<style>"
            ".stApp {direction: ltr;}"
            "[data-testid='stSidebar'], [data-testid='stMainBlockContainer'] {direction: rtl;}"
            ".stMarkdown, [data-testid='stMetric'], label {text-align: right;}"
            "</style>",
            unsafe_allow_html=True,
        )
    return language


def select_section():
    labels = dict(NAVIGATION_SECTIONS)
    return st.sidebar.radio(
        tr("Navigation"),
        list(labels),
        format_func=lambda section: tr(labels[section]),
        label_visibility="collapsed",
    )


def ensure_demo_data():
    prices_file = ROOT / "data" / "potato_prices.csv"
    stock_file = ROOT / "data" / "potato_stock.csv"
    if not prices_file.exists() or not stock_file.exists():
        st.warning("Demo datasets are missing. Generating synthetic potato market data...")
        import subprocess
        import sys

        subprocess.run([sys.executable, str(ROOT / "src" / "generate_data.py")], check=True)
    db.init_db()
    if not db.has_market_data("Potato"):
        db.load_csv_into_db(reset=False)


@st.cache_data
def load_dashboard_data():
    clean = dp.load_clean("Potato")
    clean["date"] = pd.to_datetime(clean["date"])
    return clean.sort_values(["region", "date"]).reset_index(drop=True)


@st.cache_data
def get_model_predictions(clean_df):
    for horizon in dp.HORIZONS:
        if not model_is_current(horizon):
            train_and_evaluate(horizon, clean_df)
    preds = {}
    for horizon in dp.HORIZONS:
        p, _ = predict_latest(horizon, clean_df)
        preds[horizon] = p
    return preds


def select_snapshot(clean_df):
    latest_date = clean_df["date"].max().normalize()
    options = [tr("July 2026 demo scenario"), tr("Latest available data")]
    selection = st.sidebar.radio(tr("Dashboard snapshot"), options, index=0)
    is_scenario = selection == options[0]
    snapshot_date = SCENARIO_DATE if is_scenario else latest_date
    snapshot_date = min(snapshot_date, latest_date)
    snapshot_df = clean_df[clean_df["date"] <= snapshot_date].copy()
    st.sidebar.caption(tr("Showing data through {date}.", date=f"{snapshot_date:%d %b %Y}"))
    return snapshot_df, is_scenario


def region_label(region):
    return translate_region(st.session_state.get("language", "ar"), region)


def risk_level_for_region(region_df):
    region_df = region_df.sort_values("date")
    latest = region_df.iloc[-1]
    price_history = region_df["price"].iloc[max(0, len(region_df) - 30):-1]
    price_baseline = price_history.mean() if not price_history.empty else latest["price"]
    stock_baseline = region_df["stock"].iloc[max(0, len(region_df) - 30)]
    price_change = (latest["price"] - price_baseline) / price_baseline if price_baseline else 0
    stock_change = (latest["stock"] - stock_baseline) / stock_baseline if stock_baseline else 0
    stock_ratio = latest["stock"] / region_df["stock"].rolling(30, min_periods=1).mean().iloc[-1]
    score = 0
    reasons = []

    if price_change > 0.12:
        score += 2
        reasons.append(tr("Significant price increase compared with recent local history."))
    elif price_change > 0.05:
        score += 1
        reasons.append(tr("Moderate price increase observed."))

    if stock_ratio < 0.75:
        score += 2
        reasons.append(tr("Stock level is well below the recent regional baseline."))
    elif stock_ratio < 0.9:
        score += 1
        reasons.append(tr("Stock is slightly below the recent regional norm."))

    if stock_change < -0.18:
        score += 1
        reasons.append(tr("Stock has fallen sharply over the last month."))

    if score >= 4:
        level = "RED"
    elif score >= 2:
        level = "YELLOW"
    else:
        level = "GREEN"

    if not reasons:
        reasons = [tr("No major price or supply stress detected in the recent history.")]

    return level, reasons, round(float(price_change), 4), round(float(stock_ratio), 3), latest


def build_alerts(clean_df):
    regional_rows = []
    for region, group in clean_df.groupby("region"):
        level, reasons, price_change, stock_ratio, latest = risk_level_for_region(group)
        if level != "GREEN":
            regional_rows.append(
                {
                    "region": region,
                    "date": latest["date"].strftime("%Y-%m-%d"),
                    "alert_level": level,
                    "reason": " | ".join(reasons),
                    "suggested_action": (
                        tr("Increase supply monitoring and verify warehouse movements and regional logistics before escalation.")
                        if level == "RED"
                        else tr("Track weekly stock availability and compare with local demand trends in the next 7 days.")
                    ),
                    "price_change_pct": round(float(price_change * 100), 2),
                    "stock_ratio": round(float(stock_ratio), 3),
                }
            )
    return pd.DataFrame(regional_rows, columns=ALERT_COLUMNS)


def detect_anomalies(clean_df):
    anomalies = []
    for region, group in clean_df.groupby("region"):
        group = group.sort_values("date").copy()
        if len(group) < 30:
            continue
        group["price_baseline"] = group["price"].rolling(30, min_periods=7).mean()
        group["stock_baseline"] = group["stock"].rolling(30, min_periods=7).mean()
        group["price_vs_baseline"] = group["price"] / group["price_baseline"] - 1
        group["stock_ratio"] = group["stock"] / group["stock_baseline"]
        group["price_change_7d"] = group["price"].pct_change(7)
        group["stock_drop_pct"] = group["stock"].pct_change()
        features = group[
            ["price_vs_baseline", "stock_ratio", "price_change_7d", "stock_drop_pct"]
        ].replace([float("inf"), -float("inf")], pd.NA).fillna(0)
        detector = IsolationForest(
            n_estimators=200,
            contamination=0.02,
            random_state=42,
            n_jobs=-1,
        )
        group["is_anomaly"] = detector.fit_predict(features)
        group["anomaly_score"] = -detector.decision_function(features)
        recent = group.tail(90)
        for _, row in recent[recent["is_anomaly"] == -1].iterrows():
            anomalies.append(
                {
                    "date": row["date"],
                    "region": region,
                    "price": round(float(row["price"]), 2),
                    "stock": round(float(row["stock"]), 1),
                    "anomaly_score": round(float(row["anomaly_score"]), 4),
                    "stock_drop_pct": round(float(row["stock_drop_pct"] * 100), 2),
                    "label": tr("Anomaly requiring verification"),
                }
            )
    return pd.DataFrame(
        anomalies,
        columns=["date", "region", "price", "stock", "anomaly_score", "stock_drop_pct", "label"],
    ).sort_values("anomaly_score", ascending=False, ignore_index=True)


def render_kpis(clean_df, alert_df, anomalies_df, snapshot_date):
    latest_by_region = clean_df.sort_values("date").groupby("region").tail(1).reset_index(drop=True)
    latest = latest_by_region.loc[latest_by_region["price"].idxmax()]
    current_price = float(latest["price"])
    stock_total = float(latest_by_region["stock"].sum())
    alert_levels = alert_df.get("alert_level", pd.Series(dtype="object"))
    alert_risk = (
        "RED" if alert_levels.eq("RED").any()
        else "YELLOW" if alert_levels.eq("YELLOW").any()
        else "GREEN"
    )

    price_stock_columns = st.columns(2)
    risk_columns = st.columns(2)
    price_stock_columns[0].metric(
        tr("Highest regional price · {region}", region=region_label(latest["region"])),
        f"{current_price:.2f} DZD/kg",
    )
    price_stock_columns[1].metric(tr("Estimated stock · all regions"), f"{stock_total:,.0f} tonnes")
    risk_columns[0].metric(tr("Highest regional risk"), alert_risk)
    risk_columns[1].metric(tr("Patterns to verify · last 90 days"), str(len(anomalies_df)))
    st.caption(tr("Regional metrics are the latest observations on or before {date}.", date=f"{snapshot_date:%d %b %Y}"))


def render_dashboard_charts(snapshot_df):
    focus_regions = ["Alger", "Oran", "Constantine"]
    regions = set(snapshot_df["region"].unique())
    comparison_regions = [region for region in focus_regions if region in regions]
    compare = snapshot_df[snapshot_df["region"].isin(comparison_regions)].copy()
    compare["region_display"] = compare["region"].map(region_label)
    fig_price = px.line(
        compare,
        x="date",
        y="price",
        color="region_display",
        title=tr("Potato price evolution · key demo regions"),
        markers=True,
        template="plotly_white",
        labels={"region_display": tr("region"), "date": tr("date"), "price": tr("Price")},
    )
    style_chart(fig_price)
    fig_stock = px.line(
        compare,
        x="date",
        y="stock",
        color="region_display",
        title=tr("Available stock · key demo regions"),
        markers=True,
        template="plotly_white",
        labels={"region_display": tr("region"), "date": tr("date"), "stock": tr("Stock")},
    )
    style_chart(fig_stock)
    charts_left, charts_right = st.columns(2)
    charts_left.plotly_chart(fig_price, use_container_width=True)
    charts_right.plotly_chart(fig_stock, use_container_width=True)


def render_stock_monitoring(snapshot_df):
    region_summary = snapshot_df.sort_values("date").groupby("region").tail(1).sort_values("stock")
    region_summary["region_label"] = region_summary["region"].map(region_label)
    fig = px.bar(
        region_summary,
        x="region_label",
        y="stock",
        color="stock",
        title=tr("Latest regional stock comparison"),
        template="plotly_white",
        labels={"region_label": tr("region"), "stock": tr("Stock")},
    )
    style_chart(fig)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(
        region_summary[["region_label", "date", "stock", "data_source"]].rename(
            columns={
                "region_label": tr("region"),
                "date": tr("Date"),
                "stock": tr("Stock"),
                "data_source": tr("data_source"),
            }
        ),
        use_container_width=True,
        hide_index=True,
    )
    render_manual_inputs(snapshot_df)


def render_regional_market(snapshot_df):
    region_summary = snapshot_df.sort_values("date").groupby("region").tail(1).sort_values("price")
    region_summary["region_label"] = region_summary["region"].map(region_label)
    fig_region = px.bar(
        region_summary,
        x="region_label",
        y="price",
        color="region_label",
        title=tr("Latest regional price comparison"),
        template="plotly_white",
        labels={"region_label": tr("region"), "price": tr("Price")},
    )
    style_chart(fig_region)
    st.plotly_chart(fig_region, use_container_width=True)

    map_data = region_summary.copy()
    map_data["region_label"] = map_data["region"].map(region_label)
    map_data["latitude"] = map_data["region"].map(lambda name: REGION_COORDINATES[name][0])
    map_data["longitude"] = map_data["region"].map(lambda name: REGION_COORDINATES[name][1])
    map_figure = px.scatter_geo(
        map_data,
        lat="latitude",
        lon="longitude",
        size="stock",
        color="price",
        hover_name="region_label",
        hover_data={"price": ":.2f", "stock": ":,.1f", "latitude": False, "longitude": False},
        color_continuous_scale="RdYlGn_r",
        size_max=24,
        scope="africa",
        projection="mercator",
        title=tr("Regional market map · approximate city locations"),
        labels={"price": tr("Price"), "stock": tr("Stock")},
    )
    style_chart(map_figure)
    map_figure.update_geos(
        lonaxis_range=[-9, 10],
        lataxis_range=[18, 38],
        showland=True,
        showcountries=True,
        landcolor="#f4f4f4",
        countrycolor="#c8c8c8",
    )
    map_figure.update_layout(margin={"r": 0, "t": 48, "l": 0, "b": 0})
    st.plotly_chart(map_figure, use_container_width=True)
    st.caption(tr("Approximate city coordinates only; markers are not wilaya boundaries."))


def render_price_prediction(predictions, latest_df):
    regions = sorted(latest_df["region"].unique())
    region = st.selectbox(
        tr("Focus region"),
        regions,
        index=regions.index("Alger") if "Alger" in regions else 0,
        format_func=region_label,
    )
    horizon = st.selectbox(
        tr("Forecast horizon"),
        dp.HORIZONS,
        format_func=lambda days: tr("{days}-day forecast", days=days),
    )
    pred_plot = predictions[horizon][predictions[horizon]["region"] == region].copy()
    pred_plot["target_date"] = pd.to_datetime(pred_plot["target_date"])
    region_history = latest_df[latest_df["region"] == region].sort_values("date")
    forecast_price_label = tr("Price (DZD/kg)")
    forecast_date_label = tr("Date")
    pred_fig = px.line(
        pd.concat([
            region_history[["date", "price"]].rename(
                columns={"date": forecast_date_label, "price": forecast_price_label}
            ),
            pred_plot[["target_date", "predicted_price"]].rename(
                columns={"target_date": forecast_date_label, "predicted_price": forecast_price_label}
            )
        ], ignore_index=True),
        x=forecast_date_label,
        y=forecast_price_label,
        title=tr("{region} · {days}-day forecast from latest dataset date", region=region_label(region), days=horizon),
        markers=True,
        template="plotly_white",
        labels={forecast_date_label: forecast_date_label},
    )
    style_chart(pred_fig)
    st.caption(tr("Forecast uses the latest available dataset date, independently of the selected historical snapshot."))
    st.plotly_chart(pred_fig, use_container_width=True)


def render_shortage_detection(alert_df):
    st.subheader(tr("Shortage risk overview"))
    if alert_df.empty:
        st.info(tr("No regional shortage alerts were generated for this snapshot."))
    else:
        for _, row in alert_df.iterrows():
            color = {"RED": "#d9534f", "YELLOW": "#f0ad4e", "GREEN": "#5cb85c"}[row["alert_level"]]
            st.markdown(
                f"<div style='padding: 10px; border-left: 6px solid {color}; background: #f7f7f7; margin-bottom: 10px;'>"
                f"<b>{row['alert_level']}</b> · {region_label(row['region'])} · {row['date']}<br>"
                f"{row['reason']}<br>"
                f"<i>{tr('Suggested action:')}</i> {row['suggested_action']}</div>",
                unsafe_allow_html=True,
            )


def render_anomaly_detection(anomalies_df):
    st.subheader(tr("Detected anomalies requiring verification"))
    st.info(tr(
        "Supply-chain discrepancy checks need verified production, warehouse-release, and wholesale-market volume data, which this demo does not contain."
    ))
    if anomalies_df.empty:
        st.success(tr("No unusual patterns detected in the current data window."))
    else:
        st.dataframe(anomalies_df.rename(columns={column: tr(column) for column in anomalies_df.columns}), use_container_width=True)


def render_smart_alerts(alert_df):
    st.subheader(tr("Smart Alerts"))
    st.caption(tr("Alerts are rule-based signals generated from the selected synthetic-data snapshot."))
    if alert_df.empty:
        st.info(tr("No regional shortage alerts were generated for this snapshot."))
        return
    display = alert_df.copy()
    display["region"] = display["region"].map(region_label)
    st.dataframe(
        display.rename(columns={column: tr(column) for column in display.columns}),
        use_container_width=True,
        hide_index=True,
    )


def render_reports(clean_df, alert_df):
    st.subheader(tr("Reports"))
    latest_date = clean_df["date"].max()
    st.caption(
        tr("Data source: {source} · last available observation: {date}",
           source=clean_df["data_source"].mode().iat[0],
           date=latest_date.strftime("%Y-%m-%d"))
    )
    st.warning(tr("Exported values retain their dataset source labels; synthetic data is not verified market data."))
    st.download_button(
        tr("Download market dataset (CSV)"),
        data=reporting.to_csv_bytes(clean_df),
        file_name="agri_smart_market_data.csv",
        mime="text/csv",
    )
    if not alert_df.empty:
        st.download_button(
            tr("Download current alerts (CSV)"),
            data=reporting.to_csv_bytes(alert_df),
            file_name="agri_smart_alerts.csv",
            mime="text/csv",
        )


def render_about(clean_df):
    st.subheader(tr("About AGRI-SMART DZ"))
    st.write(tr("About project description"))
    st.write(tr("About model description"))
    st.write(tr("About limitations"))
    st.caption(
        tr("Data source: {source} · last available observation: {date}",
           source=clean_df["data_source"].mode().iat[0],
           date=clean_df["date"].max().strftime("%Y-%m-%d"))
    )


def render_scenario_simulator(clean_df, predictions):
    st.subheader(tr("What-if supply scenario"))
    regions = sorted(clean_df["region"].unique())
    region = st.selectbox(
        tr("Simulation region"),
        regions,
        index=regions.index("Alger") if "Alger" in regions else 0,
        format_func=region_label,
    )
    region_latest = clean_df[clean_df["region"] == region].sort_values("date").iloc[-1]
    current_stock = float(region_latest["stock"])
    stock_adjustment = st.slider(
        tr("Change available stock (tonnes)"),
        min_value=-float(int(current_stock)),
        max_value=5000.0,
        value=0.0,
        step=1.0,
    )
    st.caption(tr(
        "Baseline date: {date} · Current synthetic stock: {stock} tonnes.",
        date=f"{region_latest['date']:%d %b %Y}",
        stock=f"{current_stock:,.1f}",
    ))
    st.warning(tr("Illustrative model sensitivity only. Synthetic data does not establish a causal or operational guarantee."))

    columns = st.columns(len(dp.HORIZONS))
    for column, horizon in zip(columns, dp.HORIZONS):
        baseline = predictions[horizon].loc[predictions[horizon]["region"] == region].iloc[0]
        if stock_adjustment == 0:
            scenario_row = baseline
        else:
            scenario, _ = predict_latest(
                horizon,
                clean_df,
                region=region,
                stock_adjustment_tonnes=stock_adjustment,
            )
            scenario_row = scenario.iloc[0]
        delta = float(scenario_row["predicted_price"] - baseline["predicted_price"])
        column.metric(
            tr("{days}-day scenario price", days=horizon),
            f"{scenario_row['predicted_price']:.2f} DZD/kg",
            delta=tr("{delta} vs baseline", delta=f"{delta:+.2f}"),
        )
        column.caption(tr(
            "Baseline {price} · 90% interval {low}–{high} DZD/kg",
            price=f"{baseline['predicted_price']:.2f}",
            low=f"{scenario_row['low_90']:.2f}",
            high=f"{scenario_row['high_90']:.2f}",
        )
        )


def render_manual_inputs(clean_df):
    st.subheader(tr("Manual stock and price entry"))
    with st.form("manual_entry"):
        c1, c2, c3, c4 = st.columns(4)
        region = c1.selectbox(tr("Region"), sorted(clean_df["region"].unique()), format_func=region_label)
        value_date = c2.date_input(tr("Date"))
        price = c3.number_input(tr("Price (DZD/kg)"), min_value=0.1, step=1.0, value=55.0)
        stock = c4.number_input(tr("Stock (tonnes)"), min_value=0.0, step=10.0, value=400.0)
        submitted = st.form_submit_button(tr("Save manual data"))
        if submitted:
            db.add_manual_observation("Potato", region, value_date, price, stock)
            load_dashboard_data.clear()
            get_model_predictions.clear()
            st.success(tr("Manual price and stock data saved to the database."))
            st.rerun()


NEW_NAVIGATION = (
    ("market_dashboard", "Dashboard"),
    ("algeria_map", "Regional map"),
    ("multi_forecast", "Price forecast"),
    ("early_warnings", "Early warnings"),
    ("data_management", "Market data management"),
    ("scenario_simulation", "Hypothetical scenario simulator"),
    ("market_reports", "Report and export"),
    ("potato_classic", "Potato demo (classic)"),
)


@st.cache_data(ttl=60)
def load_market_observations():
    return db.get_market_observations()


def _localized_region_name(row, language):
    column = {"ar": "name_ar", "en": "name_en", "fr": "name_fr"}.get(language, "name_fr")
    return row.get(column) or row.get("name_fr") or row.get("wilaya_code")


def _region_options(observations, wilayas, language):
    names = {str(value).strip() for value in observations.get("region", pd.Series(dtype=str)).dropna()}
    if "region" in observations:
        names.update(observations["region"].dropna().astype(str).str.strip())
    return sorted(names | {"All regions"})


def _normalize_wilaya_code(value):
    try:
        return str(int(str(value).strip())).zfill(2)
    except (TypeError, ValueError):
        return None


def _map_wilaya_codes(observations, wilayas):
    if observations.empty:
        return observations.assign(map_code=pd.Series(dtype=str))
    name_to_code = {}
    for row in wilayas.to_dict(orient="records"):
        for field in ("name_fr", "name_ar", "name_en"):
            name = row.get(field)
            if name:
                name_to_code[str(name).strip().casefold()] = str(row["wilaya_code"]).zfill(2)
    frame = observations.copy()
    frame["map_code"] = frame.apply(
        lambda row: _normalize_wilaya_code(row.get("wilaya_code"))
        or name_to_code.get(str(row.get("region", "")).strip().casefold()),
        axis=1,
    )
    return frame


def _latest_by_region(data):
    if data.empty:
        return data.copy()
    frame = data.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    return frame.groupby("region", as_index=False, sort=False).tail(1).reset_index(drop=True)


def _optional_nonnegative_measure(value, label):
    value = str(value).strip()
    if not value:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be a non-negative number.") from error
    if pd.isna(number) or abs(number) == float("inf") or number < 0:
        raise ValueError(f"{label} must be a finite, non-negative number.")
    return number


def _risk_for_map(risks, region):
    matching = risks[risks["region"].astype(str) == str(region)] if not risks.empty else risks
    if matching.empty:
        return "Normal availability"
    level = str(matching.iloc[0]["risk_level"]).upper()
    return {
        "YELLOW": "Under observation",
        "ORANGE": "High risk",
        "RED": "Critical risk",
    }.get(level, "Under observation")


def _selected_observations(all_data, product, category, region, date_range, quality):
    frame = all_data.copy()
    if product != "All":
        frame = frame[frame["product"] == product]
    if category != "All":
        frame = frame[frame["category"] == category]
    if region != "All regions":
        frame = frame[frame["region"].astype(str) == region]
    if quality != "All":
        frame = frame[frame["data_quality"] == quality]
    if date_range and len(date_range) == 2:
        dates = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame[dates.dt.date.between(date_range[0], date_range[1])]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def _render_dashboard(data):
    st.subheader(tr("Market Intelligence"))
    if data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    latest = _latest_by_region(data)
    latest_prices = latest["price"].dropna() if "price" in latest else pd.Series(dtype=float)
    risks = market_intelligence.detect_shortage_risks(data)
    products_count = data["product"].nunique()
    wilayas_count = data.loc[data["wilaya_code"].notna(), "wilaya_code"].nunique()
    metrics = st.columns(4)
    metrics[0].metric(tr("Products with observations"), products_count)
    metrics[1].metric(tr("Wilayas with observations"), wilayas_count)
    metrics[2].metric(
        tr("Latest average price"),
        (
            f"{latest_prices.mean():,.2f} DZD/{latest['unit'].dropna().iloc[0]}"
            if "unit" in latest and latest["unit"].nunique(dropna=True) == 1
            else f"{latest_prices.mean():,.2f} DZD"
        )
        if not latest_prices.empty and products_count == 1 and ("unit" not in latest or latest["unit"].nunique(dropna=True) <= 1)
        else "—",
    )
    metrics[3].metric(tr("Products with alerts"), risks["product"].nunique() if not risks.empty else 0)
    if products_count > 1:
        st.caption(tr("Select one product to see an average price; averages across different products are not meaningful."))

    chart_data = data[data["price"].notna()]
    if not chart_data.empty:
        left, right = st.columns(2)
        with left:
            st.markdown(f"#### {tr('Price trend')}")
            trend = chart_data.groupby(["date", "product", "unit"], as_index=False, dropna=False)["price"].median()
            trend["series"] = trend["product"].astype(str) + " (" + trend["unit"].fillna(tr("Unit")).astype(str) + ")"
            trend_figure = px.line(
                trend,
                x="date",
                y="price",
                color="series",
                facet_col="series" if trend["series"].nunique() > 1 else None,
                facet_col_wrap=2 if trend["series"].nunique() > 1 else None,
                labels={"date": tr("Date"), "price": tr("Price per unit"), "series": tr("Product") + " · " + tr("Unit")},
            )
            if trend["series"].nunique() > 1:
                trend_figure.update_yaxes(matches=None)
            st.plotly_chart(
                style_chart(trend_figure),
                use_container_width=True,
            )
        with right:
            st.markdown(f"#### {tr('Regional latest prices')}")
            regional_products = sorted(chart_data["product"].dropna().astype(str).unique())
            if regional_products:
                regional_product = st.selectbox(
                    tr("Product"),
                    regional_products,
                    key="dashboard_regional_product",
                )
                regional_data = chart_data[chart_data["product"] == regional_product]
                regional_units = sorted(regional_data["unit"].dropna().astype(str).unique())
                if len(regional_units) > 1:
                    regional_unit = st.selectbox(tr("Unit"), regional_units, key="dashboard_regional_unit")
                    regional_data = regional_data[regional_data["unit"] == regional_unit]
                regional = _latest_by_region(regional_data).dropna(subset=["price"])
                if not regional.empty:
                    regional_figure = px.bar(
                        regional.sort_values("price"),
                        x="price",
                        y="region",
                        orientation="h",
                        color="price",
                        labels={"price": tr("Price per unit"), "region": tr("Region")},
                    )
                    st.plotly_chart(style_chart(regional_figure), use_container_width=True)
                else:
                    st.info(tr("No price data available."))
    left, right = st.columns(2)
    with left:
        st.markdown(f"#### {tr('Products with the highest price increases')}")
        increase_rows = []
        for (product, region, unit), group in data.dropna(subset=["price"]).groupby(
            ["product", "region", "unit"], dropna=False
        ):
            group = group.sort_values("date")
            latest_row = group.iloc[-1]
            baseline = group.loc[group["date"] < latest_row["date"], "price"].tail(30)
            if not baseline.empty and baseline.median() > 0:
                increase_rows.append({
                    "product": product,
                    "region": region,
                    "unit": unit,
                    "price_change_pct": round((latest_row["price"] / baseline.median() - 1) * 100, 1),
                    "date": latest_row["date"].date(),
                })
        if increase_rows:
            increases = pd.DataFrame(increase_rows).nlargest(5, "price_change_pct")
            st.dataframe(increases.rename(columns={"product": tr("Product"), "region": tr("Region"), "unit": tr("Unit"), "price_change_pct": tr("Price change (%)"), "date": tr("Date")}), hide_index=True, use_container_width=True)
        else:
            st.info(tr("No price data available."))
    with right:
        st.markdown(f"#### {tr('Latest alerts')}")
        if risks.empty:
            st.success(tr("Normal availability"))
        else:
            st.dataframe(
                risks[["product", "region", "unit", "date", "risk_level", "status", "reason"]]
                .rename(columns={"product": tr("Product"), "region": tr("Region"), "unit": tr("Unit"), "date": tr("Date"), "risk_level": tr("Risk level"), "status": tr("Status"), "reason": tr("Detection explanation")}),
                hide_index=True,
                use_container_width=True,
            )
    if "quantity" in data and data["quantity"].notna().any():
        stock = _latest_by_region(data[data["quantity"].notna()])
        st.markdown(f"#### {tr('Latest stock availability')}")
        st.dataframe(stock[["product", "region", "quantity", "incoming_supply", "date", "data_quality"]].rename(columns={"product": tr("Product"), "region": tr("Region"), "quantity": tr("Available stock (optional)"), "incoming_supply": tr("Incoming supply (optional)"), "date": tr("Date"), "data_quality": tr("Data quality")}), hide_index=True, use_container_width=True)


def _render_algeria_map(data, all_data, wilayas):
    st.subheader(tr("Wilaya boundaries (GeoJSON)"))
    st.warning(tr("Boundary-source and license metadata are not documented in this repository; verify before operational use."))
    st.caption(tr("The map only colors wilayas with matching observed records. Missing areas remain explicitly marked as no data."))
    products = sorted(all_data["product"].dropna().unique()) if not all_data.empty else []
    if not products:
        st.info(tr("No observations match the selected filters."))
        return
    current_product = st.selectbox(tr("Product"), products, key="map_product")
    product_data = data[data["product"] == current_product]
    if product_data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    map_units = sorted(product_data["unit"].dropna().astype(str).unique())
    if len(map_units) > 1:
        selected_unit = st.selectbox(tr("Unit"), map_units, key="map_unit")
        product_data = product_data[product_data["unit"] == selected_unit]
    risks = market_intelligence.detect_shortage_risks(product_data)
    latest = _map_wilaya_codes(_latest_by_region(product_data), wilayas)
    latest = latest.dropna(subset=["map_code"])
    if latest.empty:
        st.warning(tr("No wilaya codes could be matched to the observed region names; map locations are not inferred."))
        return

    geo_path = ROOT / "data" / "algeria_wilayas_58.geojson"
    with geo_path.open(encoding="utf-8") as geo_file:
        geojson = json.load(geo_file)
    records = latest.to_dict(orient="records")
    by_code = {row["map_code"]: row for row in records}
    map_rows = []
    for feature in geojson.get("features", []):
        properties = feature.get("properties", {})
        code = str(properties.get("CdWilaya_2019", "")).zfill(2)
        row = by_code.get(code)
        risk = _risk_for_map(risks, row.get("region")) if row else "No data"
        map_rows.append({
            "map_code": code,
            "wilaya": properties.get("LbWilaya_2019", code),
            "region": row.get("region") if row else None,
            "price": row.get("price") if row else None,
            "unit": row.get("unit") if row else None,
            "quantity": row.get("quantity") if row else None,
            "incoming_supply": row.get("incoming_supply") if row else None,
            "date": row.get("date") if row else None,
            "source_name": row.get("source_name") if row else None,
            "data_quality": row.get("data_quality") if row else None,
            "risk": risk,
        })
    map_frame = pd.DataFrame(map_rows)
    colors = {
        "No data": "#B9C2BC",
        "Normal availability": "#2E8B57",
        "Under observation": "#E8C547",
        "High risk": "#E58A32",
        "Critical risk": "#B83333",
    }
    figure = px.choropleth(
        map_frame,
        geojson=geojson,
        locations="map_code",
        featureidkey="properties.CdWilaya_2019",
        color="risk",
        color_discrete_map=colors,
        category_orders={"risk": list(colors)},
        hover_name="wilaya",
        hover_data={"map_code": False, "risk": True, "region": True, "price": ":.2f", "quantity": ":.2f", "incoming_supply": ":.2f", "date": True, "source_name": True, "data_quality": True},
        labels={"risk": tr("Risk level"), "region": tr("Region"), "price": tr("Price per unit"), "quantity": tr("Available stock (optional)"), "incoming_supply": tr("Incoming supply (optional)"), "date": tr("Latest observation"), "source_name": tr("Source name"), "data_quality": tr("Data quality")},
    )
    figure.update_geos(fitbounds="locations", visible=False, bgcolor="#F5F7F5")
    figure.update_layout(clickmode="event+select", paper_bgcolor="#F5F7F5", margin={"l": 0, "r": 0, "t": 0, "b": 0}, legend_title_text=tr("Risk level"))
    selection = st.plotly_chart(figure, use_container_width=True, on_select="rerun", selection_mode="points", key="wilaya_choropleth")
    selected_code = None
    try:
        selected_points = selection.selection.points
        if selected_points:
            selected_code = str(selected_points[0].get("location", "")).zfill(2)
    except (AttributeError, KeyError, TypeError):
        selected_code = None
    names = {
        str(row["wilaya_code"]).zfill(2): _localized_region_name(row, st.session_state.get("language", "ar"))
        for row in wilayas.to_dict(orient="records")
    }
    codes = [str(row["wilaya_code"]).zfill(2) for row in wilayas.to_dict(orient="records")]
    if selected_code not in codes:
        selected_code = codes[0] if codes else None
    if not selected_code:
        return
    selected_wilaya = st.selectbox(tr("Map selection"), codes, index=codes.index(selected_code), format_func=lambda code: f"{code} — {names.get(code, code)}", key="wilaya_details")
    detail = map_frame[map_frame["map_code"] == selected_wilaya]
    if detail.empty or detail.iloc[0]["risk"] == "No data":
        st.info(tr("No observations match the selected filters."))
        return
    record = detail.iloc[0]
    st.markdown(f"### {names.get(selected_wilaya, record['wilaya'])}")
    columns = st.columns(4)
    unit = str(record.get("unit") or "")
    price_value = f"{record['price']:,.2f} DZD/{unit}" if pd.notna(record["price"]) and unit else f"{record['price']:,.2f} DZD" if pd.notna(record["price"]) else "—"
    columns[0].metric(tr("Latest average price"), price_value)
    columns[1].metric(tr("Available stock (optional)"), f"{record['quantity']:,.2f}" if pd.notna(record["quantity"]) else "—")
    columns[2].metric(tr("Risk level"), tr(record["risk"]))
    columns[3].metric(tr("Latest observation"), str(record["date"].date()) if pd.notna(record["date"]) else "—")
    st.caption(f"{tr('Source and quality')}: {record['source_name']} · {record['data_quality']}")
    history = product_data[product_data["region"].astype(str) == str(record["region"])].dropna(subset=["price"])
    if not history.empty:
        st.plotly_chart(style_chart(px.line(history, x="date", y="price", labels={"date": tr("Date"), "price": tr("Price per unit")})), use_container_width=True)


def _render_forecast(data, product):
    st.subheader(tr("Price forecast"))
    eligible = sorted(data["product"].dropna().unique()) if not data.empty else []
    if product == "All":
        product = st.selectbox(tr("Product"), eligible or ["Potato"], key="forecast_product")
    product_data = data[data["product"] == product] if not data.empty else pd.DataFrame()
    forecast_unit = ""
    if "unit" in product_data and product_data["unit"].dropna().nunique() > 1:
        units = sorted(product_data["unit"].dropna().astype(str).unique())
        selected_unit = st.selectbox(tr("Unit"), units, key="forecast_unit")
        product_data = product_data[product_data["unit"] == selected_unit]
        forecast_unit = selected_unit
    elif "unit" in product_data and product_data["unit"].notna().any():
        forecast_unit = str(product_data["unit"].dropna().iloc[0])
    horizon = st.radio(tr("Forecast horizon"), [7, 30], horizontal=True, format_func=lambda days: f"{days} days")
    if product_data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    result = market_intelligence.forecast_product(product_data, int(horizon))
    if result["status"] != "ok":
        st.warning(f"{tr('Insufficient data')}: {result['reason']}")
        st.caption(tr("No validated forecast is available for this product and horizon."))
        return
    st.caption(f"{tr('Predicted')} · {product} · {horizon} days · {tr('Validation metrics (chronological holdout)')}")
    st.caption(tr("Forecasts are estimates; observed market prices and test-set results are shown separately."))
    metrics = result["metrics"]
    st.dataframe(pd.DataFrame([{"model": name, **values} for name, values in metrics.items()]).rename(columns={"model": tr("Model"), "MAE": "MAE", "RMSE": "RMSE", "R2": "R²"}), hide_index=True, use_container_width=True)
    predictions = result["predictions"].melt(id_vars=["target_date", "region"], value_vars=["actual_price", "predicted_price"], var_name="series", value_name="price")
    predictions["series"] = predictions["series"].map({"actual_price": tr("Observed"), "predicted_price": tr("Predicted")})
    st.plotly_chart(style_chart(px.line(predictions, x="target_date", y="price", color="series", line_dash="region", labels={"target_date": tr("Date"), "price": tr("Price per unit"), "series": tr("Series"), "region": tr("Region")})), use_container_width=True)
    forecasts = result["forecast"]
    if "unit" not in forecasts:
        forecasts["unit"] = forecast_unit
    st.caption(f"{tr('Uncertainty interval')}: {result['uncertainty']}")
    st.dataframe(forecasts.rename(columns={"region": tr("Region"), "date": tr("Latest observation"), "price": tr("Observed"), "predicted_price": tr("Predicted"), "lower_90": tr("Lower 90% interval"), "upper_90": tr("Upper 90% interval"), "target_date": tr("Forecast date"), "model": tr("Model")}), hide_index=True, use_container_width=True)
    db.save_prediction_records(product, forecasts, metrics, int(horizon))


def _render_early_warnings(data):
    st.subheader(tr("Early warnings"))
    if data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    risks = market_intelligence.detect_shortage_risks(data)
    st.caption(tr("No automated confirmed-shortage determination is made."))
    if risks.empty:
        st.success(tr("Normal availability"))
        return
    db.save_risk_alerts(risks)
    for record in risks.to_dict(orient="records"):
        heading = f"{record['product']} · {record['region']} · {tr(record['risk_level'].title())} · {record['date']}"
        with st.expander(heading):
            st.write(f"**{tr('Status')}:** {tr(record['status'].replace('_', ' ').title())}")
            st.write(f"**{tr('Detection explanation')}:** {record['reason']}")
            indicators = []
            if pd.notna(record["price_change_pct"]):
                indicators.append(f"{tr('Price change (%)')}: {record['price_change_pct']:.1f}%")
            if pd.notna(record["stock_ratio"]):
                indicators.append(f"{tr('Stock ratio')}: {record['stock_ratio']:.2f}×")
            st.write(" · ".join(indicators) or tr("No stock/supply observations are available."))
            st.write(f"**{tr('Recommended action')}:** {record['recommended_action']}")
            st.caption(f"{tr('Data quality')}: {record['data_quality']}")
    st.dataframe(risks.rename(columns={"product": tr("Product"), "region": tr("Region"), "unit": tr("Unit"), "date": tr("Date"), "risk_level": tr("Risk level"), "status": tr("Status"), "reason": tr("Detection explanation"), "recommended_action": tr("Recommended action"), "data_quality": tr("Data quality")}), hide_index=True, use_container_width=True)


def _render_data_management(products, wilayas):
    st.subheader(tr("Market data management"))
    if db.get_database_backend() == "sqlite":
        st.warning(tr("SQLite is suitable for local demonstration, but a local file may not persist on hosted Streamlit. Configure DATABASE_URL with PostgreSQL for durable hosted storage."))
    if not wilayas.empty:
        st.caption(f"{tr('Wilayas with observations')}: {len(wilayas)}")
    with st.expander(tr("Add product")):
        with st.form("add_product_form"):
            c1, c2, c3 = st.columns(3)
            name = c1.text_input(tr("Product name"), max_chars=120)
            categories = sorted(set(products["category_name"].dropna().astype(str)) | {"Other"}) if not products.empty else ["Other"]
            category = c2.selectbox(tr("Product category"), categories)
            unit = c3.text_input(tr("Unit"), value="kg", max_chars=40)
            add_product = st.form_submit_button(tr("Add product"))
        if add_product:
            try:
                if not name.strip():
                    raise ValueError("Product name is required.")
                db.add_product(name, category, unit)
                load_market_observations.clear()
                st.success(tr("Product added."))
                st.rerun()
            except ValueError as error:
                st.error(str(error))
    with st.form("manual_observation_form"):
        st.markdown(f"#### {tr('Add observation')}")
        c1, c2 = st.columns(2)
        product_names = sorted(products["product_name"].dropna().astype(str).unique()) if not products.empty else ["Potato"]
        product = c1.selectbox(tr("Product"), product_names)
        region_rows = wilayas.to_dict(orient="records")
        language = st.session_state.get("language", "ar")
        code_to_name = {str(row["wilaya_code"]).zfill(2): _localized_region_name(row, language) for row in region_rows}
        codes = sorted(code_to_name)
        if codes:
            code = c2.selectbox(tr("Region"), codes, format_func=lambda value: f"{value} — {code_to_name[value]}")
            region = code_to_name[code]
        else:
            code, region = None, c2.text_input(tr("Region"))
        c3, c4 = st.columns(2)
        observation_date = c3.date_input(tr("Observation date"))
        market = c4.text_input(tr("Market name (optional)"))
        c5, c6, c7 = st.columns(3)
        price = c5.text_input(tr("Price (DZD per unit)"), value="")
        quantity = c6.text_input(tr("Available stock (optional)"), value="")
        incoming = c7.text_input(tr("Incoming supply (optional)"), value="")
        c8, c9 = st.columns(2)
        source = c8.text_input(tr("Source name"), value="Manual entry — unverified")
        quality = c9.selectbox(tr("Data quality"), ["UNVERIFIED", "VERIFIED"])
        submitted = st.form_submit_button(tr("Save observation"))
    if submitted:
        try:
            if not region:
                raise ValueError("Enter a region.")
            price_value = _optional_nonnegative_measure(price, "Price")
            quantity_value = _optional_nonnegative_measure(quantity, "Available stock")
            incoming_value = _optional_nonnegative_measure(incoming, "Incoming supply")
            if all(value is None for value in (price_value, quantity_value, incoming_value)):
                raise ValueError("Enter at least one measurement.")
            db.add_manual_observation(
                product, region, observation_date,
                price=price_value,
                quantity=quantity_value,
                incoming_supply=incoming_value,
                market=market, source_name=source.strip() or "Manual entry",
                data_quality=quality, wilaya_code=code,
            )
            load_market_observations.clear()
            st.success(tr("Observation saved."))
            st.rerun()
        except ValueError as error:
            st.error(str(error))

    st.markdown(f"#### {tr('Import CSV / Excel')}")
    uploaded = st.file_uploader(tr("Upload a CSV or Excel file"), type=["csv", "xlsx", "xls"])
    if uploaded is not None:
        try:
            frame = data_import.read_uploaded_file(uploaded.name, uploaded.getvalue())
            valid, issues, duplicates = data_import.normalize_frame(frame)
            st.caption(f"{tr('Valid rows')}: {len(valid)} · {tr('Duplicate rows within file: {count}', count=duplicates)}")
            if valid:
                st.dataframe(pd.DataFrame(valid).head(20), hide_index=True, use_container_width=True)
            for issue in issues[:20]:
                st.warning(issue)
            if st.button(tr("Import observations"), disabled=not valid, key="import_observations"):
                result = data_import.import_frame(frame)
                load_market_observations.clear()
                st.success(tr("Import complete: {inserted} added, {updated} updated, {skipped} skipped.", **result))
                if result["issues"]:
                    st.warning("\n".join(result["issues"][:20]))
                st.rerun()
        except (ValueError, ImportError) as error:
            st.error(str(error))

    st.markdown(f"#### {tr('Data sources')}")
    sources = db.list_data_sources()
    st.dataframe(sources, hide_index=True, use_container_width=True)
    if not products.empty:
        st.markdown(f"#### {tr('Product catalog')}")
        st.dataframe(products.rename(columns={"product_name": tr("Product"), "category_name": tr("Category"), "unit": tr("Unit"), "is_active": tr("Active")}), hide_index=True, use_container_width=True)


def _render_scenario(data, product, region):
    st.subheader(tr("Hypothetical scenario simulator"))
    product_names = sorted(data["product"].dropna().unique()) if not data.empty else []
    if product == "All":
        product = st.selectbox(tr("Product"), product_names or ["Potato"], key="scenario_product")
    product_data = data[data["product"] == product] if not data.empty else pd.DataFrame()
    if product_data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    if region == "All regions":
        available_regions = sorted(product_data["region"].dropna().astype(str).unique())
        region = st.selectbox(tr("Region"), available_regions, key="scenario_region") if available_regions else None
    region_data = product_data[product_data["region"].astype(str) == str(region)]
    latest = _latest_by_region(region_data)
    if latest.empty:
        st.info(tr("No observations match the selected filters."))
        return
    baseline = latest.iloc[0]
    if pd.isna(baseline["price"]) and pd.isna(baseline["quantity"]):
        st.info(tr("No price data available."))
        return
    st.caption(f"{tr('Latest observation')}: {baseline['date'].date()} · {baseline['data_quality']}")
    with st.form("scenario_form"):
        c1, c2, c3 = st.columns(3)
        production = c1.number_input(tr("Production change (%)"), min_value=-100.0, max_value=500.0, value=0.0, step=5.0)
        demand = c2.number_input(tr("Demand change (%)"), min_value=-100.0, max_value=500.0, value=0.0, step=5.0)
        delay = c3.number_input(tr("Distribution delay (days)"), min_value=0, max_value=365, value=0, step=1)
        c4, c5 = st.columns(2)
        stock_change = c4.number_input(tr("Stock change (%)"), min_value=-100.0, max_value=500.0, value=0.0, step=5.0)
        transport = c5.number_input(tr("Transportation cost change (%)"), min_value=-100.0, max_value=500.0, value=0.0, step=5.0)
        submitted = st.form_submit_button(tr("Run hypothetical scenario"))
    st.info(tr("Hypothetical sensitivity estimate only — not a prediction or causal claim."))
    if submitted:
        try:
            result = market_intelligence.simulate_scenario(
                None if pd.isna(baseline["price"]) else float(baseline["price"]),
                None if pd.isna(baseline["quantity"]) else float(baseline["quantity"]),
                production, demand, int(delay), stock_change, transport,
            )
            db.save_scenario_run("MARKET_SENSITIVITY", result["assumptions"], result)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(tr("Hypothetical price"), f"{result['scenario_price']:,.2f} DZD" if result["scenario_price"] is not None else "—", f"{result['price_index_pct']:+.1f}%")
            c2.metric(tr("Hypothetical supply index change"), f"{result['scenario_stock_index_pct']:+.1f}%")
            c3.metric(tr("Hypothetical risk"), tr(result["hypothetical_risk"]))
            c4.metric(tr("Latest observation"), str(baseline["date"].date()))
            st.json(result["assumptions"])
        except ValueError as error:
            st.error(str(error))


def _render_reports(all_data):
    st.subheader(tr("Report and export"))
    if all_data.empty:
        st.info(tr("No observations match the selected filters."))
        return
    last_observation = pd.to_datetime(all_data["date"]).max().date()
    period = st.selectbox(tr("Reporting period"), ["Weekly", "Monthly", "Custom"])
    default_start = last_observation - pd.Timedelta(days=6 if period == "Weekly" else 29 if period == "Monthly" else 30)
    default_start = max(default_start, pd.to_datetime(all_data["date"]).min().date())
    if period == "Custom":
        c1, c2 = st.columns(2)
        start = c1.date_input(tr("Start date"), value=default_start)
        end = c2.date_input(tr("End date"), value=last_observation)
    else:
        start, end = default_start, last_observation
        st.caption(f"{tr('Start date')}: {start} · {tr('End date')}: {end}")
    if start > end:
        st.error("Start date must be on or before end date.")
        return
    category_options = ["All"] + sorted(all_data["category"].dropna().astype(str).unique())
    report_category = st.selectbox(tr("Category"), category_options, key="report_category")
    candidates = all_data if report_category == "All" else all_data[all_data["category"] == report_category]
    products = st.multiselect(tr("Products to compare"), sorted(candidates["product"].dropna().astype(str).unique()), default=sorted(candidates["product"].dropna().astype(str).unique())[:3])
    regions = st.multiselect(tr("Regions to compare"), sorted(candidates["region"].dropna().astype(str).unique()), default=sorted(candidates["region"].dropna().astype(str).unique())[:6])
    report_data = reporting.filter_period(candidates, start, end)
    if products:
        report_data = report_data[report_data["product"].isin(products)]
    if regions:
        report_data = report_data[report_data["region"].isin(regions)]
    alerts = market_intelligence.detect_shortage_risks(report_data)
    st.caption(f"{len(report_data)} {tr('observations')} · {tr('Data source and observation dates are included in downloads.')}")
    if not report_data.empty and report_data["price"].notna().any():
        for dimension in ("product", "region"):
            comparison = reporting.comparison_series(report_data, dimension)
            if comparison[dimension].nunique() > 1:
                st.markdown(f"#### {tr('Comparison by product' if dimension == 'product' else 'Comparison by region')}")
                st.plotly_chart(
                    style_chart(
                        px.line(
                            comparison,
                            x="date",
                            y="price",
                            color=dimension,
                            facet_col="unit" if comparison["unit"].nunique(dropna=False) > 1 else None,
                            labels={
                                "date": tr("Date"),
                                "price": tr("Price per unit"),
                                dimension: tr("Product" if dimension == "product" else "Region"),
                                "unit": tr("Unit"),
                            },
                        )
                    ),
                    use_container_width=True,
                )
    st.dataframe(report_data.head(500), hide_index=True, use_container_width=True)
    csv = reporting.to_csv_bytes(report_data)
    st.download_button(tr("Download CSV"), csv, file_name="agri-smart-market-report.csv", mime="text/csv")
    excel = reporting.to_excel_bytes([("Market observations", report_data), ("Risk alerts", alerts)])
    st.download_button(tr("Download Excel"), excel, file_name="agri-smart-market-report.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    st.download_button(tr("Download PDF report"), reporting.to_pdf_bytes("AGRI-SMART DZ market report", report_data, alerts, start, end), file_name="agri-smart-market-report.pdf", mime="application/pdf")
    st.download_button(tr("Download alerts CSV"), reporting.to_csv_bytes(alerts), file_name="agri-smart-alerts.csv", mime="text/csv", disabled=alerts.empty)


def _render_classic_potato():
    st.subheader(tr("Potato demo (classic)"))
    clean = load_dashboard_data()
    options = dict(NAVIGATION_SECTIONS)
    section = st.selectbox(tr("Navigation"), list(options), format_func=lambda key: tr(options[key]), key="classic_potato_section")
    snapshot, is_scenario = select_snapshot(clean)
    alerts = build_alerts(snapshot)
    anomalies = detect_anomalies(snapshot) if section in {"dashboard", "anomaly_detection"} else pd.DataFrame()
    snapshot_date = snapshot["date"].max()
    if is_scenario and snapshot_date >= SCENARIO_DATE:
        st.info(tr("Demo scenario: July 2026 market pressure — price pressure in Alger, stock decline in Oran, and combined pressure in Constantine."))
    if section == "dashboard":
        render_kpis(snapshot, alerts, anomalies, snapshot_date)
        render_dashboard_charts(snapshot)
    elif section == "price_prediction":
        render_price_prediction(get_model_predictions(clean), clean)
    elif section == "stock_monitoring":
        render_stock_monitoring(snapshot)
    elif section in {"shortage_detection", "smart_alerts"}:
        render_shortage_detection(alerts)
    elif section == "anomaly_detection":
        render_anomaly_detection(anomalies)
    elif section == "regional_analysis":
        render_regional_market(snapshot)
    elif section == "what_if":
        render_scenario_simulator(clean, get_model_predictions(clean))
    elif section == "reports":
        render_reports(clean, alerts)
    elif section == "about":
        render_about(clean)


def main():
    select_language()
    apply_visual_theme()
    ensure_demo_data()
    all_data = load_market_observations()
    products = db.list_products()
    wilayas = db.list_wilayas()
    st.title("AGRI-SMART DZ")
    st.caption(tr("AGRI-SMART DZ · Agricultural Market Intelligence for Algeria"))

    with st.sidebar:
        st.header(tr("Navigation"))
        labels = dict(NEW_NAVIGATION)
        page = st.radio(
            tr("Navigation"),
            list(labels),
            format_func=lambda key: tr(labels[key]),
            label_visibility="collapsed",
        )
        st.divider()
        st.header(tr("Filters"))
        categories = ["All"] + sorted(products["category_name"].dropna().astype(str).unique())
        category = st.selectbox(tr("Category"), categories)
        product_rows = products if category == "All" else products[products["category_name"] == category]
        product_options = ["All"] + sorted(product_rows["product_name"].dropna().astype(str).unique())
        product = st.selectbox(tr("Product"), product_options)
        quality_options = ["All", *db.QUALITY_LEVELS]
        quality = st.selectbox(tr("Data quality"), quality_options)
        region_options = ["All regions"] + sorted(all_data["region"].dropna().astype(str).unique()) if not all_data.empty else ["All regions"]
        region = st.selectbox(tr("Region"), region_options)
        if not all_data.empty:
            min_date = pd.to_datetime(all_data["date"]).min().date()
            max_date = pd.to_datetime(all_data["date"]).max().date()
            default_start = max(min_date, max_date - pd.Timedelta(days=365))
            date_range = st.date_input(tr("Date range"), value=(default_start, max_date), min_value=min_date, max_value=max_date)
        else:
            date_range = ()
    data = _selected_observations(all_data, product, category, region, date_range, quality)
    if not data.empty and (data["data_quality"] == "SYNTHETIC_DEMO").any():
        st.warning(tr("Synthetic Demonstration Data — Not Verified Market Data."))
    elif not data.empty:
        st.info(tr("Observed market data"))
    if data.empty and page != "data_management":
        st.info(tr("No observations match the selected filters."))

    if page == "market_dashboard":
        _render_dashboard(data)
    elif page == "algeria_map":
        _render_algeria_map(data, all_data, wilayas)
    elif page == "multi_forecast":
        _render_forecast(data, product)
    elif page == "early_warnings":
        _render_early_warnings(data)
    elif page == "data_management":
        _render_data_management(products, wilayas)
    elif page == "scenario_simulation":
        _render_scenario(data, product, region)
    elif page == "market_reports":
        _render_reports(all_data)
    elif page == "potato_classic":
        _render_classic_potato()


if __name__ == "__main__":
    main()
