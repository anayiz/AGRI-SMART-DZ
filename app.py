from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.ensemble import IsolationForest

from src import data_processing as dp
from src import database as db
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
        ["العربية", "Français"],
        index=0,
    )
    language = "ar" if selection == "العربية" else "fr"
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
    ensure_demo_data()
    db.init_db()
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
        data=clean_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="agri_smart_market_data.csv",
        mime="text/csv",
    )
    if not alert_df.empty:
        st.download_button(
            tr("Download current alerts (CSV)"),
            data=alert_df.to_csv(index=False).encode("utf-8-sig"),
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


def main():
    select_language()
    apply_visual_theme()
    section = select_section()
    ensure_demo_data()
    clean_df = load_dashboard_data()
    snapshot_df, is_scenario = select_snapshot(clean_df)
    alert_df = build_alerts(snapshot_df)
    anomalies_df = (
        detect_anomalies(snapshot_df)
        if section in {"dashboard", "anomaly_detection"}
        else pd.DataFrame()
    )
    snapshot_date = snapshot_df["date"].max()

    st.title("AGRI-SMART DZ")
    st.caption(tr("AI-powered agricultural market intelligence for Algeria · Focus product: Potato"))
    st.caption(tr("All data in this demo is synthetic and clearly labeled as such; it is not a real Algerian market dataset."))
    if is_scenario:
        if snapshot_date < SCENARIO_DATE:
            st.warning(tr(
                "The dataset does not reach {date}; showing the latest available observations instead.",
                date=f"{SCENARIO_DATE:%d %b %Y}",
            ))
        else:
            st.info(tr("Demo scenario: July 2026 market pressure — price pressure in Alger, stock decline in Oran, and combined pressure in Constantine."))

    with st.sidebar:
        st.header(tr("Controls"))
        st.write(tr("Only the potato market is active in this MVP. The system is designed to extend to tomato, onion, and carrot markets later."))
        st.info(tr("Important: the app flags unusual patterns for verification; it does not accuse a person or business of speculation."))
        if st.button(tr("Refresh model training")):
            for horizon in dp.HORIZONS:
                train_and_evaluate(horizon, clean_df)
            get_model_predictions.clear()
            st.success(tr("Models retrained and saved."))

    predictions = (
        get_model_predictions(clean_df)
        if section in {"price_prediction", "what_if"}
        else None
    )

    if section == "dashboard":
        render_kpis(snapshot_df, alert_df, anomalies_df, snapshot_date)
        render_dashboard_charts(snapshot_df)
    elif section == "price_prediction":
        render_price_prediction(predictions, clean_df)
    elif section == "stock_monitoring":
        render_stock_monitoring(snapshot_df)
    elif section == "shortage_detection":
        render_shortage_detection(alert_df)
    elif section == "anomaly_detection":
        render_anomaly_detection(anomalies_df)
    elif section == "regional_analysis":
        render_regional_market(snapshot_df)
    elif section == "what_if":
        render_scenario_simulator(clean_df, predictions)
    elif section == "smart_alerts":
        render_smart_alerts(alert_df)
    elif section == "reports":
        render_reports(clean_df, alert_df)
    elif section == "about":
        render_about(clean_df)

    if section == "dashboard":
        with st.expander(tr("Recent product dataset preview")):
            preview = snapshot_df.tail(10).rename(columns={column: tr(column) for column in snapshot_df.columns})
            region_column = tr("region")
            preview[region_column] = preview[region_column].map(region_label)
            st.dataframe(preview, use_container_width=True)


if __name__ == "__main__":
    main()
