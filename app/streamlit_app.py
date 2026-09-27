"""
Streamlit dashboard — Smart Community Health Monitoring and Early
Warning System for Water-Borne Diseases in India.

Run with:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
DATA = ROOT / "data" / "processed"

st.set_page_config(
    page_title="Water-Borne Disease Early Warning System",
    page_icon="💧",
    layout="wide",
)


@st.cache_resource
def load_models():
    outbreak_bundle = joblib.load(MODELS / "outbreak_model.joblib")
    potability_bundle = joblib.load(MODELS / "potability_submodel.joblib")
    return outbreak_bundle, potability_bundle


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA / "model_dataset.csv", low_memory=False)
    return df


@st.cache_data
def load_neighbor_map():
    path = DATA / "district_neighbor_map.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


LAGGED_FEATURES = [
    "Avg_Temp_C", "Weekly_Rainfall", "rain_lag1", "rain_3wk_avg",
    "temp_lag1", "is_monsoon", "rain_anomaly", "temp_anomaly",
    "monsoon_cum_rain", "extreme_rain_count_6wk", "dry_then_deluge",
    "neighbor_outbreak_recent", "reporting_freq_52wk",
]
AGG_RULES = {
    "Avg_Temp_C": "mean", "Weekly_Rainfall": "mean", "rain_lag1": "mean",
    "rain_3wk_avg": "mean", "temp_lag1": "mean", "rain_anomaly": "mean",
    "temp_anomaly": "mean", "is_monsoon": "max", "monsoon_cum_rain": "max",
    "extreme_rain_count_6wk": "max", "dry_then_deluge": "max",
    "neighbor_outbreak_recent": "max", "reporting_freq_52wk": "mean",
    "outbreak_label": "max", "state_centroid": "first",
}


@st.cache_data
def build_monthly_lagged_panel(df: pd.DataFrame) -> pd.DataFrame:
    """One row per district-month. Every predictor is aggregated from the
    PRIOR month only (shift(1) per district) -- mirrors
    src/03_train_outbreak_model.py exactly, so the app's predictions match
    what the saved model was actually trained/evaluated on."""
    monthly = (
        df.groupby(["district_norm", "Year", "month_approx"])
        .agg(AGG_RULES)
        .reset_index()
        .sort_values(["district_norm", "Year", "month_approx"])
    )
    lagged = monthly.groupby("district_norm")[LAGGED_FEATURES].shift(1)
    lagged.columns = [f"{c}_prevmonth" for c in lagged.columns]
    out = pd.concat([
        monthly[["district_norm", "Year", "month_approx", "outbreak_label", "state_centroid"]],
        lagged,
    ], axis=1)
    return out


def contamination_risk(potability_bundle, water_params: dict) -> float:
    X = pd.DataFrame([water_params])[potability_bundle["features"]]
    X_imp = potability_bundle["imputer"].transform(X)
    proba_potable = potability_bundle["model"].predict_proba(X_imp)[:, 1][0]
    return 1 - proba_potable


def main():
    outbreak_bundle, potability_bundle = load_models()
    clf, imputer, features = (
        outbreak_bundle["model"], outbreak_bundle["imputer"], outbreak_bundle["features"]
    )
    state_encoder = outbreak_bundle["state_encoder"]
    outbreak_threshold = outbreak_bundle.get("threshold", 0.5)
    df = load_dataset()
    neighbor_map = load_neighbor_map()
    df["state_centroid"] = df["state_centroid"].fillna("unknown")

    st.title("💧 Smart Community Health Monitoring & Early Warning System")
    st.caption("Early prediction of water-borne disease outbreak risk from weather and water-quality signals")

    tab1, tab2 = st.tabs(["🔮 Predict Risk", "📊 Data Explorer"])

    # ---------------- TAB 1: Prediction ----------------
    with tab1:
        st.subheader("Module B.1 — Water Contamination Risk")
        st.caption(
            "Type in an actual water-test reading. This model was trained on the "
            "real Kaggle potability dataset (3,276 lab samples) -- nothing here is "
            "auto-filled or guessed for you, because that data has no district/date "
            "link to attach it to a place automatically."
        )
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            ph = st.number_input("pH", value=7.0)
            hardness = st.number_input("Hardness", value=196.0)
            solids = st.number_input("Solids", value=20000.0)
        with wc2:
            chloramines = st.number_input("Chloramines", value=7.0)
            sulfate = st.number_input("Sulfate", value=333.0)
            conductivity = st.number_input("Conductivity", value=425.0)
        with wc3:
            organic_carbon = st.number_input("Organic carbon", value=14.0)
            trihalomethanes = st.number_input("Trihalomethanes", value=66.0)
            turbidity = st.number_input("Turbidity", value=4.0)

        water_params = dict(
            ph=ph, Hardness=hardness, Solids=solids, Chloramines=chloramines,
            Sulfate=sulfate, Conductivity=conductivity, Organic_carbon=organic_carbon,
            Trihalomethanes=trihalomethanes, Turbidity=turbidity,
        )
        contam_risk = contamination_risk(potability_bundle, water_params)
        contam_level = "HIGH" if contam_risk >= 0.5 else ("MODERATE" if contam_risk >= 0.25 else "LOW")
        st.metric("Contamination risk probability", f"{contam_risk:.1%}", contam_level)

        st.markdown("---")
        st.subheader("Module B.2 — Disease Outbreak Risk")
        st.caption(
            "Predicts using a model trained only on real weather, season, and state -- "
            "no case-history shortcut, no synthetic inputs. This is a precursor-based "
            "early-warning forecast at MONTHLY grain: it uses only LAST month's "
            "aggregated weather/precursor signals to forecast THIS month's outbreak "
            "risk (never this month's own data), so it never needs to already know "
            "whether an outbreak is underway before it warns about one."
        )

        districts = sorted(df["district_norm"].dropna().unique())
        selected_district = st.selectbox("District", districts)
        d_hist = df[df["district_norm"] == selected_district].sort_values(["Year", "EpiWeek"])
        state_val = d_hist["state_centroid"].dropna().iloc[0] if not d_hist["state_centroid"].dropna().empty else "unknown"
        # District's month-level weather "normal" (from the training period),
        # used to turn a typed reading into an anomaly feature -- how far
        # off from what's normal for this district at this time of year.
        norms_by_month = (
            d_hist.dropna(subset=["rain_norm", "temp_norm"])
            .drop_duplicates("month_approx")
            .set_index("month_approx")[["rain_norm", "temp_norm"]]
        )
        fallback_rain_norm = d_hist["rain_norm"].mean() if not d_hist["rain_norm"].dropna().empty else 15.0
        fallback_temp_norm = d_hist["temp_norm"].mean() if not d_hist["temp_norm"].dropna().empty else 27.0

        import datetime
        neighbors = neighbor_map.get(selected_district, [])

        monthly_panel = build_monthly_lagged_panel(df)
        d_monthly = monthly_panel[monthly_panel["district_norm"] == selected_district].sort_values(
            ["Year", "month_approx"]
        )
        d_monthly_ready = d_monthly.dropna(subset=[f"{f}_prevmonth" for f in LAGGED_FEATURES])

        use_historical = st.checkbox(
            "Check against a real historical record instead (see the actual outcome)",
            value=False,
            help="Pick a past district-month from our data: inputs below are auto-filled "
                 "from that month's PREVIOUS month (exactly what the model actually uses), "
                 "and you can compare the prediction to what really happened THAT month. "
                 "Leave this unchecked (the default) to type in last month's readings "
                 "yourself for any target month -- including a future one -- for a "
                 "what-if forecast.",
        )

        if use_historical and not d_monthly_ready.empty:
            years = sorted(d_monthly_ready["Year"].unique())
            sel_year = st.selectbox("Year", years, index=len(years) - 1)
            months_for_year = sorted(d_monthly_ready[d_monthly_ready["Year"] == sel_year]["month_approx"].unique())
            sel_month_pick = st.selectbox(
                "Month being forecast (inputs below are auto-filled from the month BEFORE this)",
                months_for_year, index=len(months_for_year) - 1,
            )
            record = d_monthly_ready[
                (d_monthly_ready["Year"] == sel_year) & (d_monthly_ready["month_approx"] == sel_month_pick)
            ]
            rec = record.iloc[0]
            default_temp = float(rec["Avg_Temp_C_prevmonth"])
            default_rain = float(rec["Weekly_Rainfall_prevmonth"])
            default_rain_lag1, default_rain_3wk = float(rec["rain_lag1_prevmonth"]), float(rec["rain_3wk_avg_prevmonth"])
            default_temp_lag1, default_month = float(rec["temp_lag1_prevmonth"]), int(sel_month_pick)
            default_monsoon_cum = float(rec.get("monsoon_cum_rain_prevmonth", 0.0) or 0.0)
            default_extreme_count = float(rec.get("extreme_rain_count_6wk_prevmonth", 0.0) or 0.0)
            default_dry_deluge = bool(rec.get("dry_then_deluge_prevmonth", 0))
            default_reporting_freq = float(rec.get("reporting_freq_52wk_prevmonth", 0.0) or 0.0)
            default_neighbor_flag = bool(rec.get("neighbor_outbreak_recent_prevmonth", 0))
            ground_truth = rec
        else:
            if use_historical and d_monthly_ready.empty:
                st.info(f"No historical month with a full prior-month record exists for {selected_district} -- enter last month's readings below instead.")
            target_date = st.date_input(
                "Month being forecast (the month these predictions are FOR)",
                value=datetime.date.today(),
                help="Sets the calendar month used below (knowable in advance) and for "
                     "the seasonal-anomaly comparison -- it does not need to be today. "
                     "All the numeric inputs below should describe the PREVIOUS month, "
                     "since that's the only real data this forecast is allowed to use.",
            )
            sel_year = target_date.year
            default_month = target_date.month
            latest = d_monthly_ready.iloc[-1] if not d_monthly_ready.empty else None

            if latest is not None:
                st.caption(
                    f"Forecasting for: {target_date.strftime('%B %Y')}. Starting values below "
                    f"are {selected_district}'s most recently available PRIOR-month record, "
                    "purely as a convenience -- overwrite them with the real previous "
                    "month's numbers."
                )
                default_temp = float(latest["Avg_Temp_C_prevmonth"])
                default_rain = float(latest["Weekly_Rainfall_prevmonth"])
                default_rain_lag1, default_rain_3wk = float(latest["rain_lag1_prevmonth"]), float(latest["rain_3wk_avg_prevmonth"])
                default_temp_lag1 = float(latest["temp_lag1_prevmonth"])
                default_monsoon_cum = float(latest.get("monsoon_cum_rain_prevmonth", 0.0) or 0.0)
                default_extreme_count = float(latest.get("extreme_rain_count_6wk_prevmonth", 0.0) or 0.0)
                default_dry_deluge = bool(latest.get("dry_then_deluge_prevmonth", 0))
                default_reporting_freq = float(latest.get("reporting_freq_52wk_prevmonth", 0.0) or 0.0)
            else:
                st.warning(
                    "New data: this district has no historical record in our dataset -- "
                    "please enter all fields below yourself. Nothing is pre-filled or "
                    "guessed for it."
                )
                default_temp, default_rain = 0.0, 0.0
                default_rain_lag1, default_rain_3wk, default_temp_lag1 = 0.0, 0.0, 0.0
                default_monsoon_cum, default_extreme_count, default_dry_deluge = 0.0, 0.0, False
                default_reporting_freq = 0.0

            # No live neighbor surveillance feed exists, so this uses each
            # neighbor's MOST RECENT available prior-month record in our
            # data as the best real (not fabricated) proxy for "recent"
            # neighbor status.
            default_neighbor_flag = False
            if neighbors:
                nb_monthly = monthly_panel[monthly_panel["district_norm"].isin(neighbors)].sort_values(
                    ["Year", "month_approx"]
                )
                latest_per_nb = nb_monthly.groupby("district_norm").tail(1)
                default_neighbor_flag = bool((latest_per_nb["neighbor_outbreak_recent_prevmonth"] == 1).any())
            ground_truth = None

        st.caption(
            "**All numeric fields below describe LAST month** (the last full month "
            "before the forecast month) -- the model never sees the forecast month's "
            "own weather, by design."
        )
        wc4, wc5 = st.columns(2)
        with wc4:
            avg_temp = st.number_input("Last month's average temperature (°C)", value=default_temp)
            weekly_rain = st.number_input("Last month's average weekly rainfall (mm)", value=default_rain)
            rain_lag1 = st.number_input("Rainfall, month before that (mm)", value=default_rain_lag1)
        with wc5:
            rain_3wk = st.number_input("Last month's 3-week average rainfall (mm)", value=default_rain_3wk)
            temp_lag1 = st.number_input("Avg. temp, month before that (°C)", value=default_temp_lag1)
            month = st.number_input(
                "Month being forecast (1-12)", min_value=1, max_value=12, value=default_month,
                help="The calendar month this forecast is FOR -- known in advance, not data.",
            )
        is_monsoon = 1 if month in (6, 7, 8, 9) else 0

        st.markdown("**Additional precursor signals** (found in testing to matter more than plain weather averages)")
        wc6, wc7 = st.columns(2)
        with wc6:
            monsoon_cum_rain = st.number_input(
                "Cumulative monsoon-season rainfall through the end of last month (mm)",
                value=default_monsoon_cum, min_value=0.0,
                help="Running total of rainfall since June, as of last month -- a "
                     "season-long saturation signal, not just one reading.",
            )
            extreme_rain_count_6wk = st.number_input(
                "Number of unusually heavy-rain weeks in the last 6 weeks (up to last month)",
                value=default_extreme_count, min_value=0.0, step=1.0,
                help="How many of the 6 weeks up through last month had rainfall above this "
                     "district's own historical top-10% level.",
            )
        with wc7:
            dry_then_deluge = st.checkbox(
                "Dry spell (3+ low-rain weeks) immediately followed by a sudden heavy-rain week, last month",
                value=default_dry_deluge,
                help="A known contamination pattern: waste concentrates during a dry spell, "
                     "then a sudden downpour flushes it into water sources.",
            )
            neighbor_outbreak_recent = st.checkbox(
                "A nearby district reported an outbreak last month",
                value=default_neighbor_flag,
                help="Auto-suggested from the nearest 5 districts by real distance and their "
                     "most recently available outbreak record -- override if you have more "
                     "current information.",
            )
            reporting_freq_52wk = st.number_input(
                "Weeks in the trailing year (as of last month) this district filed ANY surveillance report",
                value=default_reporting_freq, min_value=0.0, max_value=52.0, step=1.0,
                help="How often this district reports at all, regardless of outcome. This was "
                     "the single strongest predictor found in testing -- it helps the model tell "
                     "apart a district that reports regularly (so a quiet spell is more likely a "
                     "genuinely calm period) from one that rarely reports at all (where a quiet "
                     "spell may just mean nothing was recorded, not that nothing happened).",
            )
        state_encoded = state_encoder.transform([[state_val if pd.notna(state_val) else "unknown"]])[0, 0]
        st.caption(f"State (auto-derived from district): {state_val}")

        # Anomaly features: how far this reading is from what's normal for
        # THIS district at THIS time of year (training-period norms).
        if month in norms_by_month.index:
            rain_norm = float(norms_by_month.loc[month, "rain_norm"])
            temp_norm = float(norms_by_month.loc[month, "temp_norm"])
        else:
            rain_norm, temp_norm = float(fallback_rain_norm), float(fallback_temp_norm)
        rain_anomaly = weekly_rain - rain_norm
        temp_anomaly = avg_temp - temp_norm
        st.caption(
            f"Normal for {selected_district} in month {month}: "
            f"~{rain_norm:.1f}mm rain, ~{temp_norm:.1f}°C "
            f"(this reading: {rain_anomaly:+.1f}mm, {temp_anomaly:+.1f}°C vs. normal)"
        )

        row = pd.DataFrame([{
            "Avg_Temp_C_prevmonth": avg_temp, "Weekly_Rainfall_prevmonth": weekly_rain,
            "rain_lag1_prevmonth": rain_lag1, "rain_3wk_avg_prevmonth": rain_3wk,
            "temp_lag1_prevmonth": temp_lag1,
            "is_monsoon_prevmonth": is_monsoon, "month_approx": month,
            "rain_anomaly_prevmonth": rain_anomaly, "temp_anomaly_prevmonth": temp_anomaly,
            "monsoon_cum_rain_prevmonth": monsoon_cum_rain,
            "extreme_rain_count_6wk_prevmonth": extreme_rain_count_6wk,
            "dry_then_deluge_prevmonth": int(dry_then_deluge),
            "neighbor_outbreak_recent_prevmonth": int(neighbor_outbreak_recent),
            "reporting_freq_52wk_prevmonth": reporting_freq_52wk,
            "state_encoded": state_encoded,
        }])[features]
        row_imp = imputer.transform(row)
        outbreak_proba = clf.predict_proba(row_imp)[0, 1]
        outbreak_level = (
            "HIGH" if outbreak_proba >= outbreak_threshold
            else ("MODERATE" if outbreak_proba >= outbreak_threshold / 2 else "LOW")
        )
        st.metric("Outbreak risk score", f"{outbreak_proba:.1%}", outbreak_level)
        st.caption(
            f"This is a relative risk score from the model, not a calibrated "
            f"real-world probability -- EasyEnsemble trains on artificially "
            f"rebalanced undersamples internally, so raw scores run higher than "
            f"true monthly outbreak prevalence (~12.9% in the 2024 test period) "
            f"across the board. Compare it to the tuned decision threshold below, "
            f"not to 50%, and read it as \"higher/lower risk than usual for this "
            f"district next month,\" not \"X% chance this will happen.\"  Decision "
            f"threshold: {outbreak_threshold:.1%} (tuned on training data to "
            "maximize F1 for this imbalanced, monthly-grain forecasting problem -- "
            "this model favors catching more real outbreaks (56% recall) over "
            "fewer false alarms (29% precision); see the README for the full "
            "model-comparison tradeoff)."
        )

        if ground_truth is not None:
            with st.expander("What actually happened in this district-month (ground truth)", expanded=True):
                actual = "Yes" if ground_truth.get("outbreak_label", 0) == 1 else "No"
                st.write(f"Outbreak label (actual, this forecasted month): {actual}")
                agree = (outbreak_proba >= outbreak_threshold) == (ground_truth.get("outbreak_label", 0) == 1)
                if agree:
                    st.success("Model's HIGH/not-HIGH call matches what actually happened that month.")
                else:
                    st.info(
                        "Model's call didn't match this specific month -- expected some of the "
                        "time given this model's precision/recall tradeoffs on this imbalanced "
                        "problem; not evidence of a bug on its own."
                    )

        st.markdown("---")
        st.subheader(f"Seasonal history for {selected_district}")
        st.caption(
            "For this district, how does a specific month's outbreak activity and "
            "rainfall compare year over year? Pick a month to see whether this "
            "month tends to run wetter/riskier some years than others. The axis "
            "scale is fixed for this district across all 12 months, so only the "
            "line moves when you switch months -- the scale itself won't change."
        )
        month_names = ["January", "February", "March", "April", "May", "June",
                        "July", "August", "September", "October", "November", "December"]
        default_month_idx = (int(default_month) - 1) if 1 <= int(default_month) <= 12 else 0

        # Streamlit keeps a widget's own session-state value once it has a
        # fixed `key`, and ignores `index=` on later reruns -- so without
        # this, the selector would silently "stick" to whatever month was
        # last shown instead of following a new date/district. We only
        # force it back to the upstream default when THAT default itself
        # changes (new date picked, or a district/historical-record switch
        # changed default_month) -- the user can still freely pick any
        # other month afterward without it snapping back on every rerun.
        default_tracker_key = "seasonal_history_month_default_tracker"
        if st.session_state.get(default_tracker_key) != default_month_idx:
            st.session_state["seasonal_history_month"] = default_month_idx + 1
            st.session_state[default_tracker_key] = default_month_idx

        month_view = st.selectbox(
            "Month", options=list(range(1, 13)),
            format_func=lambda m: month_names[m - 1],
            key="seasonal_history_month",
        )
        # An "outbreak" is a distinct EPISODE, not a week. A single outbreak
        # can run for several consecutive weeks -- counting each of those
        # weeks separately would count one real outbreak as several, and
        # bias any rain/outbreak comparison toward longer-running outbreaks.
        # So an episode is defined as the WEEK an outbreak starts (label
        # flips 0 -> 1) on this district's full chronological week series;
        # the month/year it's counted under is the month/year that start
        # week falls in.
        d_sorted = d_hist.sort_values(["Year", "EpiWeek"]).reset_index(drop=True)
        is_outbreak = d_sorted["outbreak_label"] == 1
        episode_start = is_outbreak & ~is_outbreak.shift(1, fill_value=False)
        episodes = d_sorted[episode_start]

        month_hist = d_hist[d_hist["month_approx"] == month_view]
        # Fixed axis scale for THIS district, computed once across ALL 12
        # months (not just the selected one) -- so switching the month
        # selector only moves the line, it never rescales the axes. Without
        # this, each month's own max would set the scale, making a small
        # change in one month look dramatic and a big change in another
        # look flat, purely from rescaling rather than real signal.
        all_months_episodes = (
            episodes.groupby(["Year", "month_approx"]).size().reset_index(name="outbreak_count")
        )
        all_months_rain = (
            d_hist.groupby(["Year", "month_approx"])["Weekly_Rainfall"].mean().reset_index(name="avg_rainfall")
        )
        district_max_count = int(all_months_episodes["outbreak_count"].max()) if not all_months_episodes.empty else 0
        district_max_rain = float(all_months_rain["avg_rainfall"].max()) if not all_months_rain.empty else 0.0

        if month_hist.empty:
            st.info(f"No historical records for {selected_district} in {month_names[month_view - 1]}.")
        else:
            month_episodes = (
                episodes[episodes["month_approx"] == month_view]
                .groupby("Year").size().reset_index(name="outbreak_count")
            )
            month_rain = (
                month_hist.groupby("Year")["Weekly_Rainfall"].mean().reset_index(name="avg_rainfall")
            )
            yearly = month_rain.merge(month_episodes, on="Year", how="left")
            yearly["outbreak_count"] = yearly["outbreak_count"].fillna(0).astype(int)
            yearly = yearly.sort_values("Year")

            years_str = yearly["Year"].astype(str)
            fig, ax1 = plt.subplots(figsize=(7, 4))
            ax1.plot(years_str, yearly["outbreak_count"], color="#c0392b", marker="o",
                     linewidth=2, label="Outbreaks (distinct episodes)")
            ax1.set_ylabel("Outbreaks (count)", color="#c0392b")
            ax1.tick_params(axis="y", labelcolor="#c0392b")
            # Fixed range/ticks for this district, same for every month.
            ax1.set_ylim(0, max(district_max_count, 1) * 1.1)
            tick_step = max(1, -(-district_max_count // 5))  # at most ~5 ticks, step >= 1
            ax1.set_yticks(range(0, district_max_count + tick_step, tick_step))

            ax2 = ax1.twinx()
            ax2.plot(years_str, yearly["avg_rainfall"], color="#2b6cb0", marker="s",
                     linewidth=2, linestyle="--", label="Avg. weekly rainfall (mm)")
            ax2.set_ylabel("Avg. weekly rainfall (mm)", color="#2b6cb0")
            ax2.tick_params(axis="y", labelcolor="#2b6cb0")
            # Fixed range for this district, same for every month.
            ax2.set_ylim(0, max(district_max_rain, 1.0) * 1.1)

            ax1.set_title(f"{month_names[month_view - 1]} in {selected_district}: outbreaks vs. rainfall, by year")
            ax1.set_xlabel("Year")
            ax1.grid(axis="x", alpha=0.2)
            plt.setp(ax1.get_xticklabels(), rotation=45, ha="right")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            st.caption(
                "Solid red line: how many distinct outbreaks STARTED in this month, "
                "that year (a single outbreak that runs 2-3 weeks still counts as one, "
                "not one per week). Dashed blue line: how much it rained on average "
                "that month. Watch whether the two lines rise and fall together year "
                "to year -- that's a visible rain-outbreak link for this specific "
                "district and month; if they move independently, rainfall alone "
                "doesn't explain this district's pattern in this month."
            )

        st.markdown("---")
        st.subheader("Combined Risk Summary")
        # NOTE: Module B.1 (contamination) and Module B.2 (outbreak) are trained
        # on separate datasets with no shared ground truth, so there is no
        # calibrated joint probability to compute here (see README/report for
        # why averaging the two raw probabilities was rejected). Instead, each
        # model's own already-tuned HIGH/MODERATE/LOW level is combined with a
        # simple, explainable OR rule: the combined level is the MORE SEVERE of
        # the two component levels, on the reasoning that either risk alone is
        # actionable -- a district should not be told "overall risk is low"
        # just because a moderate outbreak risk got diluted by averaging with a
        # low contamination reading, or vice versa.
        _SEVERITY = {"LOW": 0, "MODERATE": 1, "HIGH": 2}
        overall_level = max(contam_level, outbreak_level, key=lambda lvl: _SEVERITY[lvl])
        driver = (
            "both signals agree" if contam_level == outbreak_level
            else ("water contamination" if _SEVERITY[contam_level] > _SEVERITY[outbreak_level]
                  else "disease-outbreak risk")
        )
        c1, c2, c3 = st.columns(3)
        c1.metric("Water contamination risk", f"{contam_risk:.1%}", contam_level)
        c2.metric("Outbreak risk", f"{outbreak_proba:.1%}", outbreak_level)
        c3.metric("Overall risk (max of the two)", overall_level)
        st.caption(
            f"Combined by taking the more severe of the two independently-tuned "
            f"levels (driven by: {driver}), not by averaging their raw "
            f"probabilities -- the two models predict different things on "
            f"different datasets, so there is no validated way to blend their "
            f"raw scores into a single calibrated probability (see the "
            f"README/report for why). Trade-off to be aware of: this OR-style "
            f"rule is deliberately sensitive -- a single HIGH from either model "
            f"forces overall HIGH, with no averaging-out from the other signal. "
            f"That includes Module B.1, whose own accuracy (66.8%) is only "
            f"moderately above chance, so a false-positive contamination read "
            f"can drive an overall HIGH on its own. This was a deliberate choice "
            f"(consistent with favoring recall over precision elsewhere in this "
            f"project) but it is not free -- it is not itself validated against "
            f"any real combined outcome, since none exists in the data."
        )
        if overall_level == "HIGH":
            st.error("⚠️ High risk on at least one signal -- recommend heightened surveillance and water-quality checks.")
        elif overall_level == "MODERATE":
            st.warning("Moderate risk on at least one signal -- worth monitoring.")
        else:
            st.success("Low risk on both signals under these conditions.")

    # ---------------- TAB 2: Data Explorer ----------------
    with tab2:
        st.subheader("Explore the merged district-week dataset")
        districts = sorted(df["district_norm"].dropna().unique())
        sel = st.multiselect("Filter by district", districts, default=districts[:5])
        view = df[df["district_norm"].isin(sel)] if sel else df
        st.dataframe(
            view[["district_norm", "Year", "EpiWeek", "Avg_Temp_C", "Weekly_Rainfall",
                  "cases_reported", "outbreak_label"]].head(500),
            width="stretch",
        )
        st.markdown("**Outbreak label distribution**")
        st.bar_chart(df["outbreak_label"].value_counts())

        st.markdown("**Rainfall vs. reported cases (sample districts)**")
        chart_df = view.groupby(["Year", "EpiWeek"]).agg(
            rainfall=("Weekly_Rainfall", "mean"), cases=("cases_reported", "sum")
        ).reset_index()
        chart_df["period"] = (
            chart_df["Year"].astype(str) + "-W" + chart_df["EpiWeek"].astype(str).str.zfill(2)
        )
        chart_df = chart_df.sort_values(["Year", "EpiWeek"])
        st.line_chart(chart_df.set_index("period")[["rainfall", "cases"]])


if __name__ == "__main__":
    main()
