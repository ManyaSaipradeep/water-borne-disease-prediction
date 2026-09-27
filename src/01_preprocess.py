
import re
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

WATERBORNE_DISEASES = [
    "cholera", "typhoid", "dysentery", "diarrhea", "diarrhoea",
    "acute diarrheal disease", "acute diarrhoeal disease",
    "amoebiasis", "hepatitis a", "hepatitis e", "gastroenteritis",
]


def norm_name(x):
    if pd.isna(x):
        return np.nan
    return re.sub(r"\s+", " ", str(x).strip().lower())


def load_idsp():
    df = pd.read_csv(RAW / "dataful_idsp.csv")
    df = df.dropna(subset=["District", "year", "week"]).copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df[df["year"].between(2000, 2030)]  # drop the one corrupt row (16338)
    df["week"] = pd.to_numeric(df["week"], errors="coerce")
    df = df.dropna(subset=["week"])
    df["year"] = df["year"].astype(int)
    df["week"] = df["week"].astype(int)
    df["district_norm"] = df["District"].map(norm_name)
    df["disease_norm"] = df["disease_illness_name"].map(norm_name)
    df["cases"] = pd.to_numeric(df["cases"], errors="coerce").fillna(0)
    df["deaths"] = pd.to_numeric(df["deaths"], errors="coerce").fillna(0)
    df["is_waterborne"] = df["disease_norm"].apply(
        lambda d: isinstance(d, str) and any(k in d for k in WATERBORNE_DISEASES)
    )
    return df


def load_weather():
    df = pd.read_csv(RAW / "master_weather_features_full.csv", low_memory=False)
    # A large share of rows are column-shifted (State ended up in Year, etc).
    # Detect shifted rows (Year is non-numeric) and re-align them.
    year_numeric = pd.to_numeric(df["Year"], errors="coerce")
    shifted = year_numeric.isnull()

    fixed = pd.DataFrame(index=df.index)
    fixed["District"] = df["District"]
    fixed["State"] = np.where(shifted, df["Year"], np.nan)
    fixed["Year"] = np.where(shifted, df["EpiWeek"], df["Year"])
    fixed["EpiWeek"] = np.where(shifted, df["Avg_Temp_C"], df["EpiWeek"])
    fixed["Avg_Temp_C"] = np.where(shifted, df["Weekly_Rainfall"], df["Avg_Temp_C"])
    fixed["Weekly_Rainfall"] = np.where(
        shifted, pd.to_numeric(df["State"], errors="coerce"), df["Weekly_Rainfall"]
    )

    fixed["Year"] = pd.to_numeric(fixed["Year"], errors="coerce")
    fixed["EpiWeek"] = pd.to_numeric(fixed["EpiWeek"], errors="coerce")
    fixed["Avg_Temp_C"] = pd.to_numeric(fixed["Avg_Temp_C"], errors="coerce")
    fixed["Weekly_Rainfall"] = pd.to_numeric(fixed["Weekly_Rainfall"], errors="coerce")
    fixed = fixed.dropna(subset=["District", "Year", "EpiWeek"])
    fixed["Year"] = fixed["Year"].astype(int)
    fixed["EpiWeek"] = fixed["EpiWeek"].astype(int)
    fixed["district_norm"] = fixed["District"].map(norm_name)

    # fill remaining missing state using centroid lookup (district -> state)
    return fixed


def load_centroids():
    df = pd.read_csv(RAW / "district_wise_centroids.csv")
    df["district_norm"] = df["District"].map(norm_name)
    df = df.rename(columns={"State": "state_centroid"})
    return df[["district_norm", "state_centroid", "Latitude", "Longitude"]]


def build_outbreak_panel(idsp, weather):
    """
    Weekly outbreak label per district, built ONLY over the district x
    year x epiweek combinations that actually have a weather record
    (so every row has real environmental features -- no fabricated
    negatives outside the observed space).
    """
    agg = (
        idsp.groupby(["district_norm", "year", "week"])
        .agg(
            cases_reported=("cases", "sum"),
            deaths_reported=("deaths", "sum"),
            n_reports=("cases", "size"),
            waterborne_cases=("cases", lambda s: s[idsp.loc[s.index, "is_waterborne"]].sum()),
            any_waterborne=("is_waterborne", "any"),
        )
        .reset_index()
        .rename(columns={"year": "Year", "week": "EpiWeek"})
    )

    panel = weather.merge(
        agg, on=["district_norm", "Year", "EpiWeek"], how="left"
    )
    for col in ["cases_reported", "deaths_reported", "n_reports", "waterborne_cases"]:
        panel[col] = panel[col].fillna(0)
    panel["any_waterborne"] = panel["any_waterborne"].fillna(False)

    # Label: outbreak risk this district-week (any waterborne-disease
    # report OR a meaningfully elevated case count that week)
    panel["outbreak_label"] = (
        (panel["any_waterborne"]) | (panel["cases_reported"] >= 5)
    ).astype(int)
    return panel


def haversine(lat1, lon1, lat2, lon2):
    """Real great-circle distance (km) between two lat/long points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def add_neighbor_feature(panel, k_neighbors=5):
    """For each district-week, flags whether one of that district's
    k_neighbors nearest districts (by real centroid distance) reported an
    outbreak in the preceding 1 or 2 weeks. Never looks at the district's
    OWN history (that's the separate, unused cases_lag1/outbreak_lag1), and
    never looks across a year boundary (conservative: no signal carried
    from the previous year's last week into week 1)."""
    centroids = (
        panel[["district_norm", "Latitude", "Longitude"]]
        .dropna().drop_duplicates("district_norm").reset_index(drop=True)
    )
    n = len(centroids)
    lat = centroids["Latitude"].values
    lon = centroids["Longitude"].values

    neighbor_map = {}
    for i in range(n):
        dists = haversine(lat[i], lon[i], lat, lon)
        dists[i] = np.inf
        nearest_idx = np.argsort(dists)[:k_neighbors]
        neighbor_map[centroids["district_norm"].iloc[i]] = (
            centroids["district_norm"].iloc[nearest_idx].tolist()
        )

    week_key = panel["Year"].astype(str) + "_" + panel["EpiWeek"].astype(str).str.zfill(2)
    lookup = dict(zip(zip(week_key, panel["district_norm"]), panel["outbreak_label"]))

    def prev_week_key(year, week, back):
        wk = week - back
        return None if wk < 1 else f"{year}_{str(wk).zfill(2)}"

    years = panel["Year"].values
    weeks = panel["EpiWeek"].values
    districts = panel["district_norm"].values
    result = np.zeros(len(panel), dtype=int)
    for i in range(len(panel)):
        neighbors = neighbor_map.get(districts[i])
        if not neighbors:
            continue
        found = 0
        for back in (1, 2):
            wk = prev_week_key(years[i], weeks[i], back)
            if wk is None:
                continue
            for nb in neighbors:
                if lookup.get((wk, nb), 0) == 1:
                    found = 1
                    break
            if found:
                break
        result[i] = found
    panel["neighbor_outbreak_recent"] = result

    # Save the neighbor map itself so the dashboard can reuse the exact same
    # real-distance neighbor sets at prediction time, rather than recomputing
    # (and potentially drifting from) this logic.
    import json
    with open(OUT / "district_neighbor_map.json", "w") as f:
        json.dump(neighbor_map, f)

    return panel


def main():
    idsp = load_idsp()
    weather = load_weather()
    centroids = load_centroids()

    print(f"IDSP rows: {len(idsp)}, districts: {idsp.district_norm.nunique()}")
    print(f"Weather rows: {len(weather)}, districts: {weather.district_norm.nunique()}")

    panel = build_outbreak_panel(idsp, weather)
    panel = panel.merge(centroids, on="district_norm", how="left")

    # seasonal features
    panel = panel.sort_values(["district_norm", "Year", "EpiWeek"])
    panel["month_approx"] = ((panel["EpiWeek"] - 1) // 4.345 % 12 + 1).astype(int)
    panel["is_monsoon"] = panel["month_approx"].isin([6, 7, 8, 9]).astype(int)
    panel["rain_lag1"] = panel.groupby("district_norm")["Weekly_Rainfall"].shift(1)
    panel["rain_3wk_avg"] = (
        panel.groupby("district_norm")["Weekly_Rainfall"]
        .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    )
    panel["temp_lag1"] = panel.groupby("district_norm")["Avg_Temp_C"].shift(1)
    panel[["rain_lag1", "temp_lag1"]] = panel[["rain_lag1", "temp_lag1"]].fillna(
        panel[["Weekly_Rainfall", "Avg_Temp_C"]].mean()
    )

    # Recent-history (autoregressive) features: real signal -- a district's
    # OWN recent case/outbreak history is a strong, legitimate predictor of
    # near-future outbreaks (epidemic persistence), and unlike the removed
    # water-quality shortcut, this uses only real IDSP data already in the
    # panel, shifted so no row ever sees its own future.
    panel["cases_lag1"] = panel.groupby("district_norm")["cases_reported"].shift(1).fillna(0)
    panel["cases_3wk_avg"] = (
        panel.groupby("district_norm")["cases_reported"]
        .rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
        .shift(1).fillna(0)
    )
    panel["outbreak_lag1"] = panel.groupby("district_norm")["outbreak_label"].shift(1).fillna(0)

    # Weather anomaly features: how far this week's reading is from that
    # district's own "normal" for this time of year. This is a genuine
    # precursor-style signal (e.g. unusually heavy rain for a district's
    # typical month can precede contamination even before absolute rainfall
    # looks extreme) and, unlike the recent-case-history features, it does
    # NOT require already knowing anything about outbreaks -- only weather,
    # which is knowable in advance.
    #
    # Norms are computed at (district, month) granularity for statistical
    # stability (a weekly bucket would often have only 1-2 historical
    # samples per district), and -- to avoid leakage -- using ONLY the
    # training period (Year <= SPLIT_YEAR-1) so the test-period rows never
    # inform their own "normal."
    SPLIT_YEAR = 2024
    train_mask = panel["Year"] <= SPLIT_YEAR - 1
    norms = (
        panel.loc[train_mask]
        .groupby(["district_norm", "month_approx"])[["Weekly_Rainfall", "Avg_Temp_C"]]
        .mean()
        .rename(columns={"Weekly_Rainfall": "rain_norm", "Avg_Temp_C": "temp_norm"})
        .reset_index()
    )
    panel = panel.merge(norms, on=["district_norm", "month_approx"], how="left")
    # Fallback for any (district, month) combo with no training-period rows
    # at all (e.g. a district that only appears from 2024 onward): fall back
    # to that district's overall training-period mean, then to the global
    # training-period mean.
    district_fallback = (
        panel.loc[train_mask]
        .groupby("district_norm")[["Weekly_Rainfall", "Avg_Temp_C"]]
        .mean()
        .rename(columns={"Weekly_Rainfall": "rain_fallback", "Avg_Temp_C": "temp_fallback"})
    )
    panel = panel.merge(district_fallback, on="district_norm", how="left")
    global_rain_mean = panel.loc[train_mask, "Weekly_Rainfall"].mean()
    global_temp_mean = panel.loc[train_mask, "Avg_Temp_C"].mean()
    panel["rain_norm"] = panel["rain_norm"].fillna(panel["rain_fallback"]).fillna(global_rain_mean)
    panel["temp_norm"] = panel["temp_norm"].fillna(panel["temp_fallback"]).fillna(global_temp_mean)
    panel["rain_anomaly"] = panel["Weekly_Rainfall"] - panel["rain_norm"]
    panel["temp_anomaly"] = panel["Avg_Temp_C"] - panel["temp_norm"]
    panel = panel.drop(columns=["rain_fallback", "temp_fallback"])

    # Extreme/cumulative rainfall features -- trialed and confirmed to add
    # real signal (see src/06_trial_new_features.py for the comparison).
    # Thresholds computed from TRAINING rows only, same leakage-safe pattern.
    panel["monsoon_rain_component"] = np.where(panel["is_monsoon"] == 1, panel["Weekly_Rainfall"], 0.0)
    panel["monsoon_cum_rain"] = (
        panel.groupby(["district_norm", "Year"])["monsoon_rain_component"].cumsum()
    )
    panel = panel.drop(columns=["monsoon_rain_component"])

    extreme_thresh = (
        panel.loc[train_mask].groupby("district_norm")["Weekly_Rainfall"]
        .quantile(0.90).rename("extreme_thresh")
    )
    global_extreme_thresh = panel.loc[train_mask, "Weekly_Rainfall"].quantile(0.90)
    panel = panel.merge(extreme_thresh, on="district_norm", how="left")
    panel["extreme_thresh"] = panel["extreme_thresh"].fillna(global_extreme_thresh)
    panel["is_extreme_rain_week"] = (panel["Weekly_Rainfall"] >= panel["extreme_thresh"]).astype(int)
    panel["extreme_rain_count_6wk"] = (
        panel.groupby("district_norm")["is_extreme_rain_week"]
        .apply(lambda s: s.shift(1).rolling(6, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)

    dry_thresh = (
        panel.loc[train_mask].groupby("district_norm")["Weekly_Rainfall"]
        .quantile(0.25).rename("dry_thresh")
    )
    global_dry_thresh = panel.loc[train_mask, "Weekly_Rainfall"].quantile(0.25)
    panel = panel.merge(dry_thresh, on="district_norm", how="left")
    panel["dry_thresh"] = panel["dry_thresh"].fillna(global_dry_thresh)
    panel["is_dry_week"] = (panel["Weekly_Rainfall"] <= panel["dry_thresh"]).astype(int)
    panel["dry_streak_prior_3wk"] = (
        panel.groupby("district_norm")["is_dry_week"]
        .apply(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    )
    panel["dry_then_deluge"] = (
        (panel["dry_streak_prior_3wk"].fillna(0) >= 3) & (panel["is_extreme_rain_week"] == 1)
    ).astype(int)
    panel = panel.drop(columns=["dry_thresh", "is_dry_week", "dry_streak_prior_3wk"])
    # keep extreme_thresh (renamed) around for the dashboard to reuse at prediction time
    panel = panel.rename(columns={"extreme_thresh": "extreme_rain_threshold"})

    # Neighboring-district signal -- the strongest single feature found in
    # trials (see src/06_trial_new_features.py): whether a geographically
    # nearby district (by real centroid distance) reported an outbreak in
    # the preceding 1-2 weeks. Confirmed via a per-state breakdown to be a
    # genuine within-state signal, not just a state-reporting-bias proxy.
    panel = add_neighbor_feature(panel, k_neighbors=5)

    # Reporting-frequency feature: how often IDSP has filed ANY report for
    # this district in the trailing 52 weeks. This doesn't fix the
    # underlying "no report != confirmed no outbreak" ambiguity (see
    # README's root-cause diagnosis) -- passive surveillance data can't be
    # cleaned that way -- but it gives the model a real, legitimate way to
    # partially tell apart "district reports normally and currently has
    # nothing to report" from "district rarely reports at all," instead of
    # silently treating both as identical zeros. Computed causally (shift
    # before rolling, so no row ever sees its own future).
    panel["reported_this_week"] = (panel["n_reports"] > 0).astype(int)
    panel["reporting_freq_52wk"] = (
        panel.groupby("district_norm")["reported_this_week"]
        .apply(lambda s: s.shift(1).rolling(52, min_periods=1).sum())
        .reset_index(level=0, drop=True)
    ).fillna(0)
    panel = panel.drop(columns=["reported_this_week"])

    out_path = OUT / "model_dataset.csv"
    panel.to_csv(out_path, index=False)
    print(f"Saved merged dataset: {out_path} shape={panel.shape}")
    print("Label distribution:\n", panel["outbreak_label"].value_counts())
    print("Districts covered:", panel["district_norm"].nunique())


if __name__ == "__main__":
    main()
