from __future__ import annotations

from pathlib import Path
import argparse
import shutil
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

import analysis as base


PRICE_COL = "vwap"
SIDE_VOL_COLS = ("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid")

INTRADAY_SEQ_WINDOWS = (5, 10, 20, 30, 60)
INTRADAY_ABNORMAL_QS = (0.90,)
INTRADAY_ESTIMATION_WINDOW_RETURNS = 120
INTRADAY_FUTURE_HORIZONS = (5, 10, 20, 30, 60, 120, 180)
INTRADAY_QUANTILE_LOOKBACK = 390 * 20

DAILY_ESTIMATION_WINDOW_RETURNS = 63
DAILY_FUTURE_HORIZONS = tuple(range(1, 11))
DAILY_QUANTILE_LOOKBACK_DAYS = 60
DAILY_SEGMENTS = (
    ("seg1_open", "09:30:00", "11:30:00"),
    ("seg2_mid", "11:30:00", "13:30:00"),
    ("seg3_close", "13:30:00", "16:00:00"),
)

MIN_ESTIMATION_RETURNS = 30
EXAMPLE_INTRADAY_HORIZON = 60
EXAMPLE_DAILY_HORIZON_DAYS = 5
MAX_EXAMPLES_PER_ASSET = 1

GROUP_SCHEMES = (
    "volume_level",
    "imbalance_side",
    "exclusive_location",
    "both_above_below_high",
)

GROUP_COLS = (
    "is_high_total_volume",
    "volume_level",
    "imbalance_side",
    "exclusive_location",
    "both_above_below_high",
    "is_high_above_mid",
    "is_high_below_mid",
    "is_high_at_mid",
)

GROUP_LABEL_NAMES = {
    "volume_level": "High / Low Total Volume",
    "imbalance_side": "Imbalance Side",
    "exclusive_location": "Exclusive High-Volume Location",
    "both_above_below_high": "Both Above+Below High",
}

FREQUENCY_LABELS = {
    "intraday": "Intraday",
    "daily": "Daily",
}

SHORT_STRATEGY_TARGET_BPS = 200.0
SHORT_STRATEGY_STOP_BPS = 200.0
DEFAULT_POSITION_SHARES = 100
DEFAULT_COMMISSION_PER_TRADE = 0.0
SPREAD_COST_PER_SHARE_PER_SIDE = 0.10

# Daily RSD-based drift-strength screen configuration.
# Sampling remains 1 minute; Delta is measured in one-minute units here.
DAILY_RSD_DELTA_T = 1.0
DAILY_RSD_LAG_K = 1
DAILY_ST_THRESHOLD = 1.96
MIN_MINUTES_PER_DAY = 200
DETECTED_EVENTS_CSV_NAME = "detected_drift_events.csv"
LOCATION_COMPARE_GROUPS = ("above_mid_only", "below_mid_only", "none")
LOCATION_COMPARE_COLORS = {
    "above_mid_only": "#d62728",
    "below_mid_only": "#1f77b4",
    "none": "#222222",
}
INTRADAY_BASELINE_SEQ_WINDOW = max(INTRADAY_SEQ_WINDOWS)
INTRADAY_BASELINE_STRIDE = 15
REPORT_MODE_TAG = "daily_only_screen_sign_v1"
TABLE_IMAGE_DPI = 600
PLOT_IMAGE_DPI = 600


DISPLAY_LABELS = {
    "variable": "Measure",
    "n": "N",
    "sd": "Std. dev.",
    "outliers_removed": "Outliers removed",
    "above_volume_1d": "Above-mid volume",
    "below_volume_1d": "Below-mid volume",
    "neither_at_mid_volume_1d": "At-mid volume",
    "price_change_1d_pct": "Daily VWAP price change (%)",
    "total_volume_1d": "Total volume",
    "mu_hat": "Mean log return",
    "sigma_hat": "Log-return volatility",
    "mean_current_mu_hat": "Current mean log return",
    "mean_current_sigma_hat": "Current log-return volatility",
    "mean_future_mu_hat": "Future mean log return",
    "mean_future_sigma_hat": "Future log-return volatility",
    "screen_result": "Screen result",
    "normal": "Normal volume days",
    "above_mid_only": "Above-mid abnormal volume",
    "below_mid_only": "Below-mid abnormal volume",
    "none": "No abnormal volume",
    "other_abnormal_mix": "Mixed abnormal volume",
    "high_volume": "High volume",
    "low_volume": "Low volume",
    "above_mid_imbalance": "Above-mid imbalance",
    "below_mid_imbalance": "Below-mid imbalance",
    "both_high": "Both above and below high",
    "not_both_high": "Not both high",
}


def _display_label(value: object) -> str:
    text = str(value)
    if text.startswith("abnormal_q") and text[len("abnormal_q"):].isdigit():
        return f"Abnormal volume days (q{text[len('abnormal_q'):]})"
    if "_" not in text and text not in DISPLAY_LABELS:
        return text
    return DISPLAY_LABELS.get(text, text.replace("_", " ").title())


def _pretty_group_value(grouping_scheme: str, group_value: object) -> str:
    value = str(group_value)
    mapping = {
        "high_volume": "High volume",
        "low_volume": "Low volume",
        "above_mid_imbalance": "Above-mid imbalance",
        "below_mid_imbalance": "Below-mid imbalance",
        "neutral": "neutral",
        "below_mid_only": "Below-mid abnormal volume",
        "above_mid_only": "Above-mid abnormal volume",
        "none": "No abnormal volume",
        "other_abnormal_mix": "Mixed abnormal volume",
        "both_high": "Both above and below high",
        "not_both_high": "Not both high",
    }
    return mapping.get(value, _display_label(value))


def _signal_group_label(drift_sign: str, grouping_scheme: str, group_value: object) -> str:
    drift_text = _pretty_drift_label(drift_sign)
    group_text = _pretty_group_value(grouping_scheme, group_value)
    if grouping_scheme == "exclusive_location":
        return f"{drift_text}: {group_text}"
    if grouping_scheme == "imbalance_side":
        return f"{drift_text}: {group_text}"
    if grouping_scheme == "volume_level":
        return f"{drift_text}: {group_text}"
    if grouping_scheme == "both_above_below_high":
        return f"{drift_text}: {group_text}"
    return f"{drift_text}: {group_text}"


def _qtag(q: float) -> str:
    return f"q{int(round(q * 100))}"


def _active_abnormal_q() -> float:
    return float(INTRADAY_ABNORMAL_QS[-1])


def _active_abnormal_q_label() -> int:
    return int(round(_active_abnormal_q() * 100))


def _active_lower_q() -> float:
    return float(1.0 - _active_abnormal_q())


def _derive_ticker_path(output_path: str | Path, ticker: str) -> Path:
    output_path = Path(output_path)
    stem = output_path.stem
    suffix = output_path.suffix or ".pdf"
    return output_path.with_name(f"{stem}_{ticker}{suffix}")


def _report_marker_path(output_pdf: str | Path) -> Path:
    output_pdf = Path(output_pdf)
    return output_pdf.with_suffix(output_pdf.suffix + f".{REPORT_MODE_TAG}.done")


def _report_image_dir(output_pdf: str | Path) -> Path:
    output_pdf = Path(output_pdf)
    return output_pdf.with_name(f"{output_pdf.stem}_images")


def _slugify_plot_name(value: str) -> str:
    clean = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
    clean = "_".join(part for part in clean.split("_") if part)
    return clean[:140] or "figure"


def _save_report_figure(fig: plt.Figure, image_dir: Path | None, stem: str) -> None:
    if image_dir is None:
        return
    image_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(image_dir / f"{_slugify_plot_name(stem)}.png", dpi=PLOT_IMAGE_DPI, bbox_inches="tight")


def _write_report_marker(output_pdf: str | Path, ticker: str) -> None:
    marker = _report_marker_path(output_pdf)
    marker.write_text(f"{REPORT_MODE_TAG}\n{str(ticker).upper().strip()}\n", encoding="utf-8")


def _is_report_complete(output_pdf: str | Path) -> bool:
    output_pdf = Path(output_pdf)
    marker = _report_marker_path(output_pdf)
    return output_pdf.exists() and marker.exists()


def _pretty_drift_label(drift_sign: object) -> str:
    mapping = {
        "flat": "flat drift",
        "drifting": "drifting",
        "noise_dominant": "non-drifting",
        "non_drifting": "non-drifting",
        "missing": "missing drift",
    }
    return mapping.get(str(drift_sign), str(drift_sign).replace("_", " "))


def _drift_title(drift_sign: object) -> str:
    mapping = {
        "flat": "Flat Drift",
        "drifting": "Drifting",
        "noise_dominant": "Non-Drifting",
        "non_drifting": "Non-Drifting",
    }
    return mapping.get(str(drift_sign), str(drift_sign).replace("_", " ").title())


def _future_path_state_title(drift_sign: object) -> str:
    mapping = {
        "drifting": "Drift detected",
        "noise_dominant": "No drift detected",
        "non_drifting": "No drift detected",
        "flat": "No drift detected",
    }
    return mapping.get(str(drift_sign), str(drift_sign).replace("_", " ").title())


def _future_path_group_label(group_value: object) -> str:
    mapping = {
        "above_mid_only": "Above-mid abnormal volume",
        "below_mid_only": "Below-mid abnormal volume",
        "none": "No abnormal volume",
    }
    return mapping.get(str(group_value), _display_label(group_value))


def _ordered_drift_groups(df: pd.DataFrame) -> list[str]:
    preferred = [
        "drifting",
        "non_drifting",
        "noise_dominant",
        "flat",
    ]
    present = [str(x) for x in pd.Series(df.get("drift_sign", pd.Series(dtype="object"))).dropna().unique()]
    ordered = [x for x in preferred if x in present]
    ordered.extend(x for x in sorted(present) if x not in ordered)
    return ordered


def _safe_total_volume(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in SIDE_VOL_COLS if c in df.columns]
    if not cols:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return df[cols].apply(pd.to_numeric, errors="coerce").sum(axis=1, min_count=1)


def _safe_log_returns(price_s: pd.Series) -> pd.Series:
    px = pd.to_numeric(price_s, errors="coerce").astype("float64")
    px = px.where(px > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(px / px.shift(1))


def _safe_log_vwap_levels(price_s: pd.Series | np.ndarray) -> pd.Series:
    px = pd.to_numeric(pd.Series(price_s), errors="coerce").astype("float64")
    px = px.where(px > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_px = np.log(px)
    return pd.Series(log_px, index=px.index, dtype="float64")


def _log_vwap_increments_from_levels(log_vwap_levels: pd.Series | np.ndarray) -> pd.Series:
    x = pd.to_numeric(pd.Series(log_vwap_levels), errors="coerce").astype("float64")
    return x.diff()


def summarize_log_return_stats(
    price_s: pd.Series | np.ndarray,
    delta: float = 1.0,
    min_returns: int = MIN_ESTIMATION_RETURNS,
) -> dict[str, float]:
    log_levels = _safe_log_vwap_levels(price_s)
    returns = _log_vwap_increments_from_levels(log_levels).dropna()
    returns = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    n = int(len(returns))
    if n < min_returns or delta <= 0:
        return {"mu_hat": np.nan, "sigma_hat": np.nan, "n_obs": n}
    mean_ret = float(returns.mean())
    centered = returns - mean_ret
    sigma_hat = np.sqrt(np.mean(centered.to_numpy() ** 2) / delta) if n > 0 else np.nan
    return {
        "mu_hat": mean_ret / delta,
        "sigma_hat": float(sigma_hat) if pd.notna(sigma_hat) else np.nan,
        "n_obs": n,
    }


def fit_ait_sahalia_p1_mle_from_log_vwap_increments(
    log_vwap_increments: pd.Series | np.ndarray,
    delta: float = 1.0,
) -> dict[str, float]:
    # Local continuous-time benchmark:
    #   dX_t = mu dt + sigma dW_t
    # where the modeled state X_t is the log VWAP level, not a return series used as the state.
    #
    # For discrete observations X_0, ..., X_n with sampling gap Delta, we use the
    # Historical compatibility helper for the old diffusion-style parameterization:
    #
    #   p_X^(1)(Delta, x | x0; mu, sigma)
    #   = 1 / (sqrt(2*pi*Delta) * sigma)
    #     * exp(-(x - x0 - mu*Delta)^2 / (2*sigma^2*Delta)).
    #
    # Applying this approximation to log-VWAP level increments
    #   Delta X_i = X_i - X_{i-1},
    # the local log-likelihood is:
    #
    #   ell_n^(1)(mu, sigma)
    #   = sum_i [ -log(sigma) - 0.5*log(2*pi*Delta)
    #             - (Delta X_i - mu*Delta)^2 / (2*sigma^2*Delta) ].
    #
    # Under the local constant-parameter specification, the maximizer is the same
    # closed-form Gaussian increment summary:
    #
    #   mu_hat = (1 / (n*Delta)) * sum_i Delta X_i
    #   sigma_hat^2 = (1 / (n*Delta)) * sum_i (Delta X_i - mu_hat*Delta)^2.
    #
    # This keeps the efficient implementation while making the theory explicit.
    r = pd.Series(log_vwap_increments, dtype="float64").replace([np.inf, -np.inf], np.nan).dropna()
    n = len(r)
    if n == 0:
        return {"mu_hat": np.nan, "sigma_hat": np.nan, "n_obs": 0}
    mu_hat = r.mean() / delta
    centered = r - r.mean()
    sigma_hat = np.sqrt(np.mean(centered.to_numpy() ** 2) / delta) if n > 0 else np.nan
    return {"mu_hat": float(mu_hat), "sigma_hat": float(sigma_hat), "n_obs": int(n)}


def fit_ait_sahalia_p1_mle_from_log_vwap_levels(
    price_s: pd.Series | np.ndarray,
    delta: float = 1.0,
    min_returns: int = MIN_ESTIMATION_RETURNS,
) -> dict[str, float]:
    log_vwap_levels = _safe_log_vwap_levels(price_s)
    log_vwap_increments = _log_vwap_increments_from_levels(log_vwap_levels)
    # Explicitly verify that estimation is performed on log-VWAP level increments X_i - X_{i-1}.
    out = fit_ait_sahalia_p1_mle_from_log_vwap_increments(log_vwap_increments, delta=delta)
    if out["n_obs"] < min_returns:
        return {"mu_hat": np.nan, "sigma_hat": np.nan, "n_obs": int(out["n_obs"])}
    return out


def fit_no_jump_mle_from_returns(returns: pd.Series | np.ndarray, delta: float = 1.0) -> dict[str, float]:
    # Backward-compatible alias: the implementation matches the first-order
    # Historical compatibility wrapper once returns are interpreted as log-VWAP increments.
    return fit_ait_sahalia_p1_mle_from_log_vwap_increments(returns, delta=delta)


def fit_no_jump_mle_from_prices(price_s: pd.Series | np.ndarray, delta: float = 1.0, min_returns: int = MIN_ESTIMATION_RETURNS) -> dict[str, float]:
    # Backward-compatible alias: prices are transformed to log-VWAP levels and
    # estimation is applied to their discrete increments.
    return fit_ait_sahalia_p1_mle_from_log_vwap_levels(price_s, delta=delta, min_returns=min_returns)


def _summary_se(s: pd.Series) -> float:
    x = pd.to_numeric(s, errors="coerce").dropna()
    if len(x) < 2:
        return np.nan
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def _copy_prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["minute_dt"] = pd.to_datetime(out["minute_dt"])
    out = out.sort_values("minute_dt").reset_index(drop=True)
    if "date" not in out.columns:
        out["date"] = out["minute_dt"].dt.date
    out[PRICE_COL] = pd.to_numeric(out[PRICE_COL], errors="coerce")
    for col in SIDE_VOL_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out["total_volume"] = _safe_total_volume(out)
    out["log_vwap_level"] = _safe_log_vwap_levels(out[PRICE_COL])
    out["log_ret_vwap"] = _safe_log_returns(out[PRICE_COL])
    out["time_m"] = out["minute_dt"].dt.strftime("%H:%M:%S")
    return out


def _drop_known_split_artifacts(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = df.copy()
    if str(ticker).upper() == "WMT":
        split_effective = pd.Timestamp("2024-02-26")
        minute_dt = pd.to_datetime(out["minute_dt"])
        pre_split_mask = minute_dt < split_effective
        out.loc[pre_split_mask, PRICE_COL] = pd.to_numeric(out.loc[pre_split_mask, PRICE_COL], errors="coerce") / 3.0
    return out.reset_index(drop=True)


def _drop_extreme_daily_price_jump_artifacts(
    df: pd.DataFrame,
    jump_threshold: float = 0.50,
    median_window_days: int = 11,
) -> pd.DataFrame:
    if len(df) == 0:
        return df.copy()
    daily = build_daily_vwap_observations(df)
    if len(daily) < 5 or "daily_vwap" not in daily.columns:
        return df.reset_index(drop=True)

    px = pd.to_numeric(daily["daily_vwap"], errors="coerce").where(lambda s: s > 0)
    jump_mask = px.pct_change().abs() > jump_threshold
    if bool(jump_mask.any()):
        cutoff_date = pd.to_datetime(daily.loc[jump_mask, "date"]).max().date()
        out = _copy_prepare(df)
        out = out[out["date"] >= cutoff_date].copy()
        return out.drop(columns=[c for c in ["total_volume", "log_vwap_level", "log_ret_vwap", "time_m"] if c in out.columns]).reset_index(drop=True)

    log_px = np.log(px)
    window = max(int(median_window_days), 3)
    if window % 2 == 0:
        window += 1
    centered_median = log_px.rolling(window, center=True, min_periods=3).median()
    deviation = (log_px - centered_median).abs()
    artifact_dates = set(
        pd.to_datetime(daily.loc[deviation > np.log1p(jump_threshold), "date"]).dt.date
    )
    if not artifact_dates:
        return df.reset_index(drop=True)

    out = _copy_prepare(df)
    out = out[~out["date"].isin(artifact_dates)].copy()
    return out.drop(columns=[c for c in ["total_volume", "log_vwap_level", "log_ret_vwap", "time_m"] if c in out.columns]).reset_index(drop=True)


def _clean_loaded_dataset(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    out = _drop_known_split_artifacts(df, ticker)
    out = _drop_extreme_daily_price_jump_artifacts(out)
    return out.reset_index(drop=True)


def _daily_return_diffs(price_s: pd.Series) -> pd.Series:
    # Daily one-minute returns are defined in price differences, not log returns:
    #   r_{d,i} = p_{d,i} - p_{d,i-1}
    px = pd.to_numeric(price_s, errors="coerce").astype("float64")
    return px.diff()


def _compute_daily_rsd(returns: pd.Series, lag_k: int, delta_t: float) -> float:
    # Daily realized squared drift at lag k:
    #   RSD_d(k) = (1 / Delta) * sum_{i=k+1}^{N_d} r_{d,i} * r_{d,i-k}
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(r) <= lag_k or delta_t <= 0:
        return np.nan
    return float((r.iloc[lag_k:].to_numpy() * r.iloc[:-lag_k].to_numpy()).sum() / delta_t)


def _compute_daily_realized_quarticity(returns: pd.Series, delta_t: float) -> float:
    # This implementation uses realized quarticity:
    #   RQ_d = (1 / (3 * Delta)) * sum_i r_i^4
    # This is being used as the quarticity estimator for standardizing daily RSD.
    # This is the current required implementation.
    r = pd.to_numeric(returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().astype("float64")
    if len(r) == 0 or delta_t <= 0:
        return np.nan
    return float((r.pow(4).sum()) / (3.0 * delta_t))


def _compute_daily_st_stat(rsd_value: float, rq_value: float, delta_t: float) -> tuple[float, float]:
    # Daily variance estimate:
    #   VarHat_d = RQ_d / Delta
    # Standardized daily drift-strength statistic:
    #   ST_d(k) = RSD_d(k) / sqrt(RQ_d / Delta)
    if pd.isna(rsd_value) or pd.isna(rq_value) or delta_t <= 0:
        return (np.nan, np.nan)
    var_hat = rq_value / delta_t
    if pd.isna(var_hat) or var_hat <= 0:
        return (var_hat, np.nan)
    return (float(var_hat), float(rsd_value / np.sqrt(var_hat)))


def _classify_daily_drift(direction_value: float, st_value: float, valid_day_flag: bool) -> tuple[int, str, str]:
    # The screen decides only whether the day is drifting or not.
    if (not valid_day_flag) or pd.isna(st_value) or abs(st_value) <= DAILY_ST_THRESHOLD:
        return (0, "non_drifting", "non_drifting")
    sign_value = 0 if pd.isna(direction_value) else int(np.sign(direction_value))
    return (sign_value, "drifting", "drifting")


def compute_daily_drift_screen(raw_df: pd.DataFrame) -> pd.DataFrame:
    prepared = _copy_prepare(raw_df)
    segmented = build_segmented_daily_observations(raw_df)
    if len(prepared) == 0:
        return pd.DataFrame()

    last_segment_idx_by_date = {}
    if len(segmented) > 0:
        last_segment_idx_by_date = segmented.groupby("date")["segment_idx"].max().to_dict()

    rows: list[dict[str, object]] = []
    for trading_date, day_df in prepared.groupby("date", sort=True):
        day_df = day_df.sort_values("minute_dt").reset_index(drop=True)
        invalid_reasons: list[str] = []

        if day_df["minute_dt"].duplicated().any():
            invalid_reasons.append("duplicate_timestamps")

        unique_dt = pd.to_datetime(day_df["minute_dt"]).drop_duplicates().sort_values()
        if len(unique_dt) >= 2:
            gap_minutes = unique_dt.diff().dropna().dt.total_seconds().div(60.0)
            if (gap_minutes != 1.0).any():
                invalid_reasons.append("missing_timestamps")

        price_returns = _daily_return_diffs(day_df[PRICE_COL]).iloc[1:]
        nonfinite_returns = ~np.isfinite(pd.to_numeric(price_returns, errors="coerce"))
        if bool(nonfinite_returns.any()):
            invalid_reasons.append("nan_or_inf_returns")

        clean_returns = pd.to_numeric(price_returns, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        n_obs = int(len(clean_returns))
        if n_obs < MIN_MINUTES_PER_DAY:
            invalid_reasons.append("insufficient_minute_bars")

        price_levels = pd.to_numeric(day_df[PRICE_COL], errors="coerce")
        day_stats = summarize_log_return_stats(price_levels, delta=1.0, min_returns=max(20, MIN_MINUTES_PER_DAY // 4))
        mu_hat_d = day_stats["mu_hat"]
        direction_return_d = np.nan
        valid_prices = price_levels.dropna()
        if len(valid_prices) >= 2 and float(valid_prices.iloc[0]) > 0:
            direction_return_d = float(np.log(valid_prices.iloc[-1] / valid_prices.iloc[0]))

        rsd_value = _compute_daily_rsd(clean_returns, DAILY_RSD_LAG_K, DAILY_RSD_DELTA_T)
        rq_value = _compute_daily_realized_quarticity(clean_returns, DAILY_RSD_DELTA_T)
        var_hat, st_value = _compute_daily_st_stat(rsd_value, rq_value, DAILY_RSD_DELTA_T)
        if pd.isna(rq_value) or rq_value <= 0:
            invalid_reasons.append("nonpositive_rq")
        if pd.isna(var_hat) or var_hat <= 0:
            invalid_reasons.append("invalid_variance_denominator")
        if pd.isna(st_value) or not np.isfinite(st_value):
            invalid_reasons.append("invalid_st_stat")

        valid_day_flag = len(invalid_reasons) == 0
        drift_state, drift_label, drift_sign = _classify_daily_drift(direction_return_d, st_value, valid_day_flag)
        sign_mu_hat = 0 if pd.isna(direction_return_d) else int(np.sign(direction_return_d))

        rows.append(
            {
                "trading_date_d": pd.to_datetime(trading_date).date(),
                "date": pd.to_datetime(trading_date).date(),
                "N_d": n_obs,
                "Delta": float(DAILY_RSD_DELTA_T),
                "lag_k": int(DAILY_RSD_LAG_K),
                "mu_hat_d": mu_hat_d,
                "sign_mu_hat_d": sign_mu_hat,
                "abs_mu_hat_d": abs(mu_hat_d) if pd.notna(mu_hat_d) else np.nan,
                "direction_return_d": direction_return_d,
                "RSD_d_k": rsd_value,
                "RQ_d": rq_value,
                "VarHat_d": var_hat,
                "ST_d_k": st_value,
                "abs_ST_d": abs(st_value) if pd.notna(st_value) else np.nan,
                "drift_state_d": int(drift_state),
                "drift_label_d": drift_label,
                "drift_sign": drift_sign,
                "valid_day_flag": bool(valid_day_flag),
                "invalid_reason": ";".join(sorted(set(invalid_reasons))) if invalid_reasons else "",
                "mu_hat": mu_hat_d,
                "sigma_hat_d": day_stats.get("sigma_hat", np.nan),
                "estimation_n_obs_d": int(day_stats.get("n_obs", 0)),
            }
        )

    return pd.DataFrame(rows).sort_values("trading_date_d").reset_index(drop=True)


def _segment_name(time_str: str) -> str | None:
    if "09:30:00" <= time_str < "11:30:00":
        return "seg1_open"
    if "11:30:00" <= time_str < "13:30:00":
        return "seg2_mid"
    if "13:30:00" <= time_str <= "16:00:00":
        return "seg3_close"
    return None


def build_segmented_daily_observations(df: pd.DataFrame) -> pd.DataFrame:
    x = _copy_prepare(df)
    # Daily-frequency observations are built from three fixed intraday VWAP segments.
    x["segment_name"] = x["time_m"].map(_segment_name)
    x = x[x["segment_name"].notna()].copy()
    x["segment_order"] = x["segment_name"].map({"seg1_open": 1, "seg2_mid": 2, "seg3_close": 3})

    def _segment_vwap(g: pd.DataFrame) -> float:
        # Keep the price definition VWAP-based at every layer by aggregating minute VWAPs.
        weights = pd.to_numeric(g["total_volume"], errors="coerce").fillna(0.0)
        prices = pd.to_numeric(g[PRICE_COL], errors="coerce")
        valid = prices.notna()
        if valid.sum() == 0:
            return np.nan
        weights = weights.where(valid, 0.0)
        if weights.sum() > 0:
            return float(np.average(prices[valid], weights=weights[valid]))
        return float(prices[valid].mean())

    grouped = (
        x.groupby(["date", "segment_name", "segment_order"], as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "segment_dt": g["minute_dt"].max(),
                    "segment_vwap": _segment_vwap(g),
                    "segment_total_volume": g["total_volume"].sum(min_count=1),
                    "segment_above_mid_volume": g["trade_volume_above_mid"].sum(min_count=1),
                    "segment_below_mid_volume": g["trade_volume_below_mid"].sum(min_count=1),
                    "segment_at_mid_volume": g["trade_volume_at_mid"].sum(min_count=1),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    grouped = grouped.sort_values(["date", "segment_order"]).reset_index(drop=True)
    grouped["segment_idx"] = np.arange(len(grouped))
    grouped["log_ret_vwap"] = _safe_log_returns(grouped["segment_vwap"])
    return grouped


def build_daily_vwap_observations(df: pd.DataFrame) -> pd.DataFrame:
    x = _copy_prepare(df)

    def _daily_vwap(g: pd.DataFrame) -> float:
        weights = pd.to_numeric(g["total_volume"], errors="coerce").fillna(0.0)
        prices = pd.to_numeric(g[PRICE_COL], errors="coerce")
        valid = prices.notna()
        if valid.sum() == 0:
            return np.nan
        weights = weights.where(valid, 0.0)
        if weights.sum() > 0:
            return float(np.average(prices[valid], weights=weights[valid]))
        return float(prices[valid].mean())

    daily = (
        x.groupby("date", as_index=False)
        .apply(
            lambda g: pd.Series(
                {
                    "date_dt": g["minute_dt"].max(),
                    "daily_vwap": _daily_vwap(g),
                    "daily_total_volume": g["total_volume"].sum(min_count=1),
                }
            ),
            include_groups=False,
        )
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date").reset_index(drop=True)
    daily["day_idx"] = np.arange(len(daily))
    return daily


def build_intraday_group_feature_frame(
    df: pd.DataFrame,
    seq_windows: tuple[int, ...] = INTRADAY_SEQ_WINDOWS,
    lookback: int = INTRADAY_QUANTILE_LOOKBACK,
) -> pd.DataFrame:
    x = _copy_prepare(df)
    min_hist = max(200, lookback // 5)
    new_cols: dict[str, pd.Series] = {}
    q_tag = _qtag(_active_abnormal_q())
    low_tag = _qtag(_active_lower_q())

    above = pd.to_numeric(x["trade_volume_above_mid"], errors="coerce")
    below = pd.to_numeric(x["trade_volume_below_mid"], errors="coerce")
    at_mid = pd.to_numeric(x["trade_volume_at_mid"], errors="coerce")
    total = x["total_volume"]
    imbalance = above - below

    for w in seq_windows:
        total_sum = total.rolling(w, min_periods=w).sum()
        above_sum = above.rolling(w, min_periods=w).sum()
        below_sum = below.rolling(w, min_periods=w).sum()
        at_sum = at_mid.rolling(w, min_periods=w).sum()
        imbalance_sum = imbalance.rolling(w, min_periods=w).sum()

        new_cols[f"roll_total_volume_{w}m"] = total_sum
        new_cols[f"roll_above_mid_volume_{w}m"] = above_sum
        new_cols[f"roll_below_mid_volume_{w}m"] = below_sum
        new_cols[f"roll_at_mid_volume_{w}m"] = at_sum
        new_cols[f"roll_imbalance_diff_{w}m"] = imbalance_sum

        new_cols[f"{q_tag}_total_volume_{w}m"] = total_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_abnormal_q())
        new_cols[f"{q_tag}_above_mid_volume_{w}m"] = above_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_abnormal_q())
        new_cols[f"{q_tag}_below_mid_volume_{w}m"] = below_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_abnormal_q())
        new_cols[f"{q_tag}_at_mid_volume_{w}m"] = at_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_abnormal_q())
        new_cols[f"{q_tag}_imbalance_diff_{w}m"] = imbalance_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_abnormal_q())
        new_cols[f"{low_tag}_imbalance_diff_{w}m"] = imbalance_sum.shift(1).rolling(lookback, min_periods=min_hist).quantile(_active_lower_q())

    return pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()


def _intraday_group_labels(feature_df: pd.DataFrame, idx: int, seq_window: int) -> dict[str, object]:
    # Group logic is explicit so overlapping versus exclusive labels are auditable row by row.
    row = feature_df.iloc[idx]
    total = row.get(f"roll_total_volume_{seq_window}m")
    q_tag = _qtag(_active_abnormal_q())
    low_tag = _qtag(_active_lower_q())
    total_q = row.get(f"{q_tag}_total_volume_{seq_window}m")
    above = row.get(f"roll_above_mid_volume_{seq_window}m")
    below = row.get(f"roll_below_mid_volume_{seq_window}m")
    at_mid = row.get(f"roll_at_mid_volume_{seq_window}m")
    above_q = row.get(f"{q_tag}_above_mid_volume_{seq_window}m")
    below_q = row.get(f"{q_tag}_below_mid_volume_{seq_window}m")
    at_q = row.get(f"{q_tag}_at_mid_volume_{seq_window}m")
    imb = row.get(f"roll_imbalance_diff_{seq_window}m")
    imb_hi = row.get(f"{q_tag}_imbalance_diff_{seq_window}m")
    imb_lo = row.get(f"{low_tag}_imbalance_diff_{seq_window}m")

    is_high_total = pd.notna(total) and pd.notna(total_q) and total >= total_q
    is_high_above = pd.notna(above) and pd.notna(above_q) and above >= above_q
    is_high_below = pd.notna(below) and pd.notna(below_q) and below >= below_q
    is_high_at = pd.notna(at_mid) and pd.notna(at_q) and at_mid >= at_q

    imbalance_side = "neutral"
    if pd.notna(imb) and pd.notna(imb_hi) and imb >= imb_hi:
        imbalance_side = "above_mid_imbalance"
    elif pd.notna(imb) and pd.notna(imb_lo) and imb <= imb_lo:
        imbalance_side = "below_mid_imbalance"

    exclusive_location = "other_abnormal_mix"
    if is_high_below and not is_high_above and not is_high_at:
        exclusive_location = "below_mid_only"
    elif is_high_above and not is_high_below and not is_high_at:
        exclusive_location = "above_mid_only"
    elif (not is_high_total) and (not is_high_above) and (not is_high_below) and (not is_high_at):
        exclusive_location = "none"

    both_above_below = "both_high" if is_high_above and is_high_below else "not_both_high"

    return {
        "is_high_total_volume": bool(is_high_total),
        "volume_level": "high_volume" if is_high_total else "low_volume",
        "imbalance_side": imbalance_side,
        "exclusive_location": exclusive_location,
        "both_above_below_high": both_above_below,
        "is_high_above_mid": bool(is_high_above),
        "is_high_below_mid": bool(is_high_below),
        "is_high_at_mid": bool(is_high_at),
        "group_seq_window_mins": int(seq_window),
    }


def build_daily_group_feature_frame(
    df: pd.DataFrame,
    abnormal_qs: tuple[float, ...] = INTRADAY_ABNORMAL_QS,
    lookback_days: int = DAILY_QUANTILE_LOOKBACK_DAYS,
) -> pd.DataFrame:
    daily = base.build_daily_panel_for_excursion(df=df, price_col=PRICE_COL, vol_cols=SIDE_VOL_COLS)
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily["total_volume"] = _safe_total_volume(daily)
    daily["imbalance_diff"] = daily["trade_volume_above_mid"] - daily["trade_volume_below_mid"]
    min_hist = max(20, lookback_days // 4)

    new_cols: dict[str, pd.Series] = {}
    q_tag = _qtag(_active_abnormal_q())
    low_tag = _qtag(_active_lower_q())
    total = pd.to_numeric(daily["total_volume"], errors="coerce")
    above = pd.to_numeric(daily["trade_volume_above_mid"], errors="coerce")
    below = pd.to_numeric(daily["trade_volume_below_mid"], errors="coerce")
    at_mid = pd.to_numeric(daily["trade_volume_at_mid"], errors="coerce")
    imb = pd.to_numeric(daily["imbalance_diff"], errors="coerce")

    new_cols[f"{q_tag}_total_volume"] = total.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_abnormal_q())
    new_cols[f"{q_tag}_above_mid_volume"] = above.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_abnormal_q())
    new_cols[f"{q_tag}_below_mid_volume"] = below.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_abnormal_q())
    new_cols[f"{q_tag}_at_mid_volume"] = at_mid.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_abnormal_q())
    new_cols[f"{q_tag}_imbalance_diff"] = imb.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_abnormal_q())
    new_cols[f"{low_tag}_imbalance_diff"] = imb.shift(1).rolling(lookback_days, min_periods=min_hist).quantile(_active_lower_q())

    daily = pd.concat([daily, pd.DataFrame(new_cols, index=daily.index)], axis=1).copy()
    daily = base.add_daily_abnormal_flags_multiq(
        daily_df=daily,
        vol_cols=SIDE_VOL_COLS,
        abnormal_qs=abnormal_qs,
        baseline_window=lookback_days,
    )
    return daily


def _daily_group_labels(row: pd.Series) -> dict[str, object]:
    q_tag = _qtag(_active_abnormal_q())
    low_tag = _qtag(_active_lower_q())
    total = row.get("total_volume")
    total_q = row.get(f"{q_tag}_total_volume")
    above = row.get("trade_volume_above_mid")
    below = row.get("trade_volume_below_mid")
    at_mid = row.get("trade_volume_at_mid")
    above_q = row.get(f"{q_tag}_above_mid_volume")
    below_q = row.get(f"{q_tag}_below_mid_volume")
    at_q = row.get(f"{q_tag}_at_mid_volume")
    imb = row.get("imbalance_diff")
    imb_hi = row.get(f"{q_tag}_imbalance_diff")
    imb_lo = row.get(f"{low_tag}_imbalance_diff")

    is_high_total = pd.notna(total) and pd.notna(total_q) and total >= total_q
    is_high_above = pd.notna(above) and pd.notna(above_q) and above >= above_q
    is_high_below = pd.notna(below) and pd.notna(below_q) and below >= below_q
    is_high_at = pd.notna(at_mid) and pd.notna(at_q) and at_mid >= at_q

    imbalance_side = "neutral"
    if pd.notna(imb) and pd.notna(imb_hi) and imb >= imb_hi:
        imbalance_side = "above_mid_imbalance"
    elif pd.notna(imb) and pd.notna(imb_lo) and imb <= imb_lo:
        imbalance_side = "below_mid_imbalance"

    exclusive_location = "other_abnormal_mix"
    if is_high_below and not is_high_above and not is_high_at:
        exclusive_location = "below_mid_only"
    elif is_high_above and not is_high_below and not is_high_at:
        exclusive_location = "above_mid_only"
    elif (not is_high_total) and (not is_high_above) and (not is_high_below) and (not is_high_at):
        exclusive_location = "none"

    return {
        "is_high_total_volume": bool(is_high_total),
        "volume_level": "high_volume" if is_high_total else "low_volume",
        "imbalance_side": imbalance_side,
        "exclusive_location": exclusive_location,
        "both_above_below_high": "both_high" if is_high_above and is_high_below else "not_both_high",
        "is_high_above_mid": bool(is_high_above),
        "is_high_below_mid": bool(is_high_below),
        "is_high_at_mid": bool(is_high_at),
    }


def _is_general_no_abnormal_baseline(labels: dict[str, object]) -> bool:
    return (
        labels.get("exclusive_location") == "none"
        and labels.get("volume_level") == "low_volume"
        and labels.get("imbalance_side") == "neutral"
    )


def _fit_intraday_local(price_s: pd.Series, idx: int) -> dict[str, float]:
    start = max(0, idx - INTRADAY_ESTIMATION_WINDOW_RETURNS)
    px = price_s.iloc[start: idx + 1]
    return fit_ait_sahalia_p1_mle_from_log_vwap_levels(px, delta=1.0, min_returns=MIN_ESTIMATION_RETURNS)


def _fit_intraday_future(price_s: pd.Series, idx: int, horizon: int) -> dict[str, float]:
    end = idx + horizon
    px = price_s.iloc[idx: end + 1]
    return fit_ait_sahalia_p1_mle_from_log_vwap_levels(px, delta=1.0, min_returns=max(5, min(horizon, MIN_ESTIMATION_RETURNS // 3)))


def build_intraday_sequence_level_df(df: pd.DataFrame, ticker: str, daily_drift_df: pd.DataFrame | None = None) -> pd.DataFrame:
    feat = base.prepare_intraday_abnormal_features_multiq(
        df=df,
        price_col=PRICE_COL,
        vol_cols=SIDE_VOL_COLS,
        seq_windows=INTRADAY_SEQ_WINDOWS,
        abnormal_qs=INTRADAY_ABNORMAL_QS,
        baseline_window=INTRADAY_QUANTILE_LOOKBACK,
    )
    seq = base.build_intraday_sequence_table_multiq(
        df_feat=feat,
        vol_cols=SIDE_VOL_COLS,
        seq_windows=INTRADAY_SEQ_WINDOWS,
        abnormal_qs=INTRADAY_ABNORMAL_QS,
        min_run=2,
    )
    if len(seq) == 0:
        return pd.DataFrame()

    group_feat = build_intraday_group_feature_frame(df, seq_windows=INTRADAY_SEQ_WINDOWS, lookback=INTRADAY_QUANTILE_LOOKBACK)
    price_s = pd.to_numeric(group_feat[PRICE_COL], errors="coerce")
    drift_by_date = {}
    if daily_drift_df is not None and len(daily_drift_df) > 0:
        drift_by_date = daily_drift_df.set_index("date").to_dict("index")

    rows: list[dict[str, object]] = []
    for _, event in seq.iterrows():
        if str(event["pattern"]) not in {"trade_volume_below_mid", "trade_volume_above_mid"}:
            continue
        signal_idx = int(event["end_idx"])
        future_start_idx = signal_idx + 1
        local_fit = _fit_intraday_local(price_s, signal_idx)
        if pd.isna(local_fit["mu_hat"]):
            continue
        event_date = pd.to_datetime(event["end_dt"]).date()
        daily_drift = drift_by_date.get(event_date, {})

        row = {
            "timestamp": pd.to_datetime(event["end_dt"]),
            "date": event_date,
            "ticker": ticker,
            "frequency": "intraday",
            "trigger_pattern": event["pattern"],
            "trigger_q": int(event["q_label"]),
            "trigger_window": int(event["seq_window_mins"]),
            "signal_idx": signal_idx,
            "sequence_start_idx": future_start_idx,
            "mu_hat": local_fit["mu_hat"],
            "sigma_hat": local_fit["sigma_hat"],
            "drift_sign": daily_drift.get("drift_label_d", daily_drift.get("drift_sign", "non_drifting")),
            "estimation_n_obs": int(local_fit["n_obs"]),
            "current_window_max_up_pct": event.get("current_window_max_up_pct"),
            "current_window_max_down_pct": event.get("current_window_max_down_pct"),
            "current_window_max_abs_pct": event.get("current_window_max_abs_pct"),
            "mu_hat_d": daily_drift.get("mu_hat_d", np.nan),
            "sign_mu_hat_d": daily_drift.get("sign_mu_hat_d", np.nan),
            "abs_mu_hat_d": daily_drift.get("abs_mu_hat_d", np.nan),
            "RSD_d_k": daily_drift.get("RSD_d_k", np.nan),
            "RQ_d": daily_drift.get("RQ_d", np.nan),
            "VarHat_d": daily_drift.get("VarHat_d", np.nan),
            "ST_d_k": daily_drift.get("ST_d_k", np.nan),
            "abs_ST_d": daily_drift.get("abs_ST_d", np.nan),
            "drift_state_d": daily_drift.get("drift_state_d", 0),
            "drift_label_d": daily_drift.get("drift_label_d", "non_drifting"),
            "valid_day_flag": daily_drift.get("valid_day_flag", False),
            "invalid_reason": daily_drift.get("invalid_reason", ""),
        }
        row.update(_intraday_group_labels(group_feat, signal_idx, int(event["seq_window_mins"])))

        for h in INTRADAY_FUTURE_HORIZONS:
            fut = _fit_intraday_future(price_s, future_start_idx, h)
            future_px = price_s.iloc[future_start_idx: future_start_idx + h + 1]
            bps_metrics = _future_path_bps_metrics(future_px)
            row[f"future_mu_hat_{h}m"] = fut["mu_hat"]
            row[f"future_sigma_hat_{h}m"] = fut["sigma_hat"]
            row[f"future_n_obs_{h}m"] = int(fut["n_obs"])
            row[f"expected_cum_bps_{h}m"] = _expected_cum_bps(local_fit["mu_hat"], h, 1.0)
            row[f"realized_cum_bps_{h}m"] = bps_metrics["realized_cum_bps"]
            row[f"realized_max_up_bps_{h}m"] = bps_metrics["realized_max_up_bps"]
            row[f"realized_max_down_bps_{h}m"] = bps_metrics["realized_max_down_bps"]
            row[f"realized_max_abs_bps_{h}m"] = bps_metrics["realized_max_abs_bps"]

        rows.append(row)

    baseline_window = INTRADAY_BASELINE_SEQ_WINDOW
    for signal_idx in range(baseline_window - 1, len(group_feat), INTRADAY_BASELINE_STRIDE):
        labels = _intraday_group_labels(group_feat, signal_idx, baseline_window)
        if not _is_general_no_abnormal_baseline(labels):
            continue
        future_start_idx = signal_idx + 1
        local_fit = _fit_intraday_local(price_s, signal_idx)
        if pd.isna(local_fit["mu_hat"]):
            continue
        event_date = pd.to_datetime(group_feat.iloc[signal_idx]["minute_dt"]).date()
        daily_drift = drift_by_date.get(event_date, {})

        row = {
            "timestamp": pd.to_datetime(group_feat.iloc[signal_idx]["minute_dt"]),
            "date": event_date,
            "ticker": ticker,
            "frequency": "intraday",
            "trigger_pattern": "baseline_none",
            "trigger_q": _active_abnormal_q_label(),
            "trigger_window": int(baseline_window),
            "signal_idx": int(signal_idx),
            "sequence_start_idx": int(future_start_idx),
            "mu_hat": local_fit["mu_hat"],
            "sigma_hat": local_fit["sigma_hat"],
            "drift_sign": daily_drift.get("drift_label_d", daily_drift.get("drift_sign", "non_drifting")),
            "estimation_n_obs": int(local_fit["n_obs"]),
            "current_window_max_up_pct": np.nan,
            "current_window_max_down_pct": np.nan,
            "current_window_max_abs_pct": np.nan,
            "mu_hat_d": daily_drift.get("mu_hat_d", np.nan),
            "sign_mu_hat_d": daily_drift.get("sign_mu_hat_d", np.nan),
            "abs_mu_hat_d": daily_drift.get("abs_mu_hat_d", np.nan),
            "RSD_d_k": daily_drift.get("RSD_d_k", np.nan),
            "RQ_d": daily_drift.get("RQ_d", np.nan),
            "VarHat_d": daily_drift.get("VarHat_d", np.nan),
            "ST_d_k": daily_drift.get("ST_d_k", np.nan),
            "abs_ST_d": daily_drift.get("abs_ST_d", np.nan),
            "drift_state_d": daily_drift.get("drift_state_d", 0),
            "drift_label_d": daily_drift.get("drift_label_d", "non_drifting"),
            "valid_day_flag": daily_drift.get("valid_day_flag", False),
            "invalid_reason": daily_drift.get("invalid_reason", ""),
        }
        row.update(labels)

        for h in INTRADAY_FUTURE_HORIZONS:
            fut = _fit_intraday_future(price_s, future_start_idx, h)
            future_px = price_s.iloc[future_start_idx: future_start_idx + h + 1]
            bps_metrics = _future_path_bps_metrics(future_px)
            row[f"future_mu_hat_{h}m"] = fut["mu_hat"]
            row[f"future_sigma_hat_{h}m"] = fut["sigma_hat"]
            row[f"future_n_obs_{h}m"] = int(fut["n_obs"])
            row[f"expected_cum_bps_{h}m"] = _expected_cum_bps(local_fit["mu_hat"], h, 1.0)
            row[f"realized_cum_bps_{h}m"] = bps_metrics["realized_cum_bps"]
            row[f"realized_max_up_bps_{h}m"] = bps_metrics["realized_max_up_bps"]
            row[f"realized_max_down_bps_{h}m"] = bps_metrics["realized_max_down_bps"]
            row[f"realized_max_abs_bps_{h}m"] = bps_metrics["realized_max_abs_bps"]

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("timestamp").reset_index(drop=True)


def _fit_daily_local(segmented: pd.DataFrame, segment_idx: int) -> dict[str, float]:
    start = max(0, segment_idx - DAILY_ESTIMATION_WINDOW_RETURNS)
    px = segmented.iloc[start: segment_idx + 1]["segment_vwap"]
    return summarize_log_return_stats(px, delta=1.0 / 3.0, min_returns=MIN_ESTIMATION_RETURNS)


def _fit_daily_future(segmented: pd.DataFrame, segment_idx: int, horizon_days: int) -> dict[str, float]:
    forward_segments = 3 * horizon_days
    end = segment_idx + forward_segments
    px = segmented.iloc[segment_idx: end + 1]["segment_vwap"]
    return summarize_log_return_stats(px, delta=1.0 / 3.0, min_returns=max(2, min(forward_segments, MIN_ESTIMATION_RETURNS // 2)))


def build_daily_sequence_level_df(df: pd.DataFrame, ticker: str, daily_drift_df: pd.DataFrame | None = None) -> pd.DataFrame:
    daily = build_daily_group_feature_frame(df)
    segmented = build_segmented_daily_observations(df)
    daily_vwap = build_daily_vwap_observations(df)
    prepared_raw = _copy_prepare(df)
    if len(daily) == 0 or len(segmented) == 0 or len(daily_vwap) == 0 or len(prepared_raw) == 0:
        return pd.DataFrame()

    last_segment_idx_by_date = segmented.groupby("date")["segment_idx"].max().to_dict()
    day_idx_by_date = {pd.to_datetime(row["date"]).date(): int(row["day_idx"]) for _, row in daily_vwap.iterrows()}
    last_minute_idx_by_date = prepared_raw.reset_index().groupby("date")["index"].max().astype(int).to_dict()
    drift_by_date = {}
    if daily_drift_df is not None and len(daily_drift_df) > 0:
        drift_by_date = daily_drift_df.set_index("date").to_dict("index")

    rows: list[dict[str, object]] = []
    for _, day_row in daily.iterrows():
        event_date = pd.to_datetime(day_row["date"]).date()
        segment_idx = last_segment_idx_by_date.get(event_date)
        signal_day_idx = day_idx_by_date.get(event_date)
        signal_minute_idx = last_minute_idx_by_date.get(event_date)
        if segment_idx is None:
            continue
        if signal_day_idx is None or signal_minute_idx is None:
            continue
        future_start_idx = int(segment_idx) + 1
        future_minute_start_idx = int(signal_minute_idx) + 1

        for pattern in ("trade_volume_below_mid", "trade_volume_above_mid"):
            for q in INTRADAY_ABNORMAL_QS:
                tag = _qtag(q)
                flag_col = f"abn_{pattern}_{tag}"
                if int(day_row.get(flag_col, 0)) != 1:
                    continue

                local_fit = _fit_daily_local(segmented, int(segment_idx))
                if pd.isna(local_fit["mu_hat"]):
                    continue
                daily_drift = drift_by_date.get(event_date, {})

                row = {
                    "timestamp": segmented.loc[int(segment_idx), "segment_dt"],
                    "date": event_date,
                    "ticker": ticker,
                    "frequency": "daily",
                    "trigger_pattern": pattern,
                    "trigger_q": int(round(q * 100)),
                    "trigger_window": 1,
                    "signal_day_idx": int(signal_day_idx),
                    "signal_idx": int(segment_idx),
                    "signal_minute_idx": int(signal_minute_idx),
                    "future_minute_start_idx": int(future_minute_start_idx),
                    "sequence_start_idx": future_start_idx,
                    "mu_hat": local_fit["mu_hat"],
                    "sigma_hat": local_fit["sigma_hat"],
                    "drift_sign": daily_drift.get("drift_label_d", daily_drift.get("drift_sign", "non_drifting")),
                    "estimation_n_obs": int(local_fit["n_obs"]),
                    "current_window_max_up_pct": day_row.get("current_window_max_up_pct"),
                    "current_window_max_down_pct": day_row.get("current_window_max_down_pct"),
                    "current_window_max_abs_pct": day_row.get("current_window_max_abs_pct"),
                    "mu_hat_d": daily_drift.get("mu_hat_d", local_fit["mu_hat"]),
                    "sign_mu_hat_d": daily_drift.get("sign_mu_hat_d", 0 if pd.isna(local_fit["mu_hat"]) else int(np.sign(local_fit["mu_hat"]))),
                    "abs_mu_hat_d": daily_drift.get("abs_mu_hat_d", abs(local_fit["mu_hat"]) if pd.notna(local_fit["mu_hat"]) else np.nan),
                    "RSD_d_k": daily_drift.get("RSD_d_k", np.nan),
                    "RQ_d": daily_drift.get("RQ_d", np.nan),
                    "VarHat_d": daily_drift.get("VarHat_d", np.nan),
                    "ST_d_k": daily_drift.get("ST_d_k", np.nan),
                    "abs_ST_d": daily_drift.get("abs_ST_d", np.nan),
                    "drift_state_d": daily_drift.get("drift_state_d", 0),
                    "drift_label_d": daily_drift.get("drift_label_d", "non_drifting"),
                    "valid_day_flag": daily_drift.get("valid_day_flag", False),
                    "invalid_reason": daily_drift.get("invalid_reason", ""),
                }
                row.update(_daily_group_labels(day_row))

                for h in DAILY_FUTURE_HORIZONS:
                    fut = _fit_daily_future(segmented, future_start_idx, h)
                    future_day_px = daily_vwap.iloc[int(signal_day_idx): int(signal_day_idx) + h + 1]["daily_vwap"]
                    bps_metrics = _future_path_bps_metrics(future_day_px)
                    row[f"future_mu_hat_{h}d"] = fut["mu_hat"]
                    row[f"future_sigma_hat_{h}d"] = fut["sigma_hat"]
                    row[f"future_n_obs_{h}d"] = int(fut["n_obs"])
                    row[f"expected_cum_bps_{h}d"] = _expected_cum_bps(local_fit["mu_hat"], 3 * h, 1.0 / 3.0)
                    row[f"realized_cum_bps_{h}d"] = bps_metrics["realized_cum_bps"]
                    row[f"realized_max_up_bps_{h}d"] = bps_metrics["realized_max_up_bps"]
                    row[f"realized_max_down_bps_{h}d"] = bps_metrics["realized_max_down_bps"]
                    row[f"realized_max_abs_bps_{h}d"] = bps_metrics["realized_max_abs_bps"]

                rows.append(row)

    for _, day_row in daily.iterrows():
        labels = _daily_group_labels(day_row)
        if not _is_general_no_abnormal_baseline(labels):
            continue
        event_date = pd.to_datetime(day_row["date"]).date()
        segment_idx = last_segment_idx_by_date.get(event_date)
        signal_day_idx = day_idx_by_date.get(event_date)
        signal_minute_idx = last_minute_idx_by_date.get(event_date)
        if segment_idx is None or signal_day_idx is None or signal_minute_idx is None:
            continue
        future_start_idx = int(segment_idx) + 1
        future_minute_start_idx = int(signal_minute_idx) + 1
        local_fit = _fit_daily_local(segmented, int(segment_idx))
        if pd.isna(local_fit["mu_hat"]):
            continue
        daily_drift = drift_by_date.get(event_date, {})

        row = {
            "timestamp": segmented.loc[int(segment_idx), "segment_dt"],
            "date": event_date,
            "ticker": ticker,
            "frequency": "daily",
            "trigger_pattern": "baseline_none",
            "trigger_q": _active_abnormal_q_label(),
            "trigger_window": 1,
            "signal_day_idx": int(signal_day_idx),
            "signal_idx": int(segment_idx),
            "signal_minute_idx": int(signal_minute_idx),
            "future_minute_start_idx": int(future_minute_start_idx),
            "sequence_start_idx": future_start_idx,
            "mu_hat": local_fit["mu_hat"],
            "sigma_hat": local_fit["sigma_hat"],
            "drift_sign": daily_drift.get("drift_label_d", daily_drift.get("drift_sign", "non_drifting")),
            "estimation_n_obs": int(local_fit["n_obs"]),
            "current_window_max_up_pct": np.nan,
            "current_window_max_down_pct": np.nan,
            "current_window_max_abs_pct": np.nan,
            "mu_hat_d": daily_drift.get("mu_hat_d", local_fit["mu_hat"]),
            "sign_mu_hat_d": daily_drift.get("sign_mu_hat_d", 0 if pd.isna(local_fit["mu_hat"]) else int(np.sign(local_fit["mu_hat"]))),
            "abs_mu_hat_d": daily_drift.get("abs_mu_hat_d", abs(local_fit["mu_hat"]) if pd.notna(local_fit["mu_hat"]) else np.nan),
            "RSD_d_k": daily_drift.get("RSD_d_k", np.nan),
            "RQ_d": daily_drift.get("RQ_d", np.nan),
            "VarHat_d": daily_drift.get("VarHat_d", np.nan),
            "ST_d_k": daily_drift.get("ST_d_k", np.nan),
            "abs_ST_d": daily_drift.get("abs_ST_d", np.nan),
            "drift_state_d": daily_drift.get("drift_state_d", 0),
            "drift_label_d": daily_drift.get("drift_label_d", "non_drifting"),
            "valid_day_flag": daily_drift.get("valid_day_flag", False),
            "invalid_reason": daily_drift.get("invalid_reason", ""),
        }
        row.update(labels)

        for h in DAILY_FUTURE_HORIZONS:
            fut = _fit_daily_future(segmented, future_start_idx, h)
            future_day_px = daily_vwap.iloc[int(signal_day_idx): int(signal_day_idx) + h + 1]["daily_vwap"]
            bps_metrics = _future_path_bps_metrics(future_day_px)
            row[f"future_mu_hat_{h}d"] = fut["mu_hat"]
            row[f"future_sigma_hat_{h}d"] = fut["sigma_hat"]
            row[f"future_n_obs_{h}d"] = int(fut["n_obs"])
            row[f"expected_cum_bps_{h}d"] = _expected_cum_bps(local_fit["mu_hat"], 3 * h, 1.0 / 3.0)
            row[f"realized_cum_bps_{h}d"] = bps_metrics["realized_cum_bps"]
            row[f"realized_max_up_bps_{h}d"] = bps_metrics["realized_max_up_bps"]
            row[f"realized_max_down_bps_{h}d"] = bps_metrics["realized_max_down_bps"]
            row[f"realized_max_abs_bps_{h}d"] = bps_metrics["realized_max_abs_bps"]

        rows.append(row)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("timestamp").reset_index(drop=True)


def build_daily_short_signal_table(daily_seq_df: pd.DataFrame) -> pd.DataFrame:
    if len(daily_seq_df) == 0:
        return pd.DataFrame()
    signals = daily_seq_df[
        (daily_seq_df["frequency"] == "daily")
        & (daily_seq_df["exclusive_location"] == "below_mid_only")
    ].copy()
    if len(signals) == 0:
        return signals
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals.sort_values(["date", "timestamp", "trigger_q", "trigger_pattern"]).drop_duplicates(subset=["date"], keep="first")
    return signals.reset_index(drop=True)


def _prepare_daily_backtest_panel(raw_df: pd.DataFrame) -> pd.DataFrame:
    daily_panel = build_daily_group_feature_frame(raw_df).copy()
    if len(daily_panel) == 0:
        return daily_panel
    daily_panel["date"] = pd.to_datetime(daily_panel["date"])
    return daily_panel.sort_values("date").reset_index(drop=True)


def _restrict_daily_panel_to_recent_five_years(daily_panel: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if len(daily_panel) == 0:
        return daily_panel.copy(), "No data"
    end_date = daily_panel["date"].max()
    start_cut = end_date - pd.DateOffset(years=5)
    restricted = daily_panel[daily_panel["date"] >= start_cut].copy()
    full_years = (end_date - daily_panel["date"].min()).days >= 365 * 5
    note = "Most recent 5 years used" if full_years else "Less than 5 years available; used all available history within the dataset window"
    return restricted.reset_index(drop=True), note


def _build_buy_hold_equity(daily_panel: pd.DataFrame) -> pd.Series:
    if len(daily_panel) == 0:
        return pd.Series(dtype="float64")
    px = pd.to_numeric(daily_panel["close_price"], errors="coerce")
    if len(px.dropna()) == 0:
        return pd.Series(1.0, index=daily_panel["date"], dtype="float64")
    first_px = px.dropna().iloc[0]
    out = px / first_px
    out.index = daily_panel["date"]
    return out.astype("float64")


def run_daily_below_mid_only_short_strategy_for_ticker(
    ticker: str,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    shares: int = DEFAULT_POSITION_SHARES,
    commission_per_trade: float = DEFAULT_COMMISSION_PER_TRADE,
) -> dict[str, object]:
    raw_df = base.load_count_aligned_dataset(
        ticker=ticker,
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
    )
    raw_df = _clean_loaded_dataset(raw_df, ticker)
    daily_seq = build_daily_sequence_level_df(raw_df, ticker=ticker)
    daily_panel = _prepare_daily_backtest_panel(raw_df)
    daily_panel, history_note = _restrict_daily_panel_to_recent_five_years(daily_panel)

    summary_cols = {
        "asset": ticker,
        "number_of_trades": 0,
        "mean_trade_return_gross": np.nan,
        "mean_trade_return_net": np.nan,
        "median_trade_return_net": np.nan,
        "win_rate_gross": np.nan,
        "win_rate_net": np.nan,
        "cumulative_return_gross": np.nan,
        "cumulative_return_net": np.nan,
        "annualized_return_net": np.nan,
        "max_drawdown_net": np.nan,
        "avg_holding_days": np.nan,
        "total_commission_cost": 0.0,
        "total_spread_cost": 0.0,
        "total_trading_cost": 0.0,
        "first_signal_date": pd.NaT,
        "last_signal_date": pd.NaT,
        "history_window_note": history_note,
        "position_size_shares": shares,
    }
    if len(daily_panel) == 0:
        return {"trade_log": pd.DataFrame(), "summary": pd.DataFrame([summary_cols]), "equity_curve": pd.DataFrame(), "history_note": history_note}

    signals = build_daily_short_signal_table(daily_seq)
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals[signals["date"].isin(set(daily_panel["date"]))].copy()
    date_to_idx = {d: i for i, d in enumerate(daily_panel["date"])}
    prices = pd.to_numeric(daily_panel["close_price"], errors="coerce").astype("float64")
    initial_capital = float(prices.iloc[0] * shares) if len(prices) > 0 else np.nan

    next_available_idx = 0
    trades: list[dict[str, object]] = []
    for _, sig in signals.sort_values(["date", "timestamp"]).iterrows():
        entry_date = pd.to_datetime(sig["date"])
        if entry_date not in date_to_idx:
            continue
        entry_idx = date_to_idx[entry_date]
        if entry_idx < next_available_idx:
            continue

        entry_row = daily_panel.iloc[entry_idx]
        entry_price = float(entry_row["close_price"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        exit_idx = None
        exit_reason = "end_of_data"
        for j in range(entry_idx + 1, len(daily_panel)):
            px_j = float(daily_panel.iloc[j]["close_price"])
            if not np.isfinite(px_j) or px_j <= 0:
                continue
            short_return_j = (entry_price - px_j) / entry_price
            if short_return_j >= SHORT_STRATEGY_TARGET_BPS / 10000.0:
                exit_idx = j
                exit_reason = "take_profit_200bps"
                break
            if short_return_j <= -SHORT_STRATEGY_STOP_BPS / 10000.0:
                exit_idx = j
                exit_reason = "stop_loss_200bps"
                break
        if exit_idx is None:
            exit_idx = len(daily_panel) - 1
        if exit_idx <= entry_idx:
            continue

        exit_row = daily_panel.iloc[exit_idx]
        exit_price = float(exit_row["close_price"])
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue

        gross_pnl = (entry_price - exit_price) * shares
        gross_return = gross_pnl / (entry_price * shares)
        # Long baseline flips to short at entry and flips back to long at exit.
        commission_cost = 4.0 * float(commission_per_trade)
        spread_cost = 4.0 * SPREAD_COST_PER_SHARE_PER_SIDE * shares
        total_cost = commission_cost + spread_cost
        net_pnl = gross_pnl - total_cost
        net_return = net_pnl / (entry_price * shares)

        trades.append(
            {
                "asset": ticker,
                "entry_date": entry_row["date"],
                "exit_date": exit_row["date"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "gross_pnl_dollars": gross_pnl,
                "commission_cost_dollars": commission_cost,
                "spread_cost_dollars": spread_cost,
                "total_cost_dollars": total_cost,
                "net_pnl_dollars": net_pnl,
                "gross_return": gross_return,
                "net_return": net_return,
                "holding_days": int(exit_idx - entry_idx),
                "exit_reason": exit_reason,
                "signal_timestamp": sig["timestamp"],
                "signal_trigger_pattern": sig["trigger_pattern"],
                "signal_trigger_q": sig["trigger_q"],
                "drift_sign": sig["drift_sign"],
                "exclusive_location": sig["exclusive_location"],
            }
        )
        next_available_idx = exit_idx + 1

    trade_log = pd.DataFrame(trades)
    equity_index = daily_panel["date"]
    gross_equity_dollars = pd.Series(np.nan, index=equity_index, dtype="float64")
    net_equity_dollars = pd.Series(np.nan, index=equity_index, dtype="float64")
    if len(daily_panel) > 0:
        gross_equity_dollars.iloc[0] = initial_capital
        net_equity_dollars.iloc[0] = initial_capital
        entry_to_trade = {pd.to_datetime(t["entry_date"]): t for t in trades}
        exit_to_trade = {pd.to_datetime(t["exit_date"]): t for t in trades}
        short_active = False

        for i in range(len(daily_panel) - 1):
            date_i = pd.to_datetime(daily_panel.iloc[i]["date"])
            px_i = float(prices.iloc[i])
            px_next = float(prices.iloc[i + 1])
            if date_i in entry_to_trade:
                net_equity_dollars.iloc[i] -= entry_to_trade[date_i]["commission_cost_dollars"] / 2.0
                net_equity_dollars.iloc[i] -= entry_to_trade[date_i]["spread_cost_dollars"] / 2.0
                short_active = True

            position = -1.0 if short_active else 1.0
            pnl_interval = position * shares * (px_next - px_i)
            gross_equity_dollars.iloc[i + 1] = gross_equity_dollars.iloc[i] + pnl_interval
            net_equity_dollars.iloc[i + 1] = net_equity_dollars.iloc[i] + pnl_interval

            date_next = pd.to_datetime(daily_panel.iloc[i + 1]["date"])
            if date_next in exit_to_trade:
                net_equity_dollars.iloc[i + 1] -= exit_to_trade[date_next]["commission_cost_dollars"] / 2.0
                net_equity_dollars.iloc[i + 1] -= exit_to_trade[date_next]["spread_cost_dollars"] / 2.0
                short_active = False

    gross_equity = gross_equity_dollars / initial_capital if initial_capital and np.isfinite(initial_capital) else pd.Series(1.0, index=equity_index, dtype="float64")
    net_equity = net_equity_dollars / initial_capital if initial_capital and np.isfinite(initial_capital) else pd.Series(1.0, index=equity_index, dtype="float64")

    benchmark = _build_buy_hold_equity(daily_panel)
    equity_curve = pd.DataFrame(
        {
            "date": equity_index,
            "gross_strategy_equity": gross_equity.values,
            "net_strategy_equity": net_equity.values,
            "benchmark_equity": benchmark.reindex(equity_index).values if len(benchmark) else np.nan,
        }
    )

    if len(trade_log) > 0:
        peak = equity_curve["net_strategy_equity"].cummax()
        max_drawdown = ((equity_curve["net_strategy_equity"] / peak) - 1.0).min()
        n_days = max(len(equity_curve), 1)
        annualized = equity_curve["net_strategy_equity"].iloc[-1] ** (252.0 / n_days) - 1.0 if equity_curve["net_strategy_equity"].iloc[-1] > 0 else np.nan
        summary_cols.update(
            {
                "number_of_trades": int(len(trade_log)),
                "mean_trade_return_gross": trade_log["gross_return"].mean(),
                "mean_trade_return_net": trade_log["net_return"].mean(),
                "median_trade_return_net": trade_log["net_return"].median(),
                "win_rate_gross": (trade_log["gross_pnl_dollars"] > 0).mean(),
                "win_rate_net": (trade_log["net_pnl_dollars"] > 0).mean(),
                "cumulative_return_gross": equity_curve["gross_strategy_equity"].iloc[-1] - 1.0,
                "cumulative_return_net": equity_curve["net_strategy_equity"].iloc[-1] - 1.0,
                "annualized_return_net": annualized,
                "max_drawdown_net": max_drawdown,
                "avg_holding_days": trade_log["holding_days"].mean(),
                "total_commission_cost": trade_log["commission_cost_dollars"].sum(),
                "total_spread_cost": trade_log["spread_cost_dollars"].sum(),
                "total_trading_cost": trade_log["total_cost_dollars"].sum(),
                "first_signal_date": trade_log["entry_date"].min(),
                "last_signal_date": trade_log["entry_date"].max(),
            }
        )
    else:
        summary_cols.update(
            {
                "cumulative_return_gross": equity_curve["gross_strategy_equity"].iloc[-1] - 1.0,
                "cumulative_return_net": equity_curve["net_strategy_equity"].iloc[-1] - 1.0,
                "max_drawdown_net": 0.0,
            }
        )

    return {
        "trade_log": trade_log,
        "summary": pd.DataFrame([summary_cols]),
        "equity_curve": equity_curve,
        "history_note": history_note,
    }


def _build_equity_curve_from_regime_trades(
    daily_panel: pd.DataFrame,
    trades: list[dict[str, object]],
    shares: int,
    initial_capital: float,
    position_during_trade: float,
) -> pd.Series:
    dates = daily_panel["date"]
    prices = pd.to_numeric(daily_panel["close_price"], errors="coerce").astype("float64")
    equity_dollars = pd.Series(np.nan, index=dates, dtype="float64")
    equity_dollars.iloc[0] = initial_capital
    entry_map = {pd.to_datetime(t["entry_date"]): t for t in trades}
    exit_map = {pd.to_datetime(t["exit_date"]): t for t in trades}
    active = False

    for i in range(len(daily_panel) - 1):
        date_i = pd.to_datetime(daily_panel.iloc[i]["date"])
        px_i = float(prices.iloc[i])
        px_next = float(prices.iloc[i + 1])
        if date_i in entry_map:
            equity_dollars.iloc[i] -= entry_map[date_i]["entry_cost_dollars"]
            active = True

        position = position_during_trade if active else 1.0
        pnl_interval = position * shares * (px_next - px_i)
        equity_dollars.iloc[i + 1] = equity_dollars.iloc[i] + pnl_interval

        date_next = pd.to_datetime(daily_panel.iloc[i + 1]["date"])
        if date_next in exit_map:
            equity_dollars.iloc[i + 1] -= exit_map[date_next]["exit_cost_dollars"]
            active = False

    return equity_dollars / initial_capital


def _build_short_overlay_trades(
    ticker: str,
    daily_panel: pd.DataFrame,
    signals: pd.DataFrame,
    shares: int,
    commission_per_trade: float,
) -> list[dict[str, object]]:
    date_to_idx = {d: i for i, d in enumerate(daily_panel["date"])}
    next_available_idx = 0
    trades: list[dict[str, object]] = []

    for _, sig in signals.sort_values(["date", "timestamp"]).iterrows():
        entry_date = pd.to_datetime(sig["date"])
        if entry_date not in date_to_idx:
            continue
        entry_idx = date_to_idx[entry_date]
        if entry_idx < next_available_idx:
            continue

        entry_price = float(daily_panel.iloc[entry_idx]["close_price"])
        exit_idx = None
        exit_reason = "end_of_data"
        for j in range(entry_idx + 1, len(daily_panel)):
            px_j = float(daily_panel.iloc[j]["close_price"])
            if not np.isfinite(px_j) or px_j <= 0:
                continue
            short_return_j = (entry_price - px_j) / entry_price
            if short_return_j >= SHORT_STRATEGY_TARGET_BPS / 10000.0:
                exit_idx = j
                exit_reason = "take_profit_200bps"
                break
            if short_return_j <= -SHORT_STRATEGY_STOP_BPS / 10000.0:
                exit_idx = j
                exit_reason = "stop_loss_200bps"
                break
        if exit_idx is None:
            exit_idx = len(daily_panel) - 1
        if exit_idx <= entry_idx:
            continue

        exit_price = float(daily_panel.iloc[exit_idx]["close_price"])
        gross_pnl = (entry_price - exit_price) * shares
        # Overlay trade mechanics:
        # long -> short at entry requires two executions (sell long, sell short),
        # and short -> long at exit requires two executions (buy to cover, buy long).
        entry_commission = 2.0 * float(commission_per_trade)
        exit_commission = 2.0 * float(commission_per_trade)
        entry_spread = 2.0 * SPREAD_COST_PER_SHARE_PER_SIDE * shares
        exit_spread = 2.0 * SPREAD_COST_PER_SHARE_PER_SIDE * shares
        entry_cost = entry_commission + entry_spread
        exit_cost = exit_commission + exit_spread
        total_cost = entry_cost + exit_cost
        net_pnl = gross_pnl - total_cost
        net_return = net_pnl / (entry_price * shares)

        trades.append(
            {
                "strategy_name": "short_overlay_tp_sl",
                "asset": ticker,
                "entry_date": daily_panel.iloc[entry_idx]["date"],
                "exit_date": daily_panel.iloc[exit_idx]["date"],
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "commission_cost_dollars": entry_commission + exit_commission,
                "spread_cost_dollars": entry_spread + exit_spread,
                "entry_cost_dollars": entry_cost,
                "exit_cost_dollars": exit_cost,
                "total_cost_dollars": total_cost,
                "net_pnl_dollars": net_pnl,
                "net_return": net_return,
                "holding_days": int(exit_idx - entry_idx),
                "exit_reason": exit_reason,
                "signal_timestamp": sig["timestamp"],
                "signal_trigger_pattern": sig["trigger_pattern"],
                "signal_trigger_q": sig["trigger_q"],
                "drift_sign": sig["drift_sign"],
                "exclusive_location": sig["exclusive_location"],
            }
        )
        next_available_idx = exit_idx + 1
    return trades


def _build_sell_rebuy_3d_trades(
    ticker: str,
    daily_panel: pd.DataFrame,
    signals: pd.DataFrame,
    shares: int,
    commission_per_trade: float,
) -> list[dict[str, object]]:
    date_to_idx = {d: i for i, d in enumerate(daily_panel["date"])}
    next_available_idx = 0
    trades: list[dict[str, object]] = []

    for _, sig in signals.sort_values(["date", "timestamp"]).iterrows():
        entry_date = pd.to_datetime(sig["date"])
        if entry_date not in date_to_idx:
            continue
        entry_idx = date_to_idx[entry_date]
        if entry_idx < next_available_idx:
            continue
        exit_idx = min(entry_idx + 3, len(daily_panel) - 1)
        if exit_idx <= entry_idx:
            continue

        entry_price = float(daily_panel.iloc[entry_idx]["close_price"])
        exit_price = float(daily_panel.iloc[exit_idx]["close_price"])
        relative_pnl = (entry_price - exit_price) * shares
        entry_commission = float(commission_per_trade)
        exit_commission = float(commission_per_trade)
        entry_spread = SPREAD_COST_PER_SHARE_PER_SIDE * shares
        exit_spread = SPREAD_COST_PER_SHARE_PER_SIDE * shares
        entry_cost = entry_commission + entry_spread
        exit_cost = exit_commission + exit_spread
        total_cost = entry_cost + exit_cost
        net_pnl = relative_pnl - total_cost
        net_return = net_pnl / (entry_price * shares)

        trades.append(
            {
                "strategy_name": "sell_wait_3d_rebuy",
                "asset": ticker,
                "entry_date": daily_panel.iloc[entry_idx]["date"],
                "exit_date": daily_panel.iloc[exit_idx]["date"],
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "shares": shares,
                "commission_cost_dollars": entry_commission + exit_commission,
                "spread_cost_dollars": entry_spread + exit_spread,
                "entry_cost_dollars": entry_cost,
                "exit_cost_dollars": exit_cost,
                "total_cost_dollars": total_cost,
                "net_pnl_dollars": net_pnl,
                "net_return": net_return,
                "holding_days": int(exit_idx - entry_idx),
                "exit_reason": "fixed_3d_rebuy",
                "signal_timestamp": sig["timestamp"],
                "signal_trigger_pattern": sig["trigger_pattern"],
                "signal_trigger_q": sig["trigger_q"],
                "drift_sign": sig["drift_sign"],
                "exclusive_location": sig["exclusive_location"],
            }
        )
        next_available_idx = exit_idx + 1
    return trades


def _summarize_net_strategy(
    asset: str,
    strategy_name: str,
    trade_log: pd.DataFrame,
    net_equity: pd.Series,
    history_note: str,
    shares: int,
) -> dict[str, object]:
    base = {
        "asset": asset,
        "strategy_name": strategy_name,
        "number_of_trades": 0,
        "mean_trade_return_net": np.nan,
        "median_trade_return_net": np.nan,
        "win_rate_net": np.nan,
        "cumulative_return_net": net_equity.iloc[-1] - 1.0 if len(net_equity) else np.nan,
        "annualized_return_net": np.nan,
        "max_drawdown_net": 0.0 if len(net_equity) else np.nan,
        "avg_holding_days": np.nan,
        "total_commission_cost": 0.0,
        "total_spread_cost": 0.0,
        "total_trading_cost": 0.0,
        "first_signal_date": pd.NaT,
        "last_signal_date": pd.NaT,
        "history_window_note": history_note,
        "position_size_shares": shares,
    }
    if len(net_equity) > 0:
        peak = net_equity.cummax()
        base["max_drawdown_net"] = ((net_equity / peak) - 1.0).min()
        n_days = max(len(net_equity), 1)
        if net_equity.iloc[-1] > 0:
            base["annualized_return_net"] = net_equity.iloc[-1] ** (252.0 / n_days) - 1.0
    if len(trade_log) == 0:
        return base
    base.update(
        {
            "number_of_trades": int(len(trade_log)),
            "mean_trade_return_net": trade_log["net_return"].mean(),
            "median_trade_return_net": trade_log["net_return"].median(),
            "win_rate_net": (trade_log["net_pnl_dollars"] > 0).mean(),
            "avg_holding_days": trade_log["holding_days"].mean(),
            "total_commission_cost": trade_log["commission_cost_dollars"].sum(),
            "total_spread_cost": trade_log["spread_cost_dollars"].sum(),
            "total_trading_cost": trade_log["total_cost_dollars"].sum(),
            "first_signal_date": trade_log["entry_date"].min(),
            "last_signal_date": trade_log["entry_date"].max(),
        }
    )
    return base


def run_daily_below_mid_signal_comparison_for_ticker(
    ticker: str,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    shares: int = DEFAULT_POSITION_SHARES,
    commission_per_trade: float = DEFAULT_COMMISSION_PER_TRADE,
) -> dict[str, object]:
    raw_df = base.load_count_aligned_dataset(
        ticker=ticker,
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
    )
    raw_df = _clean_loaded_dataset(raw_df, ticker)
    daily_seq = build_daily_sequence_level_df(raw_df, ticker=ticker)
    daily_panel = _prepare_daily_backtest_panel(raw_df)
    daily_panel, history_note = _restrict_daily_panel_to_recent_five_years(daily_panel)
    if len(daily_panel) == 0:
        empty = pd.DataFrame()
        return {"trade_log": empty, "summary": empty, "equity_curve": empty, "history_note": history_note}

    signals = build_daily_short_signal_table(daily_seq)
    signals["date"] = pd.to_datetime(signals["date"])
    signals = signals[signals["date"].isin(set(daily_panel["date"]))].copy()
    prices = pd.to_numeric(daily_panel["close_price"], errors="coerce").astype("float64")
    initial_capital = float(prices.iloc[0] * shares)

    short_trades = _build_short_overlay_trades(ticker, daily_panel, signals, shares, commission_per_trade)
    flat_trades = _build_sell_rebuy_3d_trades(ticker, daily_panel, signals, shares, commission_per_trade)

    short_net_equity = _build_equity_curve_from_regime_trades(daily_panel, short_trades, shares, initial_capital, position_during_trade=-1.0)
    flat_net_equity = _build_equity_curve_from_regime_trades(daily_panel, flat_trades, shares, initial_capital, position_during_trade=0.0)
    benchmark_equity = _build_buy_hold_equity(daily_panel).reindex(daily_panel["date"]).astype("float64")

    equity_curve = pd.DataFrame(
        {
            "date": daily_panel["date"],
            "short_overlay_net_equity": short_net_equity.values,
            "sell_wait_3d_net_equity": flat_net_equity.values,
            "benchmark_equity": benchmark_equity.values,
        }
    )

    short_log = pd.DataFrame(short_trades)
    flat_log = pd.DataFrame(flat_trades)
    combined_log = pd.concat([short_log, flat_log], ignore_index=True) if len(short_log) or len(flat_log) else pd.DataFrame()

    short_summary = _summarize_net_strategy(ticker, "short_overlay_tp_sl", short_log, short_net_equity, history_note, shares)
    flat_summary = _summarize_net_strategy(ticker, "sell_wait_3d_rebuy", flat_log, flat_net_equity, history_note, shares)
    summary = pd.DataFrame([short_summary, flat_summary])
    return {"trade_log": combined_log, "summary": summary, "equity_curve": equity_curve, "history_note": history_note}


def _attach_grouping_views(seq_df: pd.DataFrame) -> pd.DataFrame:
    if len(seq_df) == 0:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    for scheme in GROUP_SCHEMES:
        sub = seq_df.copy()
        sub["grouping_scheme"] = scheme
        sub["group_value"] = sub[scheme]
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def build_group_summary_long(seq_df: pd.DataFrame) -> pd.DataFrame:
    if len(seq_df) == 0:
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    for (freq, drift_sign, scheme, group_value), grp in (
        seq_df.groupby(["frequency", "drift_sign", "grouping_scheme", "group_value"], dropna=False)
    ):
        if len(grp) == 0:
            continue
        horizons = INTRADAY_FUTURE_HORIZONS if freq == "intraday" else DAILY_FUTURE_HORIZONS
        suffix = "m" if freq == "intraday" else "d"

        for h in horizons:
            mu_col = f"future_mu_hat_{h}{suffix}"
            sig_col = f"future_sigma_hat_{h}{suffix}"
            expected_col = f"expected_cum_bps_{h}{suffix}"
            realized_col = f"realized_cum_bps_{h}{suffix}"
            max_up_col = f"realized_max_up_bps_{h}{suffix}"
            max_down_col = f"realized_max_down_bps_{h}{suffix}"
            max_abs_col = f"realized_max_abs_bps_{h}{suffix}"
            if mu_col not in grp.columns or sig_col not in grp.columns:
                continue
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "frequency": freq,
                            "drift_sign": drift_sign,
                            "grouping_scheme": scheme,
                            "group_value": group_value,
                            "horizon": h,
                            "count": int(len(grp)),
                            "mean_current_mu_hat": grp["mu_hat"].mean(),
                            "se_current_mu_hat": _summary_se(grp["mu_hat"]),
                            "mean_current_sigma_hat": grp["sigma_hat"].mean(),
                            "se_current_sigma_hat": _summary_se(grp["sigma_hat"]),
                            "mean_future_mu_hat": grp[mu_col].mean(),
                            "se_future_mu_hat": _summary_se(grp[mu_col]),
                            "mean_future_sigma_hat": grp[sig_col].mean(),
                            "se_future_sigma_hat": _summary_se(grp[sig_col]),
                            "mean_expected_cum_bps": grp[expected_col].mean() if expected_col in grp.columns else np.nan,
                            "se_expected_cum_bps": _summary_se(grp[expected_col]) if expected_col in grp.columns else np.nan,
                            "mean_realized_cum_bps": grp[realized_col].mean() if realized_col in grp.columns else np.nan,
                            "se_realized_cum_bps": _summary_se(grp[realized_col]) if realized_col in grp.columns else np.nan,
                            "mean_realized_max_up_bps": grp[max_up_col].mean() if max_up_col in grp.columns else np.nan,
                            "se_realized_max_up_bps": _summary_se(grp[max_up_col]) if max_up_col in grp.columns else np.nan,
                            "mean_realized_max_down_bps": grp[max_down_col].mean() if max_down_col in grp.columns else np.nan,
                            "se_realized_max_down_bps": _summary_se(grp[max_down_col]) if max_down_col in grp.columns else np.nan,
                            "mean_realized_max_abs_bps": grp[max_abs_col].mean() if max_abs_col in grp.columns else np.nan,
                            "se_realized_max_abs_bps": _summary_se(grp[max_abs_col]) if max_abs_col in grp.columns else np.nan,
                        }
                    ]
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_compact_summary_table(summary_long: pd.DataFrame, top_n_groups: int = 8) -> pd.DataFrame:
    if len(summary_long) == 0:
        return pd.DataFrame()
    out_rows: list[dict[str, object]] = []

    for (freq, drift_sign, scheme), grp in summary_long.groupby(["frequency", "drift_sign", "grouping_scheme"], dropna=False):
        horizon_key = grp["horizon"].max()
        terminal = grp[grp["horizon"] == horizon_key].sort_values("count", ascending=False).head(top_n_groups)
        for _, row in terminal.iterrows():
            out_rows.append(
                {
                    "frequency": freq,
                    "drift_sign": drift_sign,
                    "grouping_scheme": scheme,
                    "group_value": row["group_value"],
                    "count": int(row["count"]),
                    "mean_current_mu_hat": row["mean_current_mu_hat"],
                    "mean_current_sigma_hat": row["mean_current_sigma_hat"],
                    f"mean_expected_cum_bps_h{horizon_key}": row["mean_expected_cum_bps"],
                    f"mean_realized_cum_bps_h{horizon_key}": row["mean_realized_cum_bps"],
                    f"mean_realized_max_abs_bps_h{horizon_key}": row["mean_realized_max_abs_bps"],
                    f"mean_future_sigma_hat_h{horizon_key}": row["mean_future_sigma_hat"],
                }
            )
    return pd.DataFrame(out_rows)


def build_group_difference_tests(seq_df: pd.DataFrame) -> pd.DataFrame:
    if len(seq_df) == 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for (freq, drift_sign, scheme), grp in seq_df.groupby(["frequency", "drift_sign", "grouping_scheme"], dropna=False):
        grp = grp.copy()
        grp["group_value"] = grp["group_value"].astype(str)
        valid_groups = [g for g, sub in grp.groupby("group_value") if len(sub) > 0]
        if len(valid_groups) < 2:
            continue

        horizons = INTRADAY_FUTURE_HORIZONS if freq == "intraday" else DAILY_FUTURE_HORIZONS
        suffix = "m" if freq == "intraday" else "d"
        metric_specs = [("current", "mu_hat"), ("current", "sigma_hat")]
        metric_specs.extend((h, f"future_mu_hat_{h}{suffix}") for h in horizons)
        metric_specs.extend((h, f"future_sigma_hat_{h}{suffix}") for h in horizons)

        for horizon_label, metric_col in metric_specs:
            if metric_col not in grp.columns:
                continue
            samples = [pd.to_numeric(grp.loc[grp["group_value"] == g, metric_col], errors="coerce").dropna() for g in valid_groups]
            usable = [(g, s) for g, s in zip(valid_groups, samples) if len(s) > 0]
            if len(usable) < 2:
                continue

            if len(usable) > 2:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    omnibus_stat, omnibus_p = stats.f_oneway(*[s for _, s in usable])
                rows.append(
                    {
                        "frequency": freq,
                        "drift_sign": drift_sign,
                        "grouping_scheme": scheme,
                        "metric": metric_col,
                        "horizon": horizon_label,
                        "test_type": "anova_omnibus",
                        "group_a": "ALL",
                        "group_b": "ALL",
                        "mean_a": np.nan,
                        "mean_b": np.nan,
                        "mean_diff": np.nan,
                        "test_stat": float(omnibus_stat) if pd.notna(omnibus_stat) else np.nan,
                        "p_value": float(omnibus_p) if pd.notna(omnibus_p) else np.nan,
                        "n_a": int(sum(len(s) for _, s in usable)),
                        "n_b": int(len(usable)),
                        "significant_5pct": bool(pd.notna(omnibus_p) and omnibus_p < 0.05),
                    }
                )

            for i in range(len(usable)):
                for j in range(i + 1, len(usable)):
                    group_a, sample_a = usable[i]
                    group_b, sample_b = usable[j]
                    test_res = _welch_test(sample_a, sample_b)
                    rows.append(
                        {
                            "frequency": freq,
                            "drift_sign": drift_sign,
                            "grouping_scheme": scheme,
                            "metric": metric_col,
                            "horizon": horizon_label,
                            "test_type": "welch_ttest",
                            "group_a": group_a,
                            "group_b": group_b,
                            **test_res,
                            "significant_5pct": bool(pd.notna(test_res["p_value"]) and test_res["p_value"] < 0.05),
                        }
                    )

    return pd.DataFrame(rows)


def _expected_path(mu_hat: float, horizon_steps: int, delta: float) -> np.ndarray:
    steps = np.arange(horizon_steps + 1, dtype="float64")
    log_path = mu_hat * delta * steps
    return np.expm1(log_path)


def _expected_cum_bps(mu_hat: float, horizon_steps: int, delta: float) -> float:
    if pd.isna(mu_hat):
        return np.nan
    return float(np.expm1(mu_hat * delta * horizon_steps) * 10000.0)


def _brownian_path_bundle(
    mu_hat: float,
    sigma_hat: float,
    horizon_steps: int,
    delta: float,
    seed: int,
    n_paths: int = 20,
) -> dict[str, np.ndarray]:
    steps = np.arange(horizon_steps + 1, dtype="float64")
    mean_log_path = mu_hat * delta * steps
    std_log_path = sigma_hat * np.sqrt(delta * steps)
    lower_95 = np.expm1(mean_log_path - 1.96 * std_log_path)
    upper_95 = np.expm1(mean_log_path + 1.96 * std_log_path)

    rng = np.random.default_rng(seed)
    if horizon_steps <= 0:
        sample_paths = np.zeros((n_paths, 1), dtype="float64")
    else:
        increments = mu_hat * delta + sigma_hat * np.sqrt(delta) * rng.standard_normal((n_paths, horizon_steps))
        sample_log_paths = np.cumsum(np.column_stack([np.zeros(n_paths), increments]), axis=1)
        sample_paths = np.expm1(sample_log_paths)

    return {
        "mean_path": np.expm1(mean_log_path),
        "lower_95": lower_95,
        "upper_95": upper_95,
        "sample_paths": sample_paths,
    }


def _realized_cum_returns(price_s: pd.Series) -> np.ndarray:
    px = pd.to_numeric(price_s, errors="coerce").astype("float64")
    px = px.where(px > 0)
    if px.isna().any() or len(px) < 2:
        return np.array([])
    base_px = px.iloc[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        cum = (px / base_px) - 1.0
    return cum.to_numpy()


def _future_path_bps_metrics(price_s: pd.Series) -> dict[str, float]:
    cum = _realized_cum_returns(price_s)
    if len(cum) == 0:
        return {
            "realized_cum_bps": np.nan,
            "realized_max_up_bps": np.nan,
            "realized_max_down_bps": np.nan,
            "realized_max_abs_bps": np.nan,
        }
    realized_cum_bps = float(cum[-1] * 10000.0)
    realized_max_up_bps = float(np.nanmax(cum) * 10000.0)
    realized_max_down_bps = float(np.nanmin(cum) * 10000.0)
    realized_max_abs_bps = float(np.nanmax(np.abs(cum)) * 10000.0)
    return {
        "realized_cum_bps": realized_cum_bps,
        "realized_max_up_bps": realized_max_up_bps,
        "realized_max_down_bps": realized_max_down_bps,
        "realized_max_abs_bps": realized_max_abs_bps,
    }


def _welch_test(x: pd.Series, y: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(pd.Series(x), errors="coerce").dropna()
    y = pd.to_numeric(pd.Series(y), errors="coerce").dropna()
    if len(x) == 0 or len(y) == 0:
        return {
            "mean_a": np.nan,
            "mean_b": np.nan,
            "mean_diff": np.nan,
            "test_stat": np.nan,
            "p_value": np.nan,
            "n_a": int(len(x)),
            "n_b": int(len(y)),
        }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        stat, p_value = stats.ttest_ind(x, y, equal_var=False, nan_policy="omit")
    return {
        "mean_a": float(x.mean()),
        "mean_b": float(y.mean()),
        "mean_diff": float(x.mean() - y.mean()),
        "test_stat": float(stat) if pd.notna(stat) else np.nan,
        "p_value": float(p_value) if pd.notna(p_value) else np.nan,
        "n_a": int(len(x)),
        "n_b": int(len(y)),
    }


def rank_example_candidates(
    seq_df: pd.DataFrame,
    intraday_price_df: pd.DataFrame,
    segmented_daily_df: pd.DataFrame,
) -> pd.DataFrame:
    # Transparent example ranking: lower RMSE between expected and realized future path is better.
    if len(seq_df) == 0:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    minute_prices = pd.to_numeric(intraday_price_df[PRICE_COL], errors="coerce")

    for idx, row in seq_df.iterrows():
        if row["frequency"] == "intraday":
            horizon = EXAMPLE_INTRADAY_HORIZON
            end = int(row["sequence_start_idx"]) + horizon
            px = minute_prices.iloc[int(row["sequence_start_idx"]): end + 1]
            realized = _realized_cum_returns(px)
            expected = _expected_path(row["mu_hat"], len(px) - 1, 1.0)
        else:
            horizon = EXAMPLE_DAILY_HORIZON_DAYS
            end = int(row["sequence_start_idx"]) + 3 * horizon
            px = segmented_daily_df.iloc[int(row["sequence_start_idx"]): end + 1]["segment_vwap"]
            realized = _realized_cum_returns(px)
            expected = _expected_path(row["mu_hat"], len(px) - 1, 1.0 / 3.0)

        if len(realized) == 0 or len(realized) != len(expected):
            continue

        rmse = float(np.sqrt(np.mean((realized - expected) ** 2)))
        rows.append(
            {
                "seq_row_idx": idx,
                "frequency": row["frequency"],
                "timestamp": row["timestamp"],
                "drift_sign": row["drift_sign"],
                "rmse_fit": rmse,
                "example_horizon": horizon,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["rmse_fit", "frequency", "timestamp"]).reset_index(drop=True)


def choose_examples(
    seq_df: pd.DataFrame,
    intraday_price_df: pd.DataFrame,
    segmented_daily_df: pd.DataFrame,
    max_examples: int = MAX_EXAMPLES_PER_ASSET,
) -> pd.DataFrame:
    ranked = rank_example_candidates(seq_df, intraday_price_df, segmented_daily_df)
    if len(ranked) == 0:
        return pd.DataFrame()

    chosen_rows: list[int] = []
    seen_freq: set[str] = set()
    for _, row in ranked.iterrows():
        if len(chosen_rows) >= max_examples:
            break
        freq = row["frequency"]
        if freq not in seen_freq:
            chosen_rows.append(int(row["seq_row_idx"]))
            seen_freq.add(freq)
    for _, row in ranked.iterrows():
        if len(chosen_rows) >= max_examples:
            break
        seq_idx = int(row["seq_row_idx"])
        if seq_idx not in chosen_rows:
            chosen_rows.append(seq_idx)

    return seq_df.loc[chosen_rows].copy().reset_index(drop=True)


def _lookup_pairwise_pvalue(
    test_df: pd.DataFrame,
    frequency: str,
    drift_sign: str,
    grouping_scheme: str,
    metric: str,
    horizon: object,
    group_a: str,
    group_b: str,
) -> float:
    if len(test_df) == 0:
        return np.nan
    mask = (
        (test_df["frequency"] == frequency)
        & (test_df["drift_sign"] == drift_sign)
        & (test_df["grouping_scheme"] == grouping_scheme)
        & (test_df["metric"] == metric)
        & (test_df["horizon"] == horizon)
        & (test_df["test_type"] == "welch_ttest")
        & (
            ((test_df["group_a"] == group_a) & (test_df["group_b"] == group_b))
            | ((test_df["group_a"] == group_b) & (test_df["group_b"] == group_a))
        )
    )
    sub = test_df[mask]
    if len(sub) == 0:
        return np.nan
    return float(sub.iloc[0]["p_value"])


def _daily_annotation_lines(sub: pd.DataFrame, test_df: pd.DataFrame, drift_sign: str, scheme: str, terminal_horizon: int) -> list[str]:
    lines = [
        "Daily comparison summary",
        "Reference group = largest-n group at terminal horizon",
        "",
    ]
    terminal = sub[sub["horizon"] == terminal_horizon].sort_values("count", ascending=False)
    if len(terminal) == 0:
        return lines + ["No daily rows"]
    ref_group = str(terminal.iloc[0]["group_value"])
    lines.append(f"Reference: {_signal_group_label(drift_sign, scheme, ref_group)}")
    lines.append("")

    for _, row in terminal.iterrows():
        group_value = str(row["group_value"])
        p_mu = _lookup_pairwise_pvalue(test_df, "daily", drift_sign, scheme, "mu_hat", "current", group_value, ref_group)
        p_sig = _lookup_pairwise_pvalue(test_df, "daily", drift_sign, scheme, "sigma_hat", "current", group_value, ref_group)
        p_fut_sig = _lookup_pairwise_pvalue(test_df, "daily", drift_sign, scheme, f"future_sigma_hat_{terminal_horizon}d", terminal_horizon, group_value, ref_group)
        lines.extend(
            [
                _signal_group_label(drift_sign, scheme, group_value),
                f"n={int(row['count'])} | pre_mu={row['mean_current_mu_hat']:.4g} | pre_sigma={row['mean_current_sigma_hat']:.4g}",
                f"post_avg={row['mean_realized_cum_bps']:.2f}bps simple return | post_bandwidth={row['mean_future_sigma_hat']:.4g}",
                f"p(mu)={p_mu:.3g} | p(sigma)={p_sig:.3g} | p(future sigma {terminal_horizon}d)={p_fut_sig:.3g}",
                "",
            ]
        )
    return lines


def _seq_vol_pct_from_sigma(sigma_hat: float, horizon: float) -> float:
    if pd.isna(sigma_hat) or pd.isna(horizon) or horizon <= 0:
        return np.nan
    return float(sigma_hat * np.sqrt(horizon) * 100.0)


def _annotation_lines(
    sub: pd.DataFrame,
    test_df: pd.DataFrame,
    frequency: str,
    drift_sign: str,
    scheme: str,
    terminal_horizon: int,
) -> list[str]:
    lines = [
        f"{frequency.title()} comparison summary",
        "Reference group = largest-n group at terminal horizon",
        "",
    ]
    terminal = sub[sub["horizon"] == terminal_horizon].sort_values("count", ascending=False)
    if len(terminal) == 0:
        return lines + ["No rows"]
    ref_group = str(terminal.iloc[0]["group_value"])
    lines.append(f"Reference: {_signal_group_label(drift_sign, scheme, ref_group)}")
    lines.append("")
    suffix = "m" if frequency == "intraday" else "d"

    for _, row in terminal.iterrows():
        group_value = str(row["group_value"])
        p_mu = _lookup_pairwise_pvalue(test_df, frequency, drift_sign, scheme, "mu_hat", "current", group_value, ref_group)
        p_sig = _lookup_pairwise_pvalue(test_df, frequency, drift_sign, scheme, "sigma_hat", "current", group_value, ref_group)
        p_fut_sig = _lookup_pairwise_pvalue(test_df, frequency, drift_sign, scheme, f"future_sigma_hat_{terminal_horizon}{suffix}", terminal_horizon, group_value, ref_group)
        pre_vol_pct = _seq_vol_pct_from_sigma(row["mean_current_sigma_hat"], terminal_horizon)
        post_vol_pct = _seq_vol_pct_from_sigma(row["mean_future_sigma_hat"], terminal_horizon)
        lines.extend(
            [
                _signal_group_label(drift_sign, scheme, group_value),
                f"n={int(row['count'])} | pre_mu={row['mean_current_mu_hat']:.4g} | pre_sigma_seq={pre_vol_pct:.2f}%",
                f"future_mean_return={row['mean_realized_cum_bps'] / 100.0:.2f}% | future_sigma_seq={post_vol_pct:.2f}%",
                f"p(mu)={p_mu:.3g} | p(sigma)={p_sig:.3g} | p(future sigma)={p_fut_sig:.3g}",
                "",
            ]
        )
    return lines


def plot_grouped_future_paths_to_pdf(summary_long: pd.DataFrame, test_df: pd.DataFrame, pdf: PdfPages, ticker: str, image_dir: Path | None = None) -> None:
    if len(summary_long) == 0:
        return
    freq_specs = {
        "daily": {"horizons": DAILY_FUTURE_HORIZONS, "x_label": "Future horizon (days)"},
    }

    for freq, spec in freq_specs.items():
        for drift_sign in _ordered_drift_groups(summary_long):
            freq_df = summary_long[
                (summary_long["frequency"] == freq)
                & (summary_long["drift_sign"] == drift_sign)
                & (summary_long["grouping_scheme"] == "exclusive_location")
                & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
            ].copy()
            if len(freq_df) == 0:
                continue

            sub = freq_df.copy()
            sub["mean_future_sigma_seq_pct"] = sub.apply(lambda r: _seq_vol_pct_from_sigma(r["mean_future_sigma_hat"], r["horizon"]), axis=1)

            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            metric_defs = [
                ("mean_realized_cum_bps", "Return (%)", lambda s: s / 100.0),
                ("mean_realized_max_up_bps", "Max up (%)", lambda s: s / 100.0),
                ("mean_realized_max_down_bps", "Max down (%)", lambda s: s / 100.0),
                ("mean_future_sigma_seq_pct", "Vol (%)", lambda s: s),
            ]

            for group_value in LOCATION_COMPARE_GROUPS:
                grp = sub[sub["group_value"] == group_value].sort_values("horizon")
                if len(grp) == 0:
                    continue
                label = f"Realized average: {_pretty_group_value('exclusive_location', group_value)}"
                color = LOCATION_COMPARE_COLORS[group_value]
                for ax, (metric_col, title, transform) in zip(axes, metric_defs):
                    ax.plot(grp["horizon"], transform(grp[metric_col]), marker="o", linewidth=2.0, color=color, label=label)
                    ax.set_title(title)

            for ax in axes:
                ax.set_xlabel(spec["x_label"])
                ax.grid(alpha=0.25)
                ax.set_xticks(spec["horizons"])
                ax.set_ylabel("Percent" if ax is not axes[0] else "Percent")
            axes[0].legend(loc="best", fontsize=8)

            fig.suptitle(f"{ticker} | {_drift_title(drift_sign)} | returns and volatility")
            fig.tight_layout(rect=(0, 0, 1, 0.96))
            _save_report_figure(fig, image_dir, f"{ticker}_{freq}_{drift_sign}_future_return_volatility")
            pdf.savefig(fig)
            plt.close(fig)

            terminal_horizon = max(spec["horizons"])
            base._pdf_text_page(
                pdf,
                f"{ticker} {freq.title()} {_drift_title(drift_sign)} Notes",
                _annotation_lines(sub, test_df, freq, drift_sign, "exclusive_location", terminal_horizon),
                body_fontsize=10,
            )


def plot_location_parameter_comparison_to_pdf(
    summary_long: pd.DataFrame,
    test_df: pd.DataFrame,
    pdf: PdfPages,
    ticker: str,
    image_dir: Path | None = None,
) -> None:
    if len(summary_long) == 0:
        return

    for frequency, horizons, x_label in [
        ("daily", DAILY_FUTURE_HORIZONS, "Days after signal"),
    ]:
        for drift_sign in _ordered_drift_groups(summary_long):
            compare_labels = {
                "above_mid_only": "Above-mid abnormal volume",
                "below_mid_only": "Below-mid abnormal volume",
                "none": "No abnormal volume",
            }
            sub = summary_long[
                (summary_long["frequency"] == frequency)
                & (summary_long["drift_sign"] == drift_sign)
                & (summary_long["grouping_scheme"] == "exclusive_location")
                & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
            ].copy()
            if len(sub) == 0:
                continue

            sub["future_sigma_seq_pct"] = sub.apply(lambda r: _seq_vol_pct_from_sigma(r["mean_future_sigma_hat"], r["horizon"]), axis=1)
            sub["current_sigma_seq_pct"] = sub.apply(lambda r: _seq_vol_pct_from_sigma(r["mean_current_sigma_hat"], r["horizon"]), axis=1)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
            mu_ax, sigma_ax = axes

            for group_value in LOCATION_COMPARE_GROUPS:
                grp = sub[sub["group_value"] == group_value].sort_values("horizon")
                if len(grp) == 0:
                    continue
                label = f"Realized average: {compare_labels[group_value]}"
                color = LOCATION_COMPARE_COLORS[group_value]
                mu_ax.plot(grp["horizon"], grp["mean_future_mu_hat"], marker="o", linewidth=2.0, color=color, label=label)
                sigma_ax.plot(grp["horizon"], grp["future_sigma_seq_pct"], marker="o", linewidth=2.0, color=color, label=label)

                sigma_ax.plot(
                    grp["horizon"],
                    grp["current_sigma_seq_pct"],
                    linestyle="--",
                    linewidth=1.5,
                    color=color,
                    alpha=0.7,
                    label=f"Continuation: {_pretty_group_value('exclusive_location', group_value)}",
                )

            mu_ax.set_title("Mean log return")
            mu_ax.set_xlabel(x_label)
            mu_ax.set_ylabel("Future mean log return")
            mu_ax.set_xticks(horizons)
            mu_ax.set_yscale("symlog", linthresh=1e-5)
            mu_ax.grid(alpha=0.25)

            sigma_ax.set_title("Log-return volatility")
            sigma_ax.set_xlabel(x_label)
            sigma_ax.set_ylabel("Seq volatility (%)")
            sigma_ax.set_xticks(horizons)
            sigma_ax.set_yscale("log")
            sigma_ax.grid(alpha=0.25)
            sigma_ax.legend(loc="best", fontsize=8)
            fig.suptitle(f"{ticker} | {_drift_title(drift_sign)} | log-return parameter path")
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            _save_report_figure(fig, image_dir, f"{ticker}_{frequency}_{drift_sign}_parameter_comparison")
            pdf.savefig(fig)
            plt.close(fig)

            terminal_horizon = max(horizons)
            note_lines = [
                f"{_pretty_drift_label(drift_sign)} parameter comparison",
                "Groups: above, below, base",
                "",
            ]
            for group_value in LOCATION_COMPARE_GROUPS:
                grp_term = sub[(sub["group_value"] == group_value) & (sub["horizon"] == terminal_horizon)]
                if len(grp_term) == 0:
                    continue
                row = grp_term.iloc[0]
                p_mu = _lookup_pairwise_pvalue(
                    test_df,
                    frequency,
                    drift_sign,
                    "exclusive_location",
                    f"future_mu_hat_{terminal_horizon}{'m' if frequency == 'intraday' else 'd'}",
                    terminal_horizon,
                    group_value,
                    "none",
                )
                p_sig = _lookup_pairwise_pvalue(
                    test_df,
                    frequency,
                    drift_sign,
                    "exclusive_location",
                    f"future_sigma_hat_{terminal_horizon}{'m' if frequency == 'intraday' else 'd'}",
                    terminal_horizon,
                    group_value,
                    "none",
                )
                note_lines.extend(
                    [
                        compare_labels[group_value],
                        f"n={int(row['count'])} | pre_mu={row['mean_current_mu_hat']:.4g} | pre_sigma_seq={row['current_sigma_seq_pct']:.2f}%",
                        f"post_mu({terminal_horizon})={row['mean_future_mu_hat']:.4g} | post_sigma_seq({terminal_horizon})={row['future_sigma_seq_pct']:.2f}%",
                        f"p_mu_vs_baseline={p_mu:.3g} | p_sigma_vs_baseline={p_sig:.3g}",
                        "",
                    ]
                )
            base._pdf_text_page(pdf, f"{ticker} {frequency.title()} {_drift_title(drift_sign)} Parameter Notes", note_lines, body_fontsize=10)


def plot_example_paths_to_pdf(
    examples_df: pd.DataFrame,
    intraday_price_df: pd.DataFrame,
    segmented_daily_df: pd.DataFrame,
    pdf: PdfPages,
    ticker: str,
    image_dir: Path | None = None,
) -> None:
    if len(examples_df) == 0:
        return

    minute_prices = pd.to_numeric(intraday_price_df[PRICE_COL], errors="coerce")

    for _, row in examples_df.iterrows():
        if row["frequency"] == "intraday":
            horizon = EXAMPLE_INTRADAY_HORIZON
            suffix = f"{horizon}m"
            px = minute_prices.iloc[int(row["sequence_start_idx"]): int(row["sequence_start_idx"]) + horizon + 1]
            realized = _realized_cum_returns(px)
            horizon_steps = len(px) - 1
            bundle = _brownian_path_bundle(
                mu_hat=row["mu_hat"],
                sigma_hat=row["sigma_hat"],
                horizon_steps=horizon_steps,
                delta=1.0,
                seed=int(pd.Timestamp(row["timestamp"]).value % (2**32 - 1)),
            )
            x = np.arange(horizon_steps + 1)
            x_label = "Minutes after sequence start"
        else:
            horizon = EXAMPLE_DAILY_HORIZON_DAYS
            suffix = f"{horizon}d"
            px = segmented_daily_df.iloc[int(row["sequence_start_idx"]): int(row["sequence_start_idx"]) + 3 * horizon + 1]["segment_vwap"]
            realized = _realized_cum_returns(px)
            horizon_steps = len(px) - 1
            bundle = _brownian_path_bundle(
                mu_hat=row["mu_hat"],
                sigma_hat=row["sigma_hat"],
                horizon_steps=horizon_steps,
                delta=1.0 / 3.0,
                seed=int(pd.Timestamp(row["timestamp"]).value % (2**32 - 1)),
            )
            x = np.arange(horizon_steps + 1) / 3.0
            x_label = "Days after sequence start"

        if len(realized) == 0 or len(realized) != len(bundle["mean_path"]):
            continue

        fig, ax = plt.subplots(figsize=(11, 6))
        ax.fill_between(x, bundle["lower_95"], bundle["upper_95"], color="#9ecae1", alpha=0.35, label="Model 95% band")
        for i, sample_path in enumerate(bundle["sample_paths"][:10]):
            ax.plot(x, sample_path, linewidth=0.9, alpha=0.18, color="#3182bd", label="Sample Brownian paths" if i == 0 else None)
        ax.plot(x, bundle["mean_path"], linewidth=2.2, color="#08519c", label="Model mean path")
        ax.plot(x, realized, linewidth=2.0, color="#d62728", label="Realized cumulative return path")
        ax.axhline(0.0, color="black", alpha=0.25, linewidth=0.8)
        ax.grid(alpha=0.25)
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cumulative return")
        ax.legend(loc="best")

        annotation = textwrap.dedent(
            f"""
            signal_group = {_signal_group_label(row['drift_sign'], 'exclusive_location', row['exclusive_location'])}
            timestamp = {row['timestamp']}
            freq = {row['frequency']}
            mu_hat = {row['mu_hat']:.6g}
            sigma_hat = {row['sigma_hat']:.6g}
            expected_cum_bps = {row.get(f'expected_cum_bps_{suffix}', np.nan):.2f}
            realized_cum_bps = {row.get(f'realized_cum_bps_{suffix}', np.nan):.2f}
            realized_max_up_bps = {row.get(f'realized_max_up_bps_{suffix}', np.nan):.2f}
            realized_max_down_bps = {row.get(f'realized_max_down_bps_{suffix}', np.nan):.2f}
            realized_max_abs_bps = {row.get(f'realized_max_abs_bps_{suffix}', np.nan):.2f}
            drift_sign = {row['drift_sign']}
            volume_level = {row['volume_level']}
            imbalance_side = {row['imbalance_side']}
            exclusive_location = {row['exclusive_location']}
            both_above_below_high = {row['both_above_below_high']}
            """
        ).strip()
        ax.set_title(f"{ticker} | Past signal and future sequence path")
        fig.tight_layout()
        _save_report_figure(fig, image_dir, f"{ticker}_example_path_{row['frequency']}_{row['timestamp']}")
        pdf.savefig(fig)
        plt.close(fig)
        base._pdf_text_page(pdf, f"{ticker} Example Notes", annotation.splitlines(), body_fontsize=10)


def plot_short_strategy_equity_curve(ticker: str, equity_curve: pd.DataFrame, output_png: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(equity_curve) > 0:
        ax.plot(equity_curve["date"], equity_curve["short_overlay_net_equity"], linewidth=2.2, label="Net Strategy: Short Overlay to +/-200bps")
        ax.plot(equity_curve["date"], equity_curve["sell_wait_3d_net_equity"], linewidth=1.9, alpha=0.9, label="Net Strategy: Sell and Rebuy After 3D")
        ax.plot(equity_curve["date"], equity_curve["benchmark_equity"], linewidth=1.5, alpha=0.85, label="Benchmark: Buy and Hold")
    ax.set_title(f"{ticker} | Daily Below-Mid Abnormal Signal | Net Strategy Comparison | 5Y PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_png, dpi=200, bbox_inches="tight")
    plt.close(fig)


def append_short_strategy_plot_to_pdf(ticker: str, equity_curve: pd.DataFrame, pdf: PdfPages, summary_row: dict[str, object]) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    if len(equity_curve) > 0:
        ax.plot(equity_curve["date"], equity_curve["short_overlay_net_equity"], linewidth=2.2, label="Net Strategy: Short Overlay to +/-200bps")
        ax.plot(equity_curve["date"], equity_curve["sell_wait_3d_net_equity"], linewidth=1.9, alpha=0.9, label="Net Strategy: Sell and Rebuy After 3D")
        ax.plot(equity_curve["date"], equity_curve["benchmark_equity"], linewidth=1.5, alpha=0.85, label="Benchmark: Buy and Hold")
    ax.set_title(f"{ticker} | Daily Below-Mid Abnormal Signal | Net Strategy Comparison | 5Y PnL")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    note = textwrap.dedent(
        f"""
        short_overlay_net = {summary_row.get('short_overlay_cumulative_return_net', np.nan):.4f}
        sell_3d_net = {summary_row.get('sell_wait_3d_cumulative_return_net', np.nan):.4f}
        short_overlay_mdd = {summary_row.get('short_overlay_max_drawdown_net', np.nan):.4f}
        sell_3d_mdd = {summary_row.get('sell_wait_3d_max_drawdown_net', np.nan):.4f}
        short_overlay_trades = {summary_row.get('short_overlay_number_of_trades', np.nan)}
        sell_3d_trades = {summary_row.get('sell_wait_3d_number_of_trades', np.nan)}
        shares = {summary_row.get('position_size_shares', np.nan)}
        """
    ).strip()
    ax.text(0.02, 0.98, note, transform=ax.transAxes, va="top", ha="left", fontsize=9, family="monospace", bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def build_executive_summary_lines(
    seq_df: pd.DataFrame,
    summary_long: pd.DataFrame,
    test_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    daily_drift_df: pd.DataFrame | None = None,
) -> list[str]:
    lines: list[str] = []
    if len(seq_df) == 0:
        return ["No valid sequence-level estimates were produced for this asset."]

    lines.append("Daily drift state is binary: drifting vs non-drifting.")
    lines.append("Main groups: above, below, and base.")
    if daily_drift_df is not None and len(daily_drift_df) > 0:
        counts = daily_drift_df["drift_label_d"].value_counts()
        lines.append(
            "Daily drift-state counts: "
            + ", ".join(f"{idx}={int(val)}" for idx, val in counts.items())
        )
    if len(summary_long) > 0:
        term = summary_long[
            (summary_long["grouping_scheme"] == "exclusive_location")
            & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
        ].sort_values("horizon").groupby(["frequency", "group_value"], as_index=False).tail(1)
        if len(term) > 0:
            lines.append("Future cumulative return uses VWAP consistently, and the current output reports only daily horizons.")
            lines.append("Future paths use daily VWAP returns and daily horizon volatility.")
        daily_cases = summary_long[
            (summary_long["frequency"] == "daily")
            & (summary_long["grouping_scheme"] == "exclusive_location")
            & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
        ].sort_values("horizon").groupby(["drift_sign", "group_value"], as_index=False).tail(1)
        if len(daily_cases) > 0:
            lines.append("Daily case counts:")
            for _, row in daily_cases.sort_values(["drift_sign", "group_value"]).iterrows():
                lines.append(f"{_drift_title(row['drift_sign'])} | {_pretty_group_value('exclusive_location', row['group_value'])} | n={int(row['count'])}")
    if len(test_df) > 0:
        sig = test_df[
            (test_df["grouping_scheme"] == "exclusive_location")
            & (test_df["test_type"] == "welch_ttest")
            & (test_df["metric"].str.contains("future_", na=False))
            & (test_df["significant_5pct"])
        ]
        if len(sig) > 0:
            best = sig.sort_values("p_value").iloc[0]
            lines.append(f"Lowest future p-value: {best['p_value']:.3g} for {best['metric']}.")
        else:
            lines.append("Future differences are modest under the binary drift split.")

    if len(examples_df) > 0:
        lines.append(
            "One path example is selected by lowest out-of-sample RMSE."
        )
    else:
        lines.append("No representative examples met the minimum data requirements for the path-summary section.")

    lines.append("Local path summaries use simple log-return moments.")
    lines.append("All observations in this module are VWAP-based; daily analysis uses three fixed market segments per day.")
    return lines


def build_interpretation_lines(
    seq_df: pd.DataFrame,
    summary_long: pd.DataFrame,
    test_df: pd.DataFrame,
    examples_df: pd.DataFrame,
    daily_drift_df: pd.DataFrame | None = None,
) -> list[str]:
    lines: list[str] = []
    if len(seq_df) == 0:
        return ["Interpretation unavailable because sequence-level estimation returned no usable rows."]

    if len(examples_df) > 0:
        lines.append("Path-summary quality is evaluated using expected-path versus realized-path RMSE on future windows.")
    if len(summary_long) > 0:
        terminal = summary_long[
            (summary_long["grouping_scheme"] == "exclusive_location")
            & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
        ].sort_values("horizon").groupby(["frequency", "group_value"], as_index=False).tail(1)
        drift_disp = terminal.groupby("frequency")["mean_realized_cum_bps"].agg(lambda s: s.max() - s.min() if len(s.dropna()) else np.nan)
        vol_disp = terminal.groupby("frequency")["mean_future_sigma_hat"].agg(lambda s: s.max() - s.min() if len(s.dropna()) else np.nan)
        if drift_disp.notna().any():
            lines.append(f"Largest cross-group realized-return separation appears in: {drift_disp.sort_values(ascending=False).index[0]}.")
        if vol_disp.notna().any():
            lines.append(f"Largest cross-group volatility separation appears in: {vol_disp.sort_values(ascending=False).index[0]}.")
    if len(test_df) > 0:
        sig_rate = test_df["significant_5pct"].mean()
        lines.append(f"Across reported group-difference tests, the 5% significance rate is {sig_rate:.1%}.")
    if daily_drift_df is not None and len(daily_drift_df) > 0:
        trigger_rate = (daily_drift_df["abs_ST_d"] > DAILY_ST_THRESHOLD).mean()
        lines.append(f"The daily drifting trigger rate under the RSD screen is {trigger_rate:.1%}.")
    lines.append("Economically noticeable effects should be judged by the scale of future return and volatility spreads, not only by group labels.")
    lines.append("Large realized deviations from the summary paths can indicate jump-like behavior or fast regime change that simple return moments do not capture.")
    return lines


def build_daily_drift_screen_lines(daily_drift_df: pd.DataFrame) -> list[str]:
    if len(daily_drift_df) == 0:
        return ["No daily drift-screen rows were produced."]
    counts = daily_drift_df["drift_label_d"].value_counts()
    invalid = daily_drift_df.loc[~daily_drift_df["valid_day_flag"], "invalid_reason"].value_counts().head(10)
    lines = [
        "Daily drift-strength screen summary",
        f"Total completed trading days classified: {len(daily_drift_df)}",
        f"Drifting trigger threshold: abs(ST_d(k)) > {DAILY_ST_THRESHOLD:.2f}",
        "",
        "Drift-label counts:",
    ]
    lines.extend(f"{idx}: {int(val)}" for idx, val in counts.items())
    lines.append("")
    lines.append("Most common invalid reasons:")
    if len(invalid) > 0:
        lines.extend(f"{idx}: {int(val)}" for idx, val in invalid.items())
    else:
        lines.append("None")
    lines.append("")
    trig = daily_drift_df[daily_drift_df["abs_ST_d"] > DAILY_ST_THRESHOLD]
    lines.append(f"Triggered drifting days: {len(trig)}")
    return lines


def _cleanup_existing_report_csvs(output_pdf: Path) -> None:
    base_name = output_pdf.with_suffix("")
    pattern = f"{base_name.name}_*.csv"
    for path in base_name.parent.glob(pattern):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _cleanup_existing_report_images(image_dir: Path) -> None:
    if not image_dir.exists():
        return
    for path in image_dir.glob("*.png"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _cleanup_existing_image_export_csvs(image_dir: Path, ticker: str) -> None:
    if not image_dir.exists():
        return
    safe_ticker = str(ticker).upper().strip()
    for path in image_dir.glob(f"{safe_ticker}_*.csv"):
        try:
            path.unlink()
        except FileNotFoundError:
            pass


class _NullPdf:
    def savefig(self, *args, **kwargs) -> None:
        return None


def _format_table_number(value: object) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        abs_value = abs(float(value))
        if abs_value >= 1000:
            return f"{float(value):,.2f}"
        if abs_value >= 1:
            return f"{float(value):.4f}"
        return f"{float(value):.6f}"
    return str(value)


def _render_journal_table_figure(
    title: str,
    df: pd.DataFrame,
    note: str | None = None,
    figsize: tuple[float, float] = (11.0, 8.5),
    body_fontsize: float = 8.8,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.text(0.05, 0.96, title, fontsize=14, fontweight="bold", va="top")
    if note:
        fig.text(0.05, 0.925, note, fontsize=9.5, va="top")

    display = df.copy()
    if "variable" in display.columns:
        display["variable"] = display["variable"].map(_display_label)
    if "screen_result" in display.columns:
        display["screen_result"] = display["screen_result"].map(_display_label)
    cell_text = [[_format_table_number(value) for value in row] for row in display.to_numpy()]
    n_cols = max(len(display.columns), 1)
    table = ax.table(
        cellText=cell_text,
        colLabels=[_display_label(c) for c in display.columns],
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.05, 0.08, 0.90, 0.78],
        colWidths=[0.90 / n_cols] * n_cols,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)
    table.scale(1.0, 1.35)
    cells = table.get_celld()
    max_row = max((row for row, _ in cells.keys()), default=0)
    for (row, _col), cell in cells.items():
        cell.set_facecolor("white")
        cell.set_edgecolor("black")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(0.9)
        elif row == max_row:
            cell.visible_edges = "B"
            cell.set_linewidth(0.9)
    return fig


def add_journal_table_to_pdf(
    df: pd.DataFrame,
    pdf: PdfPages,
    ticker: str,
    section_title: str,
    image_dir: Path | None = None,
    note: str | None = None,
    max_rows_per_page: int = 18,
) -> None:
    if df is None or len(df) == 0:
        base._pdf_text_page(pdf, f"{ticker} {section_title}", ["No rows"], body_fontsize=10)
        return
    for page_idx, start in enumerate(range(0, len(df), max_rows_per_page), start=1):
        sub = df.iloc[start: start + max_rows_per_page].reset_index(drop=True)
        page_title = f"{ticker} | {section_title}"
        page_note = note
        if len(df) > max_rows_per_page:
            page_note = (note + " | " if note else "") + f"page {page_idx}"
        fig = _render_journal_table_figure(page_title, sub, note=page_note)
        _save_report_figure(fig, image_dir, f"{ticker}_{section_title}_table_page_{page_idx}")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)


def build_daily_one_day_summary_stats_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    daily = build_daily_group_feature_frame(raw_df)
    daily_vwap = build_daily_vwap_observations(raw_df)
    if len(daily) == 0:
        return pd.DataFrame()
    work = daily.copy()
    daily_vwap_small = daily_vwap[["date", "daily_vwap"]].copy()
    daily_vwap_small["date"] = pd.to_datetime(daily_vwap_small["date"])
    daily_vwap_small["price_change_1d_pct"] = pd.to_numeric(daily_vwap_small["daily_vwap"], errors="coerce").pct_change() * 100.0
    price_change = pd.to_numeric(daily_vwap_small["price_change_1d_pct"], errors="coerce")
    daily_vwap_small["price_change_outlier_flag"] = price_change.abs() > 50.0
    daily_vwap_small.loc[daily_vwap_small["price_change_outlier_flag"], "price_change_1d_pct"] = np.nan
    work["date"] = pd.to_datetime(work["date"])
    work = work.merge(daily_vwap_small[["date", "price_change_1d_pct", "price_change_outlier_flag"]], on="date", how="left")
    metric_specs = [
        ("Above-mid volume", "trade_volume_above_mid"),
        ("Below-mid volume", "trade_volume_below_mid"),
        ("At-mid volume", "trade_volume_at_mid"),
        ("Daily VWAP price change (%)", "price_change_1d_pct"),
        ("Total volume", "total_volume"),
    ]
    rows: list[dict[str, object]] = []
    for label, col in metric_specs:
        if col not in work.columns:
            continue
        x = pd.to_numeric(work[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        rows.append(
            {
                "variable": label,
                "n": int(len(x)),
                "mean": float(x.mean()) if len(x) else np.nan,
                "sd": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                "min": float(x.min()) if len(x) else np.nan,
                "max": float(x.max()) if len(x) else np.nan,
                "outliers_removed": int(work["price_change_outlier_flag"].fillna(False).sum()) if col == "price_change_1d_pct" else 0,
            }
        )
    return pd.DataFrame(rows)


def build_abnormal_screen_frequency_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    daily = build_daily_group_feature_frame(raw_df)
    if len(daily) == 0:
        return pd.DataFrame()
    labels = daily.apply(_daily_group_labels, axis=1, result_type="expand")
    abnormal_mask = labels["exclusive_location"].astype(str) != "none"
    total = int(len(labels))
    normal_count = int((~abnormal_mask).sum())
    abnormal_count = int(abnormal_mask.sum())
    rows = [
        {"screen_result": "Normal volume days", "count": normal_count, "percent": (normal_count / total * 100.0) if total else np.nan},
        {"screen_result": f"Abnormal volume days (q{_active_abnormal_q_label()})", "count": abnormal_count, "percent": (abnormal_count / total * 100.0) if total else np.nan},
    ]
    return pd.DataFrame(rows)


def plot_abnormal_screen_frequency_to_pdf(
    frequency_df: pd.DataFrame,
    pdf: PdfPages,
    ticker: str,
    image_dir: Path | None = None,
) -> None:
    if frequency_df is None or len(frequency_df) == 0:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    colors = ["#4d4d4d", "#c23b22"]
    screen_labels = frequency_df["screen_result"].map(_display_label)
    bars = ax.bar(screen_labels, frequency_df["percent"], color=colors[:len(frequency_df)], width=0.55)
    for bar, (_, row) in zip(bars, frequency_df.iterrows()):
        height = float(row["percent"])
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(1.0, float(frequency_df["percent"].max()) * 0.025),
            f"{height:.1f}%\nn={int(row['count'])}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_title(f"{ticker} | daily q{_active_abnormal_q_label()} abnormal-volume screen frequency")
    ax.set_ylabel("Share of trading days (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, min(100.0, float(frequency_df["percent"].max()) * 1.22 + 4.0))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    _save_report_figure(fig, image_dir, f"{ticker}_q{_active_abnormal_q_label()}_abnormal_screen_frequency")
    pdf.savefig(fig)
    plt.close(fig)


def plot_diffusion_baseline_average_paths_to_pdf(
    summary_long: pd.DataFrame,
    pdf: PdfPages,
    ticker: str,
    image_dir: Path | None = None,
) -> None:
    if len(summary_long) == 0:
        return
    for drift_sign in _ordered_drift_groups(summary_long):
        sub = summary_long[
            (summary_long["frequency"] == "daily")
            & (summary_long["drift_sign"] == drift_sign)
            & (summary_long["grouping_scheme"] == "exclusive_location")
            & (summary_long["group_value"].isin(LOCATION_COMPARE_GROUPS))
        ].copy()
        if len(sub) == 0:
            continue
        abn_model_source = sub[sub["group_value"].isin(("above_mid_only", "below_mid_only"))].copy()
        no_abn_model_source = sub[sub["group_value"] == "none"].copy()
        if len(abn_model_source) == 0:
            continue
        first_horizon = float(min(DAILY_FUTURE_HORIZONS))
        abn_model_source = abn_model_source[abn_model_source["horizon"].astype(float) == first_horizon]
        no_abn_model_source = no_abn_model_source[no_abn_model_source["horizon"].astype(float) == first_horizon]
        if len(abn_model_source) == 0:
            continue
        abn_weights = abn_model_source["count"].astype(float).clip(lower=1.0).to_numpy()
        abn_mu_hat = float(np.average(abn_model_source["mean_current_mu_hat"].astype(float), weights=abn_weights))
        abn_sigma_hat = float(np.average(abn_model_source["mean_current_sigma_hat"].astype(float), weights=abn_weights))
        horizons = np.array([0, *DAILY_FUTURE_HORIZONS], dtype="float64")
        abn_mean_log = abn_mu_hat * horizons
        abn_std_log = abn_sigma_hat * np.sqrt(horizons)
        abn_model_mean = np.expm1(abn_mean_log) * 100.0
        abn_model_lower = np.expm1(abn_mean_log - 1.96 * abn_std_log) * 100.0
        abn_model_upper = np.expm1(abn_mean_log + 1.96 * abn_std_log) * 100.0

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
        mean_ax, vol_ax = axes
        mean_ax.fill_between(horizons, abn_model_lower, abn_model_upper, color="#bdbdbd", alpha=0.30, label="Abnormal volume continuation 95%")

        no_abn_mean = None
        no_abn_sigma_hat = np.nan
        if len(no_abn_model_source) > 0:
            no_abn_row = no_abn_model_source.iloc[0]
            no_abn_mu_hat = float(no_abn_row["mean_current_mu_hat"])
            no_abn_sigma_hat = float(no_abn_row["mean_current_sigma_hat"])
            no_abn_mean = np.expm1(no_abn_mu_hat * horizons) * 100.0

        for group_value in LOCATION_COMPARE_GROUPS:
            grp = sub[sub["group_value"] == group_value].sort_values("horizon")
            if len(grp) == 0:
                continue
            color = LOCATION_COMPARE_COLORS[group_value]
            label = f"Realized average: {_future_path_group_label(group_value)}"
            x = grp["horizon"].astype(float).to_numpy()
            y = (grp["mean_realized_cum_bps"].astype(float) / 100.0).to_numpy()
            mean_ax.plot(np.r_[0.0, x], np.r_[0.0, y], marker="o", linewidth=2.0, color=color, label=label)
            vol_ax.plot(
                x,
                grp.apply(lambda r: _seq_vol_pct_from_sigma(r["mean_future_sigma_hat"], r["horizon"]), axis=1),
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )

        mean_ax.plot(
            horizons,
            abn_model_mean,
            color="#111111",
            linewidth=3.0,
            linestyle=":",
            zorder=8,
            label="Abnormal volume continuation",
        )
        if no_abn_mean is not None:
            mean_ax.plot(
                horizons,
                no_abn_mean,
                color="#7b3294",
                linewidth=3.0,
                linestyle=":",
                zorder=8,
                label="No abnormal volume continuation",
            )

        abn_model_vol = [0.0] + [_seq_vol_pct_from_sigma(abn_sigma_hat, h) for h in DAILY_FUTURE_HORIZONS]
        vol_ax.plot(horizons, abn_model_vol, color="#111111", linewidth=3.0, linestyle=":", zorder=8, label="Abnormal volume continuation")
        if pd.notna(no_abn_sigma_hat):
            no_abn_model_vol = [0.0] + [_seq_vol_pct_from_sigma(no_abn_sigma_hat, h) for h in DAILY_FUTURE_HORIZONS]
            vol_ax.plot(
                horizons,
                no_abn_model_vol,
                color="#7b3294",
                linewidth=3.0,
                linestyle=":",
                zorder=8,
                label="No abnormal volume continuation",
            )

        mean_ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.25)
        mean_ax.set_title("Mean path")
        mean_ax.set_xlabel("Days after signal")
        mean_ax.set_ylabel("Cumulative return (%)")
        mean_ax.set_xticks(DAILY_FUTURE_HORIZONS)
        mean_ax.grid(alpha=0.25)
        mean_ax.legend(loc="best", fontsize=8)

        vol_ax.set_title("Volatility path")
        vol_ax.set_xlabel("Days after signal")
        vol_ax.set_ylabel("Sequence volatility (%)")
        vol_ax.set_xticks(DAILY_FUTURE_HORIZONS)
        vol_ax.grid(alpha=0.25)
        vol_ax.legend(loc="best", fontsize=8)

        fig.suptitle(f"{ticker} | {_future_path_state_title(drift_sign)} | future sequence: continuation vs realized average")
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        _save_report_figure(fig, image_dir, f"{ticker}_diffusion_baseline_average_paths_{drift_sign}")
        pdf.savefig(fig)
        plt.close(fig)


def _metadata_lines(
    ticker: str,
    raw_df: pd.DataFrame,
    count_base_dir: str,
    main_base_dir: str,
) -> list[str]:
    min_dt = pd.to_datetime(raw_df["minute_dt"]).min()
    max_dt = pd.to_datetime(raw_df["minute_dt"]).max()
    return [
        f"Ticker: {ticker}",
        f"Date range: {min_dt} to {max_dt}",
        "Frequencies analyzed in the current output: daily only",
        "Reported split uses two states: drifting and non-drifting.",
        f"Daily local estimation window: past {DAILY_ESTIMATION_WINDOW_RETURNS} segment log-VWAP increments (1/3-day observations)",
        f"Daily future horizons: {list(DAILY_FUTURE_HORIZONS)} days",
        f"Daily quantile lookback: {DAILY_QUANTILE_LOOKBACK_DAYS} trading days",
        f"Abnormal-volume threshold: rolling {_active_abnormal_q_label()}th percentile throughout the report.",
        "Location groups used in the figures: only above-mid abnormal, only below-mid abnormal, and a general baseline with no abnormal volume.",
        "At-mid-only cases are excluded from the main comparison figures.",
        "Future-sequence return definition: simple cumulative return from signal-time VWAP to horizon-end VWAP, i.e. Pt/P0 - 1.",
        "Cumulative return at time t means total return from signal time up to t, not the one-step return for only that interval.",
        "This report does not run intraday minute-level post-signal analysis.",
        "Daily future returns use aggregated daily VWAP built from minute VWAP.",
        "Daily segmentation: 09:30-11:30, 11:30-13:30, 13:30-16:00 ET",
        "Daily drift-strength test uses 1-minute price-difference returns r_{d,i} = p_{d,i} - p_{d,i-1}.",
        "RSD_d(k) = (1 / Delta) * sum r_{d,i} r_{d,i-k}; RQ_d = (1 / (3 * Delta)) * sum r_{d,i}^4.",
        "VarHat_d = RQ_d / Delta and ST_d(k) = RSD_d(k) / sqrt(RQ_d / Delta).",
        f"Daily drifting screen: abs(ST_d(k)) > {DAILY_ST_THRESHOLD:.2f}, with lag k = {DAILY_RSD_LAG_K}.",
        "No directional conditioning is used.",
        "Current and future mu/sigma fields in downstream tables are simple log-return summaries kept for compatibility, not fitted likelihood outputs.",
        "Observation note: all prices are VWAP-based; segment VWAP uses volume-weighted minute VWAP when volume exists, else simple average of minute VWAPs.",
        "Color map used throughout: red = only above-mid abnormal, blue = only below-mid abnormal, black = general baseline with no abnormal volume.",
        f"Data directories: count={count_base_dir}, main={main_base_dir}",
    ]


def run_local_nojump_mle_for_ticker(
    ticker: str,
    output_pdf: str | Path,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    raw_df = base.load_count_aligned_dataset(
        ticker=ticker,
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
    )
    raw_df = _clean_loaded_dataset(raw_df, ticker)

    daily_drift_df = compute_daily_drift_screen(raw_df)

    daily_seq = build_daily_sequence_level_df(raw_df, ticker=ticker, daily_drift_df=daily_drift_df)
    combined_seq = daily_seq.copy()
    if len(combined_seq) > 0:
        combined_seq = combined_seq[combined_seq["exclusive_location"].isin(LOCATION_COMPARE_GROUPS)].copy()
    combined_grouped = _attach_grouping_views(combined_seq)
    summary_long = build_group_summary_long(combined_grouped)
    test_df = build_group_difference_tests(combined_grouped)
    daily_summary_stats_df = build_daily_one_day_summary_stats_table(raw_df)
    abnormal_frequency_df = build_abnormal_screen_frequency_table(raw_df)
    segmented_daily = build_segmented_daily_observations(raw_df)
    prepared_raw = _copy_prepare(raw_df)
    examples_df = choose_examples(combined_seq, prepared_raw, segmented_daily)

    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    image_dir = _report_image_dir(output_pdf)
    image_dir.mkdir(parents=True, exist_ok=True)

    if save_csv:
        _cleanup_existing_report_csvs(output_pdf)
        _cleanup_existing_report_images(image_dir)
        base_name = output_pdf.with_suffix("")
        if len(combined_seq) > 0:
            combined_seq.to_csv(base_name.with_name(base_name.name + "_sequence_level.csv"), index=False)
        if len(daily_drift_df) > 0:
            daily_drift_df.to_csv(base_name.with_name(base_name.name + "_daily_drift_screen.csv"), index=False)
        detected_events_path = base_name.with_name(base_name.name + f"_{DETECTED_EVENTS_CSV_NAME}")
        daily_drift_df[daily_drift_df["abs_ST_d"] > DAILY_ST_THRESHOLD].to_csv(detected_events_path, index=False)
        if len(summary_long) > 0:
            summary_long.to_csv(base_name.with_name(base_name.name + "_group_summary_long.csv"), index=False)
        if len(test_df) > 0:
            test_df.to_csv(base_name.with_name(base_name.name + "_group_difference_tests.csv"), index=False)
        if len(daily_summary_stats_df) > 0:
            daily_summary_stats_df.to_csv(base_name.with_name(base_name.name + "_daily_one_day_summary_stats.csv"), index=False)
        if len(abnormal_frequency_df) > 0:
            abnormal_frequency_df.to_csv(base_name.with_name(base_name.name + "_abnormal_screen_frequency.csv"), index=False)

    with PdfPages(output_pdf) as pdf:
        base._pdf_text_page(pdf, f"{ticker} Daily Drift Screen Analysis", _metadata_lines(ticker, raw_df, count_base_dir, main_base_dir))
        base._pdf_text_page(pdf, f"{ticker} Executive Summary", build_executive_summary_lines(combined_seq, summary_long, test_df, examples_df, daily_drift_df))
        base._pdf_text_page(pdf, f"{ticker} Daily Drift Screen", build_daily_drift_screen_lines(daily_drift_df))
        add_journal_table_to_pdf(
            daily_summary_stats_df,
            pdf,
            ticker,
            "Daily Summary Stats",
            image_dir=image_dir,
            note="Volume fields are one trading day totals; price change is daily VWAP percent change.",
        )
        plot_abnormal_screen_frequency_to_pdf(abnormal_frequency_df, pdf, ticker, image_dir=image_dir)
        plot_diffusion_baseline_average_paths_to_pdf(summary_long, pdf, ticker, image_dir=image_dir)
        plot_grouped_future_paths_to_pdf(summary_long, test_df, pdf, ticker, image_dir=image_dir)
        plot_location_parameter_comparison_to_pdf(summary_long, test_df, pdf, ticker, image_dir=image_dir)

        if len(examples_df) > 0:
            base._pdf_text_page(
                pdf,
                f"{ticker} Example Selection Rule",
                [
                    "Examples are ranked by out-of-sample RMSE between the summary-path cumulative return and the realized cumulative VWAP return path.",
                    "Selection is automatic and transparent: choose the lowest-RMSE valid daily examples under the current daily-only reporting output.",
                    "Example plots show the summary-path cumulative return path, a 95% band, simulated sample paths, and the realized cumulative return path on the same axis.",
                    "Example annotations also report expected cumulative return and realized excursion metrics in basis points over the plotted horizon.",
                    "Path summaries are based on simple log-return mean and volatility estimates over the corresponding historical window.",
                ],
            )
            plot_example_paths_to_pdf(examples_df, prepared_raw, segmented_daily, pdf, ticker, image_dir=image_dir)

        base._pdf_text_page(pdf, f"{ticker} Interpretation", build_interpretation_lines(combined_seq, summary_long, test_df, examples_df, daily_drift_df))

    _write_report_marker(output_pdf, ticker)

    return {
        "sequence_level_df": combined_seq,
        "daily_drift_screen_df": daily_drift_df,
        "detected_drift_events_df": daily_drift_df[daily_drift_df["abs_ST_d"] > DAILY_ST_THRESHOLD].copy(),
        "summary_long_df": summary_long,
        "daily_one_day_summary_stats_df": daily_summary_stats_df,
        "abnormal_screen_frequency_df": abnormal_frequency_df,
        "compact_summary_df": pd.DataFrame(),
        "group_difference_tests_df": test_df,
        "examples_df": examples_df,
    }


def export_analysis2_images_for_ticker(
    ticker: str,
    image_dir: str | Path,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    save_csv: bool = True,
) -> dict[str, pd.DataFrame]:
    ticker = str(ticker).upper().strip()
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    _cleanup_existing_report_images(image_dir)
    _cleanup_existing_image_export_csvs(image_dir, ticker)

    raw_df = base.load_count_aligned_dataset(
        ticker=ticker,
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
    )
    raw_df = _clean_loaded_dataset(raw_df, ticker)

    daily_drift_df = compute_daily_drift_screen(raw_df)
    daily_seq = build_daily_sequence_level_df(raw_df, ticker=ticker, daily_drift_df=daily_drift_df)
    combined_seq = daily_seq.copy()
    if len(combined_seq) > 0:
        combined_seq = combined_seq[combined_seq["exclusive_location"].isin(LOCATION_COMPARE_GROUPS)].copy()
    combined_grouped = _attach_grouping_views(combined_seq)
    summary_long = build_group_summary_long(combined_grouped)
    test_df = build_group_difference_tests(combined_grouped)
    daily_summary_stats_df = build_daily_one_day_summary_stats_table(raw_df)
    abnormal_frequency_df = build_abnormal_screen_frequency_table(raw_df)
    segmented_daily = build_segmented_daily_observations(raw_df)
    prepared_raw = _copy_prepare(raw_df)
    examples_df = choose_examples(combined_seq, prepared_raw, segmented_daily)

    if save_csv:
        if len(combined_seq) > 0:
            combined_seq.to_csv(image_dir / f"{ticker}_sequence_level.csv", index=False)
        if len(summary_long) > 0:
            summary_long.to_csv(image_dir / f"{ticker}_group_summary_long.csv", index=False)
        if len(test_df) > 0:
            test_df.to_csv(image_dir / f"{ticker}_group_difference_tests.csv", index=False)
        if len(daily_summary_stats_df) > 0:
            daily_summary_stats_df.to_csv(image_dir / f"{ticker}_daily_one_day_summary_stats.csv", index=False)
        if len(abnormal_frequency_df) > 0:
            abnormal_frequency_df.to_csv(image_dir / f"{ticker}_abnormal_screen_frequency.csv", index=False)

    null_pdf = _NullPdf()
    add_journal_table_to_pdf(
        daily_summary_stats_df,
        null_pdf,
            ticker,
            "Daily Summary Stats",
        image_dir=image_dir,
        note="Volume fields are one trading day totals; price change is daily VWAP percent change.",
    )
    plot_abnormal_screen_frequency_to_pdf(abnormal_frequency_df, null_pdf, ticker, image_dir=image_dir)
    plot_diffusion_baseline_average_paths_to_pdf(summary_long, null_pdf, ticker, image_dir=image_dir)
    plot_grouped_future_paths_to_pdf(summary_long, test_df, null_pdf, ticker, image_dir=image_dir)
    plot_location_parameter_comparison_to_pdf(summary_long, test_df, null_pdf, ticker, image_dir=image_dir)
    if len(examples_df) > 0:
        plot_example_paths_to_pdf(examples_df, prepared_raw, segmented_daily, null_pdf, ticker, image_dir=image_dir)

    return {
        "sequence_level_df": combined_seq,
        "daily_drift_screen_df": daily_drift_df,
        "summary_long_df": summary_long,
        "daily_one_day_summary_stats_df": daily_summary_stats_df,
        "abnormal_screen_frequency_df": abnormal_frequency_df,
        "group_difference_tests_df": test_df,
        "examples_df": examples_df,
    }


def _aggregate_cross_asset_average(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()

    work = df.copy()
    if "asset" not in work.columns:
        raise ValueError("cross-asset aggregation requires an 'asset' column")

    numeric_cols = [
        c for c in work.columns
        if c not in set(group_cols + ["asset"]) and pd.api.types.is_numeric_dtype(work[c])
    ]
    agg_map = {c: "mean" for c in numeric_cols}
    out = (
        work.groupby(group_cols, dropna=False, as_index=False)
        .agg(agg_map)
        .reset_index(drop=True)
    )
    asset_counts = (
        work.groupby(group_cols, dropna=False)["asset"]
        .nunique()
        .rename("asset_count")
        .reset_index()
    )
    out = out.merge(asset_counts, on=group_cols, how="left")
    return out


def build_cross_asset_average_report(
    results: dict[str, dict[str, pd.DataFrame]],
) -> dict[str, pd.DataFrame]:
    summary_frames: list[pd.DataFrame] = []
    drift_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for ticker, result in results.items():
        summary_long = result.get("summary_long_df", pd.DataFrame())
        if len(summary_long) > 0:
            tmp = summary_long.copy()
            tmp["asset"] = str(ticker).upper()
            summary_frames.append(tmp)

        daily_drift = result.get("daily_drift_screen_df", pd.DataFrame())
        if len(daily_drift) > 0:
            tmp = daily_drift.copy()
            tmp["asset"] = str(ticker).upper()
            drift_frames.append(tmp)

        test_df = result.get("group_difference_tests_df", pd.DataFrame())
        if len(test_df) > 0:
            tmp = test_df.copy()
            tmp["asset"] = str(ticker).upper()
            test_frames.append(tmp)

    summary_all = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    drift_all = pd.concat(drift_frames, ignore_index=True) if drift_frames else pd.DataFrame()
    test_all = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    summary_avg = _aggregate_cross_asset_average(
        summary_all,
        ["frequency", "drift_sign", "grouping_scheme", "group_value", "horizon"],
    ) if len(summary_all) > 0 else pd.DataFrame()
    test_avg = _aggregate_cross_asset_average(
        test_all,
        ["frequency", "drift_sign", "grouping_scheme", "group_value", "metric", "test_type", "group_a", "group_b"],
    ) if len(test_all) > 0 else pd.DataFrame()

    if len(drift_all) > 0:
        drift_summary = (
            drift_all.groupby("asset", dropna=False)
            .agg(
                trading_days=("date", "count"),
                triggered_days=("abs_ST_d", lambda s: int((pd.to_numeric(s, errors="coerce") > DAILY_ST_THRESHOLD).sum())),
                avg_abs_st=("abs_ST_d", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            )
            .reset_index(drop=False)
        )
    else:
        drift_summary = pd.DataFrame()

    return {
        "summary_long_avg_df": summary_avg,
        "group_difference_tests_avg_df": test_avg,
        "daily_drift_asset_summary_df": drift_summary,
        "summary_long_all_df": summary_all,
    }


def write_cross_asset_average_pdf(
    output_pdf: str | Path,
    aggregated: dict[str, pd.DataFrame],
    completed_tickers: list[str],
) -> Path:
    output_pdf = Path(output_pdf)
    output_pdf.parent.mkdir(parents=True, exist_ok=True)

    summary_avg = aggregated.get("summary_long_avg_df", pd.DataFrame())
    test_avg = aggregated.get("group_difference_tests_avg_df", pd.DataFrame())
    drift_summary = aggregated.get("daily_drift_asset_summary_df", pd.DataFrame())

    with PdfPages(output_pdf) as pdf:
        base._pdf_text_page(
            pdf,
            "All-Asset Average Summary",
            [
                f"Mode: {REPORT_MODE_TAG}",
                f"Completed assets: {len(completed_tickers)}",
                f"Assets included: {', '.join(completed_tickers)}" if completed_tickers else "Assets included: none",
                "This summary averages per-asset daily-only analysis outputs.",
                f"Abnormal-volume threshold: {_active_abnormal_q_label()}th percentile.",
                "All plots use daily horizons; subdaily post-signal analysis is excluded from execution.",
            ],
        )

        if len(drift_summary) > 0:
            base._pdf_dataframe_table_pages(
                pdf,
                base_title="All-Asset Average Summary",
                section_title="Per-Asset Drift Summary",
                df=drift_summary.round(6),
                max_rows_per_page=20,
                max_cols_per_page=6,
            )

        if len(summary_avg) > 0:
            plot_grouped_future_paths_to_pdf(summary_avg, test_avg, pdf, "ALL_ASSETS_AVG")
            plot_location_parameter_comparison_to_pdf(summary_avg, test_avg, pdf, "ALL_ASSETS_AVG")
            compact = build_compact_summary_table(summary_avg)
            if len(compact) > 0:
                base._pdf_dataframe_table_pages(
                    pdf,
                    base_title="All-Asset Average Summary",
                    section_title="Compact Group Summary",
                    df=compact.round(6),
                    max_rows_per_page=20,
                    max_cols_per_page=8,
                )

    return output_pdf


def load_saved_ticker_report_frames(output_pdf: str | Path, ticker: str) -> dict[str, pd.DataFrame]:
    out_path = _derive_ticker_path(output_pdf, ticker)
    base_name = Path(out_path).with_suffix("")
    file_map = {
        "summary_long_df": base_name.with_name(base_name.name + "_group_summary_long.csv"),
        "group_difference_tests_df": base_name.with_name(base_name.name + "_group_difference_tests.csv"),
        "daily_drift_screen_df": base_name.with_name(base_name.name + "_daily_drift_screen.csv"),
    }
    out: dict[str, pd.DataFrame] = {}
    for key, path in file_map.items():
        if path.exists():
            out[key] = pd.read_csv(path)
        else:
            out[key] = pd.DataFrame()
    return out


def run_local_nojump_mle_batch(
    output_pdf: str | Path = "daily_drift_screen_report.pdf",
    tickers: list[str] | None = None,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    save_csv: bool = True,
    resume: bool = True,
) -> dict[str, dict[str, pd.DataFrame]]:
    global INTRADAY_ABNORMAL_QS

    if tickers is None:
        tickers = base._get_common_tickers(
            count_base_dir=count_base_dir,
            main_base_dir=main_base_dir,
            main_session_type=main_session_type,
        )

    results: dict[str, dict[str, pd.DataFrame]] = {}
    completed_tickers: list[str] = []

    for ticker in tickers:
        out_path = _derive_ticker_path(output_pdf, ticker)
        if resume and _is_report_complete(out_path):
            print(f"[analysis2] Skipping completed ticker {ticker} at {out_path}", flush=True)
            results[ticker] = load_saved_ticker_report_frames(output_pdf, ticker)
            completed_tickers.append(str(ticker).upper())
            continue

        print(f"[analysis2] Running daily screen/sign report for {ticker}", flush=True)
        results[ticker] = run_local_nojump_mle_for_ticker(
            ticker=ticker,
            output_pdf=out_path,
            count_base_dir=count_base_dir,
            main_base_dir=main_base_dir,
            main_session_type=main_session_type,
            save_csv=save_csv,
        )
        completed_tickers.append(str(ticker).upper())

    if results:
        average_report = build_cross_asset_average_report(results)
        base_output = Path(output_pdf)
        summary_pdf = base_output.with_name(base_output.stem + "_all_assets_average_summary.pdf")
        write_cross_asset_average_pdf(summary_pdf, average_report, sorted(completed_tickers))

    return results


def generate_quantile_variant_reports(
    quantiles: tuple[float, ...] = (0.90,),
    output_dir: str | Path = "analysis2_quantile_variant_reports",
    tickers: list[str] | None = None,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    save_csv: bool = True,
) -> dict[str, object]:
    global INTRADAY_ABNORMAL_QS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tickers is None:
        tickers = base._get_common_tickers(
            count_base_dir=count_base_dir,
            main_base_dir=main_base_dir,
            main_session_type=main_session_type,
        )

    original_qs = INTRADAY_ABNORMAL_QS
    results: dict[str, object] = {}
    try:
        for q in quantiles:
            if float(q) >= 0.95:
                continue
            INTRADAY_ABNORMAL_QS = (float(q),)
            q_label = _qtag(q)
            q_dir = output_dir / q_label
            q_dir.mkdir(parents=True, exist_ok=True)
            q_results: dict[str, object] = {}
            for ticker in tickers:
                out_pdf = q_dir / f"daily_drift_screen_report_{q_label}_{ticker}.pdf"
                q_results[ticker] = run_local_nojump_mle_for_ticker(
                    ticker=ticker,
                    output_pdf=out_pdf,
                    count_base_dir=count_base_dir,
                    main_base_dir=main_base_dir,
                    main_session_type=main_session_type,
                    save_csv=save_csv,
                )
            results[q_label] = q_results
    finally:
        INTRADAY_ABNORMAL_QS = original_qs
    return {"output_dir": output_dir, "results": results}


def run_daily_below_mid_only_short_strategy_batch(
    output_dir: str | Path = "analysis2_batch_reports",
    tickers: list[str] | None = None,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    shares: int = DEFAULT_POSITION_SHARES,
    commission_per_trade: float = DEFAULT_COMMISSION_PER_TRADE,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if tickers is None:
        tickers = base._get_common_tickers(
            count_base_dir=count_base_dir,
            main_base_dir=main_base_dir,
            main_session_type=main_session_type,
        )

    all_results: dict[str, object] = {}
    summary_frames: list[pd.DataFrame] = []
    report_path = output_dir / "analysis2_daily_below_mid_only_short_backtest_report.pdf"

    with PdfPages(report_path) as pdf:
        base._pdf_text_page(
            pdf,
            "Strategy Definition",
            [
                "Signal: daily exclusive_location == below_mid_only from the existing analysis2 pipeline.",
                "Portfolio baseline: buy and hold by default.",
                f"When a daily below_mid_only signal appears, flip from long to short at the daily close and stay short until +{SHORT_STRATEGY_TARGET_BPS:.0f} bps profit or -{SHORT_STRATEGY_STOP_BPS:.0f} bps loss is reached, then revert to long.",
                "Comparison strategy: when the same signal appears, sell at the daily close, stay flat for 3 trading days, then buy back and resume buy-and-hold.",
                f"Position size: fixed {shares} shares per trade.",
                f"Commission per side: {commission_per_trade:.2f} dollars.",
                f"Spread cost: {SPREAD_COST_PER_SHARE_PER_SIDE:.2f} dollars/share at entry and {SPREAD_COST_PER_SHARE_PER_SIDE:.2f} dollars/share at exit.",
                "All reported strategy curves are net of trading costs.",
            ],
        )

        for ticker in tickers:
            result = run_daily_below_mid_signal_comparison_for_ticker(
                ticker=ticker,
                count_base_dir=count_base_dir,
                main_base_dir=main_base_dir,
                main_session_type=main_session_type,
                shares=shares,
                commission_per_trade=commission_per_trade,
            )
            all_results[ticker] = result
            trade_log = result["trade_log"]
            summary = result["summary"]
            equity_curve = result["equity_curve"]

            trade_log.to_csv(output_dir / f"analysis2_daily_below_mid_only_short_trades_{ticker}.csv", index=False)
            summary.to_csv(output_dir / f"analysis2_daily_below_mid_only_short_summary_{ticker}.csv", index=False)
            plot_short_strategy_equity_curve(ticker, equity_curve, output_dir / f"analysis2_daily_below_mid_only_short_equity_{ticker}.png")

            summary_frames.append(summary)
            short_row = summary[summary["strategy_name"] == "short_overlay_tp_sl"].iloc[0].to_dict() if len(summary[summary["strategy_name"] == "short_overlay_tp_sl"]) else {}
            flat_row = summary[summary["strategy_name"] == "sell_wait_3d_rebuy"].iloc[0].to_dict() if len(summary[summary["strategy_name"] == "sell_wait_3d_rebuy"]) else {}
            compare_note = {
                "short_overlay_cumulative_return_net": short_row.get("cumulative_return_net", np.nan),
                "sell_wait_3d_cumulative_return_net": flat_row.get("cumulative_return_net", np.nan),
                "short_overlay_max_drawdown_net": short_row.get("max_drawdown_net", np.nan),
                "sell_wait_3d_max_drawdown_net": flat_row.get("max_drawdown_net", np.nan),
                "short_overlay_number_of_trades": short_row.get("number_of_trades", np.nan),
                "sell_wait_3d_number_of_trades": flat_row.get("number_of_trades", np.nan),
                "position_size_shares": shares,
            }
            base._pdf_text_page(
                pdf,
                f"{ticker} Strategy Snapshot",
                [
                    f"History window note: {result['history_note']}",
                    f"Short overlay trades: {int(short_row.get('number_of_trades', 0) or 0)} | cumulative net return: {short_row.get('cumulative_return_net', np.nan)}",
                    f"Sell/rebuy 3D trades: {int(flat_row.get('number_of_trades', 0) or 0)} | cumulative net return: {flat_row.get('cumulative_return_net', np.nan)}",
                    f"First signal date: {min([d for d in [short_row.get('first_signal_date'), flat_row.get('first_signal_date')] if pd.notna(d)], default=pd.NaT)}",
                    f"Last signal date: {max([d for d in [short_row.get('last_signal_date'), flat_row.get('last_signal_date')] if pd.notna(d)], default=pd.NaT)}",
                    f"Short overlay total spread cost: {short_row.get('total_spread_cost', np.nan)} | commission: {short_row.get('total_commission_cost', np.nan)}",
                    f"Sell/rebuy 3D total spread cost: {flat_row.get('total_spread_cost', np.nan)} | commission: {flat_row.get('total_commission_cost', np.nan)}",
                ],
            )
            append_short_strategy_plot_to_pdf(ticker, equity_curve, pdf, compare_note)

        summary_table = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
        summary_table.to_csv(output_dir / "analysis2_daily_below_mid_only_short_summary.csv", index=False)
        if len(summary_table) > 0:
            base._pdf_dataframe_table_pages(
                pdf,
                base_title="Daily Below-Mid Signal Comparison",
                section_title="Per-Asset Summary",
                df=summary_table.round(6),
                max_rows_per_page=20,
                max_cols_per_page=8,
            )

    return {"results": all_results, "summary_table": summary_table, "report_path": report_path}


def generate_common_asset_pdf_package(
    output_dir: str | Path = "analysis2_batch_reports",
    package_name: str = "analysis2_common_assets_pdf_bundle",
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    save_csv: bool = False,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    common_tickers = base._get_common_tickers(
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
    )

    results = run_local_nojump_mle_batch(
        output_pdf=output_dir / "daily_drift_screen_report.pdf",
        tickers=common_tickers,
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
        main_session_type=main_session_type,
        save_csv=save_csv,
    )

    manifest_path = output_dir / "manifest.txt"
    manifest_lines = [
        "analysis2 common-asset PDF package",
        f"tickers: {', '.join(common_tickers)}",
        f"pdf_count: {len(common_tickers)}",
        f"save_csv: {save_csv}",
        "",
    ]
    manifest_lines.extend(
        str((output_dir / f"daily_drift_screen_report_{ticker}.pdf").name)
        for ticker in common_tickers
    )
    manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")

    archive_base = output_dir.parent / package_name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=output_dir)
    return {
        "tickers": common_tickers,
        "output_dir": output_dir,
        "archive_path": Path(archive_path),
        "results": results,
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Daily drift-screen analysis on VWAP with sign-based direction assignment and simple return summaries.")
    parser.add_argument("--ticker", default=None, help="Single ticker to run. If omitted, run batch over common tickers.")
    parser.add_argument("--tickers", default=None, help="Comma-separated ticker list for a partial batch run.")
    parser.add_argument("--output", default="daily_drift_screen_report.pdf", help="Base output PDF path.")
    parser.add_argument("--output-dir", default="analysis2_batch_reports", help="Directory used by the package-generation helper.")
    parser.add_argument("--images-only", action="store_true", help="Export report plots and journal-style tables as PNG files only.")
    parser.add_argument("--image-dir", default=None, help="Directory for --images-only PNG output. Defaults to a folder named after the ticker.")
    parser.add_argument("--package-common-assets", action="store_true", help="Generate one PDF per common ticker into a directory and zip the result.")
    parser.add_argument("--run-daily-below-mid-short", action="store_true", help="Run the daily below_mid_only short strategy batch.")
    parser.add_argument("--count-base-dir", default="data_2020_2025_count")
    parser.add_argument("--main-base-dir", default="data_2020_2025")
    parser.add_argument("--main-session-type", default="regular")
    parser.add_argument("--no-save-csv", action="store_true")
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    if args.run_daily_below_mid_short:
        run_daily_below_mid_only_short_strategy_batch(
            output_dir=args.output_dir,
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
        )
    elif args.package_common_assets:
        generate_common_asset_pdf_package(
            output_dir=args.output_dir,
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
            save_csv=not args.no_save_csv,
        )
    elif args.images_only and args.ticker:
        export_analysis2_images_for_ticker(
            ticker=args.ticker.upper(),
            image_dir=args.image_dir or args.ticker.lower(),
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
            save_csv=not args.no_save_csv,
        )
    elif args.ticker:
        run_local_nojump_mle_for_ticker(
            ticker=args.ticker.upper(),
            output_pdf=_derive_ticker_path(args.output, args.ticker.upper()),
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
            save_csv=not args.no_save_csv,
        )
    elif args.tickers:
        run_local_nojump_mle_batch(
            output_pdf=args.output,
            tickers=[x.strip().upper() for x in str(args.tickers).split(",") if str(x).strip()],
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
            save_csv=not args.no_save_csv,
            resume=True,
        )
    else:
        run_local_nojump_mle_batch(
            output_pdf=args.output,
            count_base_dir=args.count_base_dir,
            main_base_dir=args.main_base_dir,
            main_session_type=args.main_session_type,
            save_csv=not args.no_save_csv,
        )
