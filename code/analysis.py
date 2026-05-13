# Auto-generated from analysis.ipynb

# ===== Cell 0 =====
from pathlib import Path
import sys
import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats

DEFAULT_INTRADAY_HORIZONS = (5,) + tuple(range(10, 181, 10))


def _qtag(q):
    return f"q{int(round(q * 100))}"


def _safe_ttest_zero(x):
    s = pd.Series(x).dropna()
    n = len(s)
    if n < 2:
        return {
            "n": n,
            "mean": s.mean() if n > 0 else np.nan,
            "sd": np.nan,
            "tstat": np.nan,
            "pvalue": np.nan,
        }
    tstat, pvalue = stats.ttest_1samp(s, popmean=0.0, nan_policy="omit")
    return {
        "n": n,
        "mean": s.mean(),
        "sd": s.std(ddof=1),
        "tstat": tstat,
        "pvalue": pvalue,
    }


def _get_common_tickers(
    count_base_dir="data_2020_2025_count",
    main_base_dir="data_2020_2025",
    main_session_type="regular",
):
    count_base = Path(count_base_dir)
    main_base = Path(main_base_dir)

    count_tickers = {p.name.upper() for p in count_base.iterdir() if p.is_dir()}
    main_tickers = {p.name.upper() for p in main_base.iterdir() if p.is_dir()}
    valid = []

    for ticker in sorted(count_tickers & main_tickers):
        count_file = count_base / ticker / f"{ticker}_2020_2025_1min_count_regular.csv"
        main_file = main_base / ticker / f"{ticker}_2020_2025_1min_{main_session_type}.csv"
        if count_file.exists() and main_file.exists():
            valid.append(ticker)

    return valid


def _pdf_text_page(pdf, title, lines, figsize=(11, 8.5), title_fontsize=16, body_fontsize=10.5, line_height=0.032):
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    y = 0.96
    plt.text(0.03, y, title, fontsize=title_fontsize, fontweight="bold", va="top")
    y -= 0.06

    for line in lines:
        plt.text(0.03, y, str(line), fontsize=body_fontsize, va="top", family="monospace")
        y -= line_height
        if y < 0.05:
            pdf.savefig(fig)
            plt.close(fig)
            fig = plt.figure(figsize=figsize)
            plt.axis("off")
            y = 0.96

    pdf.savefig(fig)
    plt.close(fig)


def _save_current_figure_to_pdf(pdf):
    fig = plt.gcf()
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def _format_trigger_count_tables(result_dict):
    tables = []

    intraday_summary = result_dict["intraday"]["summary_df"]
    if len(intraday_summary) > 0:
        tables.append((
            "Intraday trigger counts:",
            intraday_summary[["pattern", "seq_window_mins", "q_label", "n_sequences"]]
            .sort_values(["pattern", "q_label", "seq_window_mins"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Intraday trigger counts:", pd.DataFrame({"note": ["No intraday triggers"]})))

    daily_summary = result_dict["daily"]["summary_df"]
    if len(daily_summary) > 0:
        tables.append((
            "Daily trigger counts:",
            daily_summary[["pattern", "q_label", "n_abnormal_days"]]
            .sort_values(["pattern", "q_label"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Daily trigger counts:", pd.DataFrame({"note": ["No daily triggers"]})))

    weekly_summary = result_dict["weekly"]["summary_df"]
    if len(weekly_summary) > 0:
        tables.append((
            "Weekly trigger counts:",
            weekly_summary[["pattern", "q_label", "n_abnormal_weeks"]]
            .sort_values(["pattern", "q_label"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Weekly trigger counts:", pd.DataFrame({"note": ["No weekly triggers"]})))

    imbalance_intraday_summary = result_dict["imbalance"]["intraday"]["summary_df"]
    if len(imbalance_intraday_summary) > 0:
        tables.append((
            "Intraday imbalance trigger counts:",
            imbalance_intraday_summary[["pattern", "seq_window_mins", "q_label", "n_sequences"]]
            .sort_values(["pattern", "q_label", "seq_window_mins"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Intraday imbalance trigger counts:", pd.DataFrame({"note": ["No intraday imbalance triggers"]})))

    imbalance_daily_summary = result_dict["imbalance"]["daily"]["summary_df"]
    if len(imbalance_daily_summary) > 0:
        tables.append((
            "Daily imbalance trigger counts:",
            imbalance_daily_summary[["pattern", "q_label", "n_abnormal_days"]]
            .sort_values(["pattern", "q_label"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Daily imbalance trigger counts:", pd.DataFrame({"note": ["No daily imbalance triggers"]})))

    imbalance_weekly_summary = result_dict["imbalance"]["weekly"]["summary_df"]
    if len(imbalance_weekly_summary) > 0:
        tables.append((
            "Weekly imbalance trigger counts:",
            imbalance_weekly_summary[["pattern", "q_label", "n_abnormal_weeks"]]
            .sort_values(["pattern", "q_label"])
            .reset_index(drop=True)
        ))
    else:
        tables.append(("Weekly imbalance trigger counts:", pd.DataFrame({"note": ["No weekly imbalance triggers"]})))

    return tables


def _chunk_list(items, chunk_size):
    chunk_size = max(int(chunk_size), 1)
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _derive_chunk_pdf_path(pdf_path, chunk_idx, chunk_total):
    pdf_path = Path(pdf_path)
    if chunk_total <= 1:
        return str(pdf_path)
    return str(pdf_path.with_name(f"{pdf_path.stem}_part{chunk_idx:02d}{pdf_path.suffix}"))


def _derive_ticker_pdf_path(pdf_path, ticker):
    pdf_path = Path(pdf_path)
    safe_ticker = str(ticker).strip().upper()
    return str(pdf_path.with_name(f"{pdf_path.stem}_{safe_ticker}{pdf_path.suffix}"))


def _chunk_dataframe_columns(df, max_cols_per_block=6):
    if df is None:
        return []
    columns = list(df.columns)
    if len(columns) == 0:
        return [df]
    col_chunks = _chunk_list(columns, max_cols_per_block)
    return [df.loc[:, cols].copy() for cols in col_chunks]


def _format_pdf_table_value(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{value:.6g}"
    return str(value)


def _wrap_pdf_table_text(text, width=14):
    text = str(text)
    if not text:
        return ""
    wrapped = textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)
    return wrapped if wrapped else text


def _render_pdf_table_page(
    pdf,
    title,
    df,
    note=None,
    figsize=(11, 8.5),
    title_fontsize=14,
    body_fontsize=8.2,
):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    fig.text(0.03, 0.965, title, fontsize=title_fontsize, fontweight="bold", va="top")
    if note:
        fig.text(0.03, 0.925, note, fontsize=10, va="top")

    display_df = df.copy()
    col_labels = [_wrap_pdf_table_text(col, width=14) for col in display_df.columns]
    cell_text = [
        [_wrap_pdf_table_text(_format_pdf_table_value(value), width=16) for value in row]
        for row in display_df.to_numpy()
    ]
    n_cols = max(len(display_df.columns), 1)
    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
        bbox=[0.03, 0.05, 0.94, 0.83],
        colWidths=[0.94 / n_cols] * n_cols,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(body_fontsize)
    table.scale(1, 1.45)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#EAEAEA")
            cell.set_text_props(weight="bold")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def _pdf_fixed_count_pages(pdf, title, tables, n_pages=3, body_fontsize=7.4, line_height=0.0195):
    for section_title, df_table in tables:
        _pdf_dataframe_table_pages(
            pdf,
            base_title=title,
            section_title=section_title.rstrip(":"),
            df=df_table,
            max_rows_per_page=int(np.ceil(max(len(df_table), 1) / n_pages)),
            max_cols_per_page=6,
            empty_message="No rows",
            body_fontsize=body_fontsize,
            line_height=line_height,
        )


def _pdf_dataframe_table_pages(
    pdf,
    base_title,
    section_title,
    df,
    max_rows_per_page=10,
    max_cols_per_page=6,
    empty_message="No rows",
    body_fontsize=8.2,
    line_height=0.024,
):
    if df is None or len(df) == 0:
        _pdf_text_page(
            pdf,
            title=f"{base_title} - {section_title}",
            lines=[empty_message],
            title_fontsize=14,
            body_fontsize=body_fontsize,
            line_height=line_height,
        )
        return

    row_ids = list(range(len(df)))
    row_chunks = _chunk_list(row_ids, max_rows_per_page)
    col_chunks = _chunk_dataframe_columns(df, max_cols_per_block=max_cols_per_page)
    total_pages = len(row_chunks) * len(col_chunks)
    page_idx = 0

    for row_block_idx, row_chunk in enumerate(row_chunks, start=1):
        sub = df.iloc[row_chunk].reset_index(drop=True)
        row_col_chunks = _chunk_dataframe_columns(sub, max_cols_per_block=max_cols_per_page)
        for col_block_idx, col_block in enumerate(row_col_chunks, start=1):
            page_idx += 1
            page_title = f"{base_title} - {section_title}"
            page_note = None
            if total_pages > 1:
                page_note = (
                    f"Page {page_idx}/{total_pages} | "
                    f"row block {row_block_idx}/{len(row_chunks)} | "
                    f"column block {col_block_idx}/{len(col_chunks)}"
                )
            _render_pdf_table_page(
                pdf,
                title=page_title,
                df=col_block,
                note=page_note,
                title_fontsize=14,
                body_fontsize=body_fontsize,
            )


def _render_progress(iteration, total, prefix="Progress", width=28, final=False):
    total = max(int(total), 1)
    iteration = min(max(int(iteration), 0), total)
    filled = int(width * iteration / total)
    bar = "#" * filled + "-" * (width - filled)
    end = "\n" if final or iteration >= total else "\r"
    print(f"{prefix} [{bar}] {iteration}/{total}", end=end, file=sys.stdout, flush=True)


def _compute_future_excursion_arrays(price_arr, horizons):
    price_s = pd.Series(price_arr, dtype="float64")
    rev = price_s.iloc[::-1]
    cache = {}

    for h in horizons:
        future_max = rev.shift(1).rolling(h, min_periods=1).max().iloc[::-1].to_numpy()
        future_min = rev.shift(1).rolling(h, min_periods=1).min().iloc[::-1].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            up = 100.0 * (future_max - price_arr) / price_arr
            down = 100.0 * (future_min - price_arr) / price_arr
        abs_move = np.maximum(np.abs(up), np.abs(down))
        cache[h] = {
            "up": up,
            "down": down,
            "abs": abs_move,
        }

    return cache


def _rolling_baseline_stats(values, normal_mask, baseline_window):
    valid = normal_mask & np.isfinite(values)
    s = pd.Series(np.where(valid, values, np.nan), dtype="float64")
    shifted = s.shift(1)
    mean = shifted.rolling(baseline_window, min_periods=1).mean().to_numpy()
    std = shifted.rolling(baseline_window, min_periods=2).std(ddof=1).to_numpy()
    return mean, std


def _future_excursion_pct_from_index(price_arr, idx, horizon):
    p0 = price_arr[idx]
    if np.isnan(p0) or p0 == 0:
        return np.nan, np.nan, np.nan

    end = min(len(price_arr), idx + horizon + 1)
    future = price_arr[idx + 1:end]
    future = future[~np.isnan(future)]

    if len(future) == 0:
        return np.nan, np.nan, np.nan

    up_pct = 100.0 * (future.max() - p0) / p0
    down_pct = 100.0 * (future.min() - p0) / p0
    abs_pct = max(abs(up_pct), abs(down_pct))
    return up_pct, down_pct, abs_pct

# ===== Cell 1 =====
def load_count_aligned_dataset(
    ticker: str,
    count_base_dir: str = "data_2020_2025_count",
    main_base_dir: str = "data_2020_2025",
    main_session_type: str = "regular",
    first_time: str = "09:30:00",
):
    ticker = ticker.upper().strip()
    main_session_type = main_session_type.lower().strip()

    count_file = Path(count_base_dir) / ticker / f"{ticker}_2020_2025_1min_count_regular.csv"
    main_file = Path(main_base_dir) / ticker / f"{ticker}_2020_2025_1min_{main_session_type}.csv"

    count_df = pd.read_csv(count_file)
    main_df = pd.read_csv(main_file)

    if "minute_dt" in count_df.columns:
        count_df["minute_dt"] = pd.to_datetime(count_df["minute_dt"])
    if "minute_dt" in main_df.columns:
        main_df["minute_dt"] = pd.to_datetime(main_df["minute_dt"])

    if "date" in count_df.columns:
        count_df["date"] = pd.to_datetime(count_df["date"]).dt.date
    elif "minute_dt" in count_df.columns:
        count_df["date"] = count_df["minute_dt"].dt.date

    if "date" in main_df.columns:
        main_df["date"] = pd.to_datetime(main_df["date"]).dt.date
    elif "minute_dt" in main_df.columns:
        main_df["date"] = main_df["minute_dt"].dt.date

    if "time_m" not in count_df.columns and "minute_dt" in count_df.columns:
        count_df["time_m"] = count_df["minute_dt"].dt.strftime("%H:%M:%S")
    if "time_m" not in main_df.columns and "minute_dt" in main_df.columns:
        main_df["time_m"] = main_df["minute_dt"].dt.strftime("%H:%M:%S")

    count_df["if_first_min"] = (count_df["time_m"] == first_time).astype(int)

    merge_keys = [c for c in ["minute_dt", "date", "time_m", "sym_root"] if c in count_df.columns and c in main_df.columns]

    main_keep_cols = merge_keys + [c for c in main_df.columns if c not in count_df.columns]
    main_df_small = main_df[main_keep_cols].copy()

    merged_df = count_df.merge(main_df_small, on=merge_keys, how="left")
    merged_df = merged_df.sort_values("minute_dt").reset_index(drop=True)

    return merged_df

def _copy_and_prepare(df):
    out = df.copy()
    out["minute_dt"] = pd.to_datetime(out["minute_dt"])
    out = out.sort_values("minute_dt").reset_index(drop=True)

    if "date" not in out.columns:
        out["date"] = out["minute_dt"].dt.date

    if "vwap" in out.columns:
        out["vwap_diff"] = out.groupby("date")["vwap"].diff()

    if {"ask", "bid"}.issubset(out.columns):
        out["spread"] = out["ask"] - out["bid"]

    out["month"] = out["minute_dt"].dt.to_period("M").dt.to_timestamp()
    return out


def _sample_series(s, max_points=300000, clip_q=0.995, dropna=True):
    x = s.copy()
    if dropna:
        x = x.dropna()

    if len(x) == 0:
        return x

    lo = x.quantile(1 - clip_q)
    hi = x.quantile(clip_q)
    x = x[(x >= lo) & (x <= hi)]

    if len(x) > max_points:
        x = x.sample(max_points, random_state=42)

    return x


def _safe_corr_stats(x, y):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(z)
    if n < 3:
        return {"n": n, "corr": np.nan, "pvalue": np.nan}

    corr, pvalue = stats.pearsonr(z["x"], z["y"])
    return {"n": n, "corr": corr, "pvalue": pvalue}


def _safe_ks_2samp(x, y):
    xs = pd.Series(x).dropna()
    ys = pd.Series(y).dropna()
    if len(xs) < 2 or len(ys) < 2:
        return {"n_x": len(xs), "n_y": len(ys), "stat": np.nan, "pvalue": np.nan}
    stat, pvalue = stats.ks_2samp(xs, ys, alternative="two-sided", method="auto")
    return {"n_x": len(xs), "n_y": len(ys), "stat": stat, "pvalue": pvalue}


def _get_total_volume_series(x):
    if "trade_volume_total" in x.columns:
        return pd.to_numeric(x["trade_volume_total"], errors="coerce")

    side_cols = [c for c in ["trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"] if c in x.columns]
    if len(side_cols) == 0:
        return pd.Series(np.nan, index=x.index, dtype="float64")

    out = pd.Series(0.0, index=x.index, dtype="float64")
    for col in side_cols:
        out = out.add(pd.to_numeric(x[col], errors="coerce"), fill_value=0.0)
    return out


def _volume_series_specs(x):
    specs = [("total_volume", _get_total_volume_series(x), "Total Volume")]
    for col, label in [
        ("trade_volume_above_mid", "Above Mid Volume"),
        ("trade_volume_at_mid", "At Mid Volume"),
        ("trade_volume_below_mid", "Below Mid Volume"),
    ]:
        if col in x.columns:
            specs.append((col, pd.to_numeric(x[col], errors="coerce"), label))
    return specs


def build_intraday_drift_volume_panel(
    df,
    price_col="vwap",
    unit_mins=20,
    rolling_window=20,
):
    x = _copy_and_prepare(df)
    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    vol = _get_total_volume_series(x)
    vol_unit = vol.rolling(unit_mins, min_periods=unit_mins).sum()

    panel = pd.DataFrame({
        "time": x["minute_dt"],
        "drift_pct": 100.0 * (x[price_col] - x[price_col].shift(unit_mins)) / x[price_col].shift(unit_mins),
        "abs_drift_pct": (100.0 * (x[price_col] - x[price_col].shift(unit_mins)) / x[price_col].shift(unit_mins)).abs(),
        "volume_change": np.log1p(vol_unit).diff(),
    })
    panel["abs_volume_change"] = panel["volume_change"].abs()
    panel["rolling_corr_signed"] = panel["drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(panel["volume_change"])
    panel["rolling_corr_abs"] = panel["abs_drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(panel["abs_volume_change"])
    return panel


def build_daily_drift_volume_panel(
    df,
    price_col="vwap",
    rolling_window=20,
):
    daily = build_daily_panel_for_excursion(df=df, price_col=price_col)
    vol = _get_total_volume_series(_copy_and_prepare(df))
    daily_vol = (
        _copy_and_prepare(df)
        .assign(_total_vol=vol)
        .groupby("date", as_index=False)["_total_vol"]
        .sum()
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily = daily.merge(daily_vol, on="date", how="left")
    daily["drift_pct"] = 100.0 * (daily["close_price"] - daily["close_price"].shift(1)) / daily["close_price"].shift(1)
    daily["abs_drift_pct"] = daily["drift_pct"].abs()
    daily["volume_change"] = np.log1p(daily["_total_vol"]).diff()
    daily["abs_volume_change"] = daily["volume_change"].abs()
    daily["rolling_corr_signed"] = daily["drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(daily["volume_change"])
    daily["rolling_corr_abs"] = daily["abs_drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(daily["abs_volume_change"])
    daily["time"] = pd.to_datetime(daily["date"])
    return daily


def build_weekly_drift_volume_panel(
    df,
    price_col="vwap",
    rolling_window=20,
):
    weekly = build_weekly_panel_for_excursion(df=df, price_col=price_col)
    base = _copy_and_prepare(df)
    vol = _get_total_volume_series(base)
    weekly_vol = (
        base.assign(_total_vol=vol, week=base["minute_dt"].dt.to_period("W").astype(str))
        .groupby("week", as_index=False)["_total_vol"]
        .sum()
        .sort_values("week")
        .reset_index(drop=True)
    )
    weekly = weekly.merge(weekly_vol, on="week", how="left")
    weekly["drift_pct"] = 100.0 * (weekly["close_price"] - weekly["close_price"].shift(1)) / weekly["close_price"].shift(1)
    weekly["abs_drift_pct"] = weekly["drift_pct"].abs()
    weekly["volume_change"] = np.log1p(weekly["_total_vol"]).diff()
    weekly["abs_volume_change"] = weekly["volume_change"].abs()
    weekly["rolling_corr_signed"] = weekly["drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(weekly["volume_change"])
    weekly["rolling_corr_abs"] = weekly["abs_drift_pct"].rolling(rolling_window, min_periods=max(5, rolling_window // 2)).corr(weekly["abs_volume_change"])
    weekly["time"] = pd.PeriodIndex(weekly["week"], freq="W").to_timestamp()
    return weekly


def summarize_drift_volume_panel(panel, layer_name):
    signed = _safe_corr_stats(panel["drift_pct"], panel["volume_change"])
    abs_stats = _safe_corr_stats(panel["abs_drift_pct"], panel["abs_volume_change"])
    return pd.DataFrame([{
        "layer": layer_name,
        "n_obs": int(panel[["drift_pct", "volume_change"]].dropna().shape[0]),
        "signed_corr": signed["corr"],
        "signed_pvalue": signed["pvalue"],
        "abs_corr": abs_stats["corr"],
        "abs_pvalue": abs_stats["pvalue"],
        "rolling_signed_corr_mean": panel["rolling_corr_signed"].mean(),
        "rolling_abs_corr_mean": panel["rolling_corr_abs"].mean(),
    }])


def build_intraday_volume_max_drift_panels(
    df,
    price_col="vwap",
    unit_mins=20,
):
    x = _copy_and_prepare(df)
    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    price_arr = x[price_col].to_numpy(dtype="float64")
    excursion = _compute_future_excursion_arrays(price_arr, (unit_mins,))[unit_mins]
    out = {}
    for vol_key, vol_series, vol_label in _volume_series_specs(x):
        vol_unit = vol_series.rolling(unit_mins, min_periods=unit_mins).sum()
        out[vol_key] = {
            "label": vol_label,
            "panel": pd.DataFrame({
                "time": x["minute_dt"],
                "volume_value": vol_unit,
                "max_up_drift_pct": excursion["up"],
                "max_down_drift_pct": excursion["down"],
            }),
        }
    return out


def build_daily_volume_max_drift_panels(
    df,
    price_col="vwap",
):
    daily = build_daily_panel_for_excursion(df=df, price_col=price_col)
    base = _copy_and_prepare(df)
    out = {}
    for vol_key, vol_series, vol_label in _volume_series_specs(base):
        daily_vol = (
            base.assign(_vol=vol_series)
            .groupby("date", as_index=False)["_vol"]
            .sum()
            .sort_values("date")
            .reset_index(drop=True)
        )
        panel = daily.merge(daily_vol, on="date", how="left")
        panel["time"] = pd.to_datetime(panel["date"])
        panel = panel.rename(columns={
            "_vol": "volume_value",
            "next_day_max_up_pct": "max_up_drift_pct",
            "next_day_max_down_pct": "max_down_drift_pct",
        })
        out[vol_key] = {
            "label": vol_label,
            "panel": panel[["time", "volume_value", "max_up_drift_pct", "max_down_drift_pct"]].copy(),
        }
    return out


def build_weekly_volume_max_drift_panels(
    df,
    price_col="vwap",
):
    weekly = build_weekly_panel_for_excursion(df=df, price_col=price_col)
    base = _copy_and_prepare(df)
    out = {}
    for vol_key, vol_series, vol_label in _volume_series_specs(base):
        weekly_vol = (
            base.assign(_vol=vol_series, week=base["minute_dt"].dt.to_period("W").astype(str))
            .groupby("week", as_index=False)["_vol"]
            .sum()
            .sort_values("week")
            .reset_index(drop=True)
        )
        panel = weekly.merge(weekly_vol, on="week", how="left")
        panel["time"] = pd.PeriodIndex(panel["week"], freq="W").to_timestamp()
        panel = panel.rename(columns={
            "_vol": "volume_value",
            "next_week_max_up_pct": "max_up_drift_pct",
            "next_week_max_down_pct": "max_down_drift_pct",
        })
        out[vol_key] = {
            "label": vol_label,
            "panel": panel[["time", "volume_value", "max_up_drift_pct", "max_down_drift_pct"]].copy(),
        }
    return out


def run_volume_max_drift_analysis(
    df,
    price_col="vwap",
    intraday_unit_mins=(30, 60),
):
    intraday_windows = tuple(int(v) for v in intraday_unit_mins)
    return {
        "intraday": {
            window: build_intraday_volume_max_drift_panels(df=df, price_col=price_col, unit_mins=window)
            for window in intraday_windows
        },
        "daily": build_daily_volume_max_drift_panels(df=df, price_col=price_col),
        "weekly": build_weekly_volume_max_drift_panels(df=df, price_col=price_col),
    }


def _plot_regression_scatter(ax, x, y, title, xlabel, ylabel, max_plot_points=2500):
    z = pd.DataFrame({"x": x, "y": y}).dropna()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if len(z) == 0:
        return

    x_plot = np.log1p(z["x"].to_numpy(dtype="float64"))
    y_plot = z["y"].to_numpy(dtype="float64")
    stats_out = _safe_corr_stats(x_plot, y_plot)
    if len(z) >= 2 and np.nanmin(x_plot) != np.nanmax(x_plot):
        slope, intercept = np.polyfit(x_plot, y_plot, 1)
        x_line = np.linspace(np.nanmin(x_plot), np.nanmax(x_plot), 100)
        ax.plot(x_line, intercept + slope * x_line, color="black", linewidth=1.2)

    plot_df = pd.DataFrame({"x": x_plot, "y": y_plot})
    if len(plot_df) > max_plot_points:
        plot_df = plot_df.sample(max_plot_points, random_state=42)

    ax.scatter(
        plot_df["x"],
        plot_df["y"],
        s=9,
        alpha=0.45,
        facecolors="white",
        edgecolors="black",
        linewidths=0.35,
        rasterized=True,
    )

    ax.grid(alpha=0.25)
    ax.text(
        0.02,
        0.97,
        f"corr={stats_out['corr']:.3f}  p={stats_out['pvalue']:.3g}  n={stats_out['n']}  plot_n={len(plot_df)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85, "pad": 2.5},
    )


def plot_volume_max_drift_layer_to_pdf(panel, pdf, ticker, layer_name, volume_label):
    if len(panel) == 0:
        return

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)
    _plot_regression_scatter(
        axes[0],
        panel["volume_value"],
        panel["max_up_drift_pct"],
        title=f"{ticker} - {layer_name.title()} {volume_label} vs Max Up Drift",
        xlabel=f"log(1 + {volume_label.lower()})",
        ylabel="Max up drift (%)",
    )
    _plot_regression_scatter(
        axes[1],
        panel["volume_value"],
        panel["max_down_drift_pct"],
        title=f"{ticker} - {layer_name.title()} {volume_label} vs Max Down Drift",
        xlabel=f"log(1 + {volume_label.lower()})",
        ylabel="Max down drift (%)",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    _save_current_figure_to_pdf(pdf)


def build_volume_drift_pvalue_table(volume_max_drift_result):
    rows = []

    for window, volume_map in sorted(volume_max_drift_result["intraday"].items()):
        for volume_key, volume_item in volume_map.items():
            panel = volume_item["panel"]
            up_stats = _safe_corr_stats(np.log1p(panel["volume_value"]), panel["max_up_drift_pct"])
            down_stats = _safe_corr_stats(np.log1p(panel["volume_value"]), panel["max_down_drift_pct"])
            rows.append({
                "layer": "intraday",
                "window": f"{window}m",
                "volume": volume_key,
                "n_up": up_stats["n"],
                "up_corr_p": up_stats["pvalue"],
                "n_down": down_stats["n"],
                "down_corr_p": down_stats["pvalue"],
            })

    for layer_name in ["daily", "weekly"]:
        for volume_key, volume_item in volume_max_drift_result[layer_name].items():
            panel = volume_item["panel"]
            up_stats = _safe_corr_stats(np.log1p(panel["volume_value"]), panel["max_up_drift_pct"])
            down_stats = _safe_corr_stats(np.log1p(panel["volume_value"]), panel["max_down_drift_pct"])
            rows.append({
                "layer": layer_name,
                "window": layer_name,
                "volume": volume_key,
                "n_up": up_stats["n"],
                "up_corr_p": up_stats["pvalue"],
                "n_down": down_stats["n"],
                "down_corr_p": down_stats["pvalue"],
            })

    return pd.DataFrame(rows)


def build_intraday_pvalue_table(
    seq_outcome_df,
    df_feat,
    q_labels=(90, 95, 99),
    seq_windows=(30, 60),
):
    rows = []
    if len(seq_outcome_df) == 0:
        return pd.DataFrame()

    price_arr = pd.to_numeric(df_feat["price"], errors="coerce").to_numpy(dtype="float64")
    excursion_cache = _compute_future_excursion_arrays(price_arr, seq_windows)

    for pattern in ["trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"]:
        for seq_window in seq_windows:
            for q_label in q_labels:
                tag = f"q{int(q_label)}"
                g = seq_outcome_df[
                    (seq_outcome_df["pattern"] == pattern)
                    & (seq_outcome_df["seq_window_mins"] == seq_window)
                    & (seq_outcome_df["q_label"] == q_label)
                ].copy()
                if len(g) == 0:
                    continue

                abn_flag_col = f"abn_{pattern}_{seq_window}m_{tag}"
                if abn_flag_col not in df_feat.columns:
                    continue
                normal_mask = df_feat[abn_flag_col].fillna(0).astype(int).to_numpy() == 0
                all_mask = np.isfinite(excursion_cache[seq_window]["up"])

                for side in ["up", "down", "abs"]:
                    abn_col = f"future_max_{side}_{seq_window}m_pct"
                    base_col = f"baseline_max_{side}_{seq_window}m_pct"
                    abn_vals = g[abn_col]
                    base_dist = pd.Series(excursion_cache[seq_window][side][normal_mask])
                    all_dist = pd.Series(excursion_cache[seq_window][side][all_mask])
                    t_stats = _safe_ttest_zero(g[abn_col] - g[base_col])
                    ks_base = _safe_ks_2samp(abn_vals, base_dist)
                    ks_all = _safe_ks_2samp(abn_vals, all_dist)
                    rows.append({
                        "layer": "intraday",
                        "window": f"{seq_window}m",
                        "pattern": pattern,
                        "q": tag,
                        "side": side,
                        "n_abn": len(pd.Series(abn_vals).dropna()),
                        "t_p_vs_base": t_stats["pvalue"],
                        "ks_p_vs_base": ks_base["pvalue"],
                        "ks_p_vs_all": ks_all["pvalue"],
                    })

    return pd.DataFrame(rows)


def build_daily_pvalue_table(
    daily_df,
    q_labels=(90, 95, 99),
):
    rows = []
    if len(daily_df) == 0:
        return pd.DataFrame()

    for pattern in ["trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"]:
        for q_label in q_labels:
            tag = f"q{int(q_label)}"
            abn_flag_col = f"abn_{pattern}_{tag}"
            if abn_flag_col not in daily_df.columns:
                continue
            abn = daily_df[daily_df[abn_flag_col].fillna(0).astype(int) == 1].copy()
            baseline = daily_df[daily_df[abn_flag_col].fillna(0).astype(int) == 0].copy()
            if len(abn) == 0:
                continue

            for side in ["up", "down", "abs"]:
                abn_col = f"next_day_max_{side}_pct"
                base_col = f"baseline_next_day_max_{side}_{pattern}_{tag}_pct"
                t_stats = _safe_ttest_zero(abn[abn_col] - abn[base_col])
                ks_base = _safe_ks_2samp(abn[abn_col], baseline[abn_col])
                ks_all = _safe_ks_2samp(abn[abn_col], daily_df[abn_col])
                rows.append({
                    "layer": "daily",
                    "window": "day",
                    "pattern": pattern,
                    "q": tag,
                    "side": side,
                    "n_abn": len(pd.Series(abn[abn_col]).dropna()),
                    "t_p_vs_base": t_stats["pvalue"],
                    "ks_p_vs_base": ks_base["pvalue"],
                    "ks_p_vs_all": ks_all["pvalue"],
                })

    return pd.DataFrame(rows)


def build_weekly_pvalue_table(
    weekly_df,
    q_labels=(90, 95, 99),
):
    rows = []
    if len(weekly_df) == 0:
        return pd.DataFrame()

    for pattern in ["trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"]:
        for q_label in q_labels:
            tag = f"q{int(q_label)}"
            abn_flag_col = f"abn_{pattern}_{tag}"
            if abn_flag_col not in weekly_df.columns:
                continue
            abn = weekly_df[weekly_df[abn_flag_col].fillna(0).astype(int) == 1].copy()
            baseline = weekly_df[weekly_df[abn_flag_col].fillna(0).astype(int) == 0].copy()
            if len(abn) == 0:
                continue

            for side in ["up", "down", "abs"]:
                abn_col = f"next_week_max_{side}_pct"
                base_col = f"baseline_next_week_max_{side}_{pattern}_{tag}_pct"
                t_stats = _safe_ttest_zero(abn[abn_col] - abn[base_col])
                ks_base = _safe_ks_2samp(abn[abn_col], baseline[abn_col])
                ks_all = _safe_ks_2samp(abn[abn_col], weekly_df[abn_col])
                rows.append({
                    "layer": "weekly",
                    "window": "week",
                    "pattern": pattern,
                    "q": tag,
                    "side": side,
                    "n_abn": len(pd.Series(abn[abn_col]).dropna()),
                    "t_p_vs_base": t_stats["pvalue"],
                    "ks_p_vs_base": ks_base["pvalue"],
                    "ks_p_vs_all": ks_all["pvalue"],
                })

    return pd.DataFrame(rows)


def _prettify_pvalue_table(df):
    if df is None or len(df) == 0:
        return df
    out = df.copy()
    if "pattern" in out.columns:
        out["pattern"] = out["pattern"].replace({
            "trade_volume_below_mid": "below_mid",
            "trade_volume_above_mid": "above_mid",
            "trade_volume_at_mid": "at_mid",
        })
    if "volume" in out.columns:
        out["volume"] = out["volume"].replace({
            "total_volume": "total",
            "trade_volume_below_mid": "below_mid",
            "trade_volume_above_mid": "above_mid",
            "trade_volume_at_mid": "at_mid",
        })
    return out


def run_drift_volume_link_analysis(
    df,
    price_col="vwap",
    intraday_unit_mins=20,
    rolling_window=20,
):
    intraday = build_intraday_drift_volume_panel(df=df, price_col=price_col, unit_mins=intraday_unit_mins, rolling_window=rolling_window)
    daily = build_daily_drift_volume_panel(df=df, price_col=price_col, rolling_window=rolling_window)
    weekly = build_weekly_drift_volume_panel(df=df, price_col=price_col, rolling_window=rolling_window)
    summary = pd.concat([
        summarize_drift_volume_panel(intraday, "intraday"),
        summarize_drift_volume_panel(daily, "daily"),
        summarize_drift_volume_panel(weekly, "weekly"),
    ], ignore_index=True)
    return {
        "intraday": intraday,
        "daily": daily,
        "weekly": weekly,
        "summary": summary,
    }


def build_intraday_imbalance_features_multiq(
    df,
    above_col="trade_volume_above_mid",
    below_col="trade_volume_below_mid",
    seq_windows=(5, 10, 20, 30),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=390 * 20,
):
    x = _copy_and_prepare(df)
    x["price"] = pd.to_numeric(x["vwap"], errors="coerce")
    x["spread"] = pd.to_numeric(x.get("spread"), errors="coerce")
    x[above_col] = pd.to_numeric(x[above_col], errors="coerce")
    x[below_col] = pd.to_numeric(x[below_col], errors="coerce")

    imbalance = x[above_col] - x[below_col]
    min_hist = max(200, baseline_window // 5)
    new_cols = {}

    for w in seq_windows:
        cum_s = imbalance.rolling(w, min_periods=w).sum()
        base_abs_mean_s = cum_s.abs().shift(1).rolling(baseline_window, min_periods=min_hist).mean()
        rel_s = cum_s.abs() / base_abs_mean_s

        for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
            new_cols[f"cumseq_{pattern}_{w}m"] = cum_s
            new_cols[f"baseline_mean_{pattern}_{w}m"] = base_abs_mean_s
            new_cols[f"rel_to_mean_{pattern}_{w}m"] = rel_s

        for q in abnormal_qs:
            tag = _qtag(q)
            upper_q = cum_s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(q)
            lower_q = cum_s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(1 - q)
            new_cols[f"baseline_q_imbalance_above_mid_{w}m_{tag}"] = upper_q
            new_cols[f"baseline_q_imbalance_below_mid_{w}m_{tag}"] = lower_q
            new_cols[f"abn_imbalance_above_mid_{w}m_{tag}"] = (cum_s >= upper_q).astype(int)
            new_cols[f"abn_imbalance_below_mid_{w}m_{tag}"] = (cum_s <= lower_q).astype(int)

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_intraday_imbalance_event_baselines_multiq(
    seq_outcome_df,
    df_feat,
    horizons=DEFAULT_INTRADAY_HORIZONS,
    baseline_window=390 * 20,
    excursion_cache=None,
    progress_prefix=None,
):
    if len(seq_outcome_df) == 0:
        return seq_outcome_df.copy()

    out = seq_outcome_df.copy()
    price_arr = df_feat["price"].to_numpy(dtype="float64")
    cache = excursion_cache if excursion_cache is not None else _compute_future_excursion_arrays(price_arr, horizons)

    init_cols = {}
    for h in horizons:
        init_cols[f"baseline_max_up_{h}m_pct"] = np.nan
        init_cols[f"baseline_max_down_{h}m_pct"] = np.nan
        init_cols[f"baseline_max_abs_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_up_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_down_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_abs_{h}m_pct"] = np.nan
    out = pd.concat([out, pd.DataFrame(init_cols, index=out.index)], axis=1).copy()

    combos = list(out[["pattern", "seq_window_mins", "q_label"]].drop_duplicates().itertuples(index=False, name=None))
    combo_total = len(combos)

    for combo_idx, (pattern, seq_window, q_label) in enumerate(combos, start=1):
        if progress_prefix:
            _render_progress(combo_idx - 1, combo_total, prefix=progress_prefix)

        above_col = f"abn_imbalance_above_mid_{int(seq_window)}m_q{int(q_label)}"
        below_col = f"abn_imbalance_below_mid_{int(seq_window)}m_q{int(q_label)}"
        normal_mask = (
            (df_feat[above_col].fillna(0).to_numpy() == 0)
            & (df_feat[below_col].fillna(0).to_numpy() == 0)
        )
        combo_mask = (
            (out["pattern"] == pattern)
            & (out["seq_window_mins"] == seq_window)
            & (out["q_label"] == q_label)
        )
        event_indices = out.loc[combo_mask, "end_idx"].astype(int).to_numpy()
        target_rows = out.index[combo_mask]

        for h in horizons:
            mean_up, std_up = _rolling_baseline_stats(cache[h]["up"], normal_mask, baseline_window)
            mean_down, std_down = _rolling_baseline_stats(cache[h]["down"], normal_mask, baseline_window)
            mean_abs, std_abs = _rolling_baseline_stats(cache[h]["abs"], normal_mask, baseline_window)

            out.loc[target_rows, f"baseline_max_up_{h}m_pct"] = mean_up[event_indices]
            out.loc[target_rows, f"baseline_max_down_{h}m_pct"] = mean_down[event_indices]
            out.loc[target_rows, f"baseline_max_abs_{h}m_pct"] = mean_abs[event_indices]
            out.loc[target_rows, f"baseline_sd_up_{h}m_pct"] = std_up[event_indices]
            out.loc[target_rows, f"baseline_sd_down_{h}m_pct"] = std_down[event_indices]
            out.loc[target_rows, f"baseline_sd_abs_{h}m_pct"] = std_abs[event_indices]

    if progress_prefix:
        _render_progress(combo_total, combo_total, prefix=progress_prefix, final=True)

    return out


def run_intraday_imbalance_effect_module_multiq(
    df,
    seq_windows=(5, 10, 20, 30),
    abnormal_qs=(0.90, 0.95, 0.99),
    horizons=DEFAULT_INTRADAY_HORIZONS,
    baseline_window=390 * 20,
    min_run=2,
    progress_prefix=None,
):
    df_feat = build_intraday_imbalance_features_multiq(
        df=df,
        seq_windows=seq_windows,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    seq_df = build_intraday_sequence_table_multiq(
        df_feat=df_feat,
        vol_cols=("imbalance_above_mid", "imbalance_below_mid"),
        seq_windows=seq_windows,
        abnormal_qs=abnormal_qs,
        min_run=min_run,
    )

    excursion_cache = _compute_future_excursion_arrays(df_feat["price"].to_numpy(dtype="float64"), horizons)

    seq_outcome_df = attach_intraday_future_excursions_multiq(
        seq_df=seq_df,
        df_feat=df_feat,
        horizons=horizons,
        excursion_cache=excursion_cache,
    )

    seq_outcome_df = build_intraday_imbalance_event_baselines_multiq(
        seq_outcome_df=seq_outcome_df,
        df_feat=df_feat,
        horizons=horizons,
        baseline_window=baseline_window,
        excursion_cache=excursion_cache,
        progress_prefix=progress_prefix,
    )

    summary_df = summarize_intraday_multiq(
        seq_outcome_df=seq_outcome_df,
        horizons=horizons,
        global_future_means={
            (side, h): pd.Series(excursion_cache[h][side]).dropna().mean()
            for h in horizons
            for side in ["up", "down", "abs"]
        },
    )

    return df_feat, seq_df, seq_outcome_df, summary_df


def build_daily_imbalance_panel_for_excursion(df, price_col="vwap"):
    daily = build_daily_panel_for_excursion(df=df, price_col=price_col)
    daily["imbalance_diff"] = daily["trade_volume_above_mid"] - daily["trade_volume_below_mid"]
    daily["imbalance_abs"] = daily["imbalance_diff"].abs()
    return daily


def add_daily_imbalance_flags_multiq(
    daily_df,
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = daily_df.copy()
    min_hist = max(5, baseline_window // 4)
    s = pd.to_numeric(x["imbalance_diff"], errors="coerce")
    abs_base_mean = s.abs().shift(1).rolling(baseline_window, min_periods=min_hist).mean()

    new_cols = {}
    for side in ["imbalance_above_mid", "imbalance_below_mid"]:
        new_cols[f"baseline_mean_{side}"] = abs_base_mean
        new_cols[f"rel_{side}"] = s.abs() / abs_base_mean

    for q in abnormal_qs:
        tag = _qtag(q)
        upper_q = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(q)
        lower_q = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(1 - q)
        new_cols[f"baseline_q_imbalance_above_mid_{tag}"] = upper_q
        new_cols[f"baseline_q_imbalance_below_mid_{tag}"] = lower_q
        new_cols[f"abn_imbalance_above_mid_{tag}"] = (s >= upper_q).astype(int)
        new_cols[f"abn_imbalance_below_mid_{tag}"] = (s <= lower_q).astype(int)

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_daily_imbalance_event_baselines_multiq(
    daily_df,
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = daily_df.copy()

    init_cols = {}
    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)
            init_cols[f"baseline_next_day_max_up_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_max_down_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_max_abs_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_up_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_down_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_abs_{pattern}_{tag}_pct"] = np.nan
    x = pd.concat([x, pd.DataFrame(init_cols, index=x.index)], axis=1).copy()

    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)

            any_abn_col = (
                x[f"abn_imbalance_above_mid_{tag}"].fillna(0).astype(int)
                | x[f"abn_imbalance_below_mid_{tag}"].fillna(0).astype(int)
            )

            for i in x.index:
                start = max(0, i - baseline_window)
                hist = x.iloc[start:i].copy()
                if len(hist) == 0:
                    continue

                hist = hist[any_abn_col.iloc[start:i].to_numpy() == 0].copy()
                if len(hist) == 0:
                    continue

                x.at[i, f"baseline_next_day_max_up_{pattern}_{tag}_pct"] = hist["next_day_max_up_pct"].mean()
                x.at[i, f"baseline_next_day_max_down_{pattern}_{tag}_pct"] = hist["next_day_max_down_pct"].mean()
                x.at[i, f"baseline_next_day_max_abs_{pattern}_{tag}_pct"] = hist["next_day_max_abs_pct"].mean()
                x.at[i, f"baseline_next_day_sd_up_{pattern}_{tag}_pct"] = hist["next_day_max_up_pct"].std(ddof=1)
                x.at[i, f"baseline_next_day_sd_down_{pattern}_{tag}_pct"] = hist["next_day_max_down_pct"].std(ddof=1)
                x.at[i, f"baseline_next_day_sd_abs_{pattern}_{tag}_pct"] = hist["next_day_max_abs_pct"].std(ddof=1)

    return x


def summarize_daily_imbalance_multiq(
    daily_df,
    abnormal_qs=(0.90, 0.95, 0.99),
):
    rows = []

    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)
            abn = daily_df[daily_df[f"abn_{pattern}_{tag}"] == 1].copy()
            if len(abn) == 0:
                continue

            row = {
                "pattern": pattern,
                "q_label": int(round(q * 100)),
                "n_abnormal_days": len(abn),
                "avg_rel_to_baseline": abn[f"rel_{pattern}"].mean(),
                "current_window_max_up_pct": abn["current_window_max_up_pct"].mean(),
                "current_window_max_down_pct": abn["current_window_max_down_pct"].mean(),
                "current_window_max_abs_pct": abn["current_window_max_abs_pct"].mean(),
            }

            for side in ["up", "down", "abs"]:
                abn_col = f"next_day_max_{side}_pct"
                base_col = f"baseline_next_day_max_{side}_{pattern}_{tag}_pct"
                diff = abn[abn_col] - abn[base_col]
                tt = _safe_ttest_zero(diff)

                row[f"abn_next_day_max_{side}_pct"] = abn[abn_col].mean()
                row[f"baseline_next_day_max_{side}_pct"] = abn[base_col].mean()
                row[f"excess_next_day_max_{side}_pct"] = tt["mean"]
                row[f"excess_sd_next_day_max_{side}_pct"] = tt["sd"]
                row[f"excess_t_next_day_max_{side}_pct"] = tt["tstat"]
                row[f"excess_p_next_day_max_{side}_pct"] = tt["pvalue"]
                row[f"global_future_max_{side}_pct"] = daily_df[abn_col].mean()

            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["pattern", "q_label"]).reset_index(drop=True)


def run_daily_imbalance_effect_module_multiq(
    df,
    price_col="vwap",
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    daily_df = build_daily_imbalance_panel_for_excursion(df=df, price_col=price_col)
    daily_df = add_daily_imbalance_flags_multiq(
        daily_df=daily_df,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )
    daily_df = build_daily_imbalance_event_baselines_multiq(
        daily_df=daily_df,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )
    daily_summary = summarize_daily_imbalance_multiq(
        daily_df=daily_df,
        abnormal_qs=abnormal_qs,
    )
    return daily_df, daily_summary


def build_weekly_imbalance_panel_for_excursion(df, price_col="vwap"):
    weekly = build_weekly_panel_for_excursion(df=df, price_col=price_col)
    weekly["imbalance_diff"] = weekly["trade_volume_above_mid"] - weekly["trade_volume_below_mid"]
    weekly["imbalance_abs"] = weekly["imbalance_diff"].abs()
    return weekly


def add_weekly_imbalance_flags_multiq(
    weekly_df,
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = weekly_df.copy()
    min_hist = max(5, baseline_window // 4)
    s = pd.to_numeric(x["imbalance_diff"], errors="coerce")
    abs_base_mean = s.abs().shift(1).rolling(baseline_window, min_periods=min_hist).mean()

    new_cols = {}
    for side in ["imbalance_above_mid", "imbalance_below_mid"]:
        new_cols[f"baseline_mean_{side}"] = abs_base_mean
        new_cols[f"rel_{side}"] = s.abs() / abs_base_mean

    for q in abnormal_qs:
        tag = _qtag(q)
        upper_q = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(q)
        lower_q = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(1 - q)
        new_cols[f"baseline_q_imbalance_above_mid_{tag}"] = upper_q
        new_cols[f"baseline_q_imbalance_below_mid_{tag}"] = lower_q
        new_cols[f"abn_imbalance_above_mid_{tag}"] = (s >= upper_q).astype(int)
        new_cols[f"abn_imbalance_below_mid_{tag}"] = (s <= lower_q).astype(int)

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_weekly_imbalance_event_baselines_multiq(
    weekly_df,
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = weekly_df.copy()

    init_cols = {}
    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)
            init_cols[f"baseline_next_week_max_up_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_max_down_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_max_abs_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_up_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_down_{pattern}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_abs_{pattern}_{tag}_pct"] = np.nan
    x = pd.concat([x, pd.DataFrame(init_cols, index=x.index)], axis=1).copy()

    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)

            any_abn_col = (
                x[f"abn_imbalance_above_mid_{tag}"].fillna(0).astype(int)
                | x[f"abn_imbalance_below_mid_{tag}"].fillna(0).astype(int)
            )

            for i in x.index:
                start = max(0, i - baseline_window)
                hist = x.iloc[start:i].copy()
                if len(hist) == 0:
                    continue

                hist = hist[any_abn_col.iloc[start:i].to_numpy() == 0].copy()
                if len(hist) == 0:
                    continue

                x.at[i, f"baseline_next_week_max_up_{pattern}_{tag}_pct"] = hist["next_week_max_up_pct"].mean()
                x.at[i, f"baseline_next_week_max_down_{pattern}_{tag}_pct"] = hist["next_week_max_down_pct"].mean()
                x.at[i, f"baseline_next_week_max_abs_{pattern}_{tag}_pct"] = hist["next_week_max_abs_pct"].mean()
                x.at[i, f"baseline_next_week_sd_up_{pattern}_{tag}_pct"] = hist["next_week_max_up_pct"].std(ddof=1)
                x.at[i, f"baseline_next_week_sd_down_{pattern}_{tag}_pct"] = hist["next_week_max_down_pct"].std(ddof=1)
                x.at[i, f"baseline_next_week_sd_abs_{pattern}_{tag}_pct"] = hist["next_week_max_abs_pct"].std(ddof=1)

    return x


def summarize_weekly_imbalance_multiq(
    weekly_df,
    abnormal_qs=(0.90, 0.95, 0.99),
):
    rows = []

    for pattern in ["imbalance_above_mid", "imbalance_below_mid"]:
        for q in abnormal_qs:
            tag = _qtag(q)
            abn = weekly_df[weekly_df[f"abn_{pattern}_{tag}"] == 1].copy()
            if len(abn) == 0:
                continue

            row = {
                "pattern": pattern,
                "q_label": int(round(q * 100)),
                "n_abnormal_weeks": len(abn),
                "avg_rel_to_baseline": abn[f"rel_{pattern}"].mean(),
                "current_window_max_up_pct": abn["current_window_max_up_pct"].mean(),
                "current_window_max_down_pct": abn["current_window_max_down_pct"].mean(),
                "current_window_max_abs_pct": abn["current_window_max_abs_pct"].mean(),
            }

            for side in ["up", "down", "abs"]:
                abn_col = f"next_week_max_{side}_pct"
                base_col = f"baseline_next_week_max_{side}_{pattern}_{tag}_pct"
                diff = abn[abn_col] - abn[base_col]
                tt = _safe_ttest_zero(diff)

                row[f"abn_next_week_max_{side}_pct"] = abn[abn_col].mean()
                row[f"baseline_next_week_max_{side}_pct"] = abn[base_col].mean()
                row[f"excess_next_week_max_{side}_pct"] = tt["mean"]
                row[f"excess_sd_next_week_max_{side}_pct"] = tt["sd"]
                row[f"excess_t_next_week_max_{side}_pct"] = tt["tstat"]
                row[f"excess_p_next_week_max_{side}_pct"] = tt["pvalue"]
                row[f"global_future_max_{side}_pct"] = weekly_df[abn_col].mean()

            rows.append(row)

    if len(rows) == 0:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["pattern", "q_label"]).reset_index(drop=True)


def run_weekly_imbalance_effect_module_multiq(
    df,
    price_col="vwap",
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    weekly_df = build_weekly_imbalance_panel_for_excursion(df=df, price_col=price_col)
    weekly_df = add_weekly_imbalance_flags_multiq(
        weekly_df=weekly_df,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )
    weekly_df = build_weekly_imbalance_event_baselines_multiq(
        weekly_df=weekly_df,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )
    weekly_summary = summarize_weekly_imbalance_multiq(
        weekly_df=weekly_df,
        abnormal_qs=abnormal_qs,
    )
    return weekly_df, weekly_summary


def plot_trade_volume_mid_distributions(
    df,
    bins=120,
    max_points=300000,
    clip_q=0.995,
    use_log1p=True,
):
    x = _copy_and_prepare(df)

    cols = [
        "trade_volume_at_mid",
        "trade_volume_below_mid",
        "trade_volume_above_mid",
    ]

    for col in cols:
        if col not in x.columns:
            raise ValueError(f"Missing column: {col}")

    for col in cols:
        s = _sample_series(x[col], max_points=max_points, clip_q=clip_q)
        if use_log1p:
            s = np.log1p(s)
            xlabel = f"log(1 + {col})"
        else:
            xlabel = col

        plt.figure(figsize=(10, 5))
        plt.hist(s, bins=bins)
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"Distribution of {col}")
        plt.tight_layout()
        plt.show()


def plot_vwap_diff_distribution(
    df,
    bins=120,
    max_points=300000,
    clip_q=0.995,
):
    x = _copy_and_prepare(df)

    if "vwap_diff" not in x.columns:
        raise ValueError("Need column vwap to compute vwap_diff.")

    s = _sample_series(x["vwap_diff"], max_points=max_points, clip_q=clip_q)

    plt.figure(figsize=(10, 5))
    plt.hist(s, bins=bins)
    plt.xlabel("Minute-to-minute VWAP difference")
    plt.ylabel("Frequency")
    plt.title("Distribution of VWAP Differences")
    plt.tight_layout()
    plt.show()


def plot_spread_distribution(
    df,
    bins=120,
    max_points=300000,
    clip_q=0.995,
):
    x = _copy_and_prepare(df)

    if "spread" not in x.columns:
        raise ValueError("Need bid and ask columns to compute spread.")

    s = _sample_series(x["spread"], max_points=max_points, clip_q=clip_q)
    s = s[s > 0]

    plt.figure(figsize=(10, 5))
    if len(s) > 0:
        plt.hist(s, bins=np.logspace(np.log10(s.min()), np.log10(s.max()), bins))
        plt.xscale("log")
    else:
        plt.text(0.5, 0.5, "No positive bid-ask spreads available", ha="center", va="center")
        plt.xticks([])
    plt.xlabel("Ask - Bid Spread")
    plt.ylabel("Frequency")
    plt.title("Distribution of Bid-Ask Spread (Log Scale)")
    plt.tight_layout()
    plt.show()


def plot_monthly_average_spread(df):
    x = _copy_and_prepare(df)

    if "spread" not in x.columns:
        raise ValueError("Need bid and ask columns to compute spread.")

    monthly = x.groupby("month", as_index=False)["spread"].mean()

    plt.figure(figsize=(12, 5))
    plt.plot(monthly["month"], monthly["spread"])
    plt.xlabel("Month")
    plt.ylabel("Average Spread")
    plt.title("Monthly Average Bid-Ask Spread")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_monthly_average_trade_volume_components(df):
    x = _copy_and_prepare(df)

    cols = [
        "trade_volume_at_mid",
        "trade_volume_below_mid",
        "trade_volume_above_mid",
    ]
    for col in cols:
        if col not in x.columns:
            raise ValueError(f"Missing column: {col}")

    monthly = (
        x.groupby("month", as_index=False)[cols]
        .mean()
        .sort_values("month")
    )

    for col in cols:
        plt.figure(figsize=(12, 5))
        plt.plot(monthly["month"], monthly[col])
        plt.xlabel("Month")
        plt.ylabel(f"Average {col}")
        plt.title(f"Monthly Average of {col}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()


def plot_all_microstructure_summary(df):
    plot_trade_volume_mid_distributions(df)
    plot_vwap_diff_distribution(df)
    plot_spread_distribution(df)
    plot_monthly_average_spread(df)
    plot_monthly_average_trade_volume_components(df)

def prepare_extreme_diff_regimes(
    df,
    price_col="vwap",
    top_pct=0.1,
    bottom_pct=0.1,
):
    x = df.copy()
    x["minute_dt"] = pd.to_datetime(x["minute_dt"])
    x = x.sort_values("minute_dt").reset_index(drop=True)
    x["date"] = x["minute_dt"].dt.date

    daily = (
        x.groupby("date", as_index=False)[price_col]
        .agg(day_min="min", day_max="max")
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily["next_day_min"] = daily["day_min"].shift(-1)
    daily["next_day_max"] = daily["day_max"].shift(-1)

    daily["diff_max_max"] = (daily["next_day_max"] - daily["day_max"]).abs()
    daily["diff_max_min"] = (daily["next_day_max"] - daily["day_min"]).abs()
    daily["diff_min_max"] = (daily["next_day_min"] - daily["day_max"]).abs()
    daily["diff_min_min"] = (daily["next_day_min"] - daily["day_min"]).abs()

    daily["extreme_diff_vol"] = daily[
        ["diff_max_max", "diff_max_min", "diff_min_max", "diff_min_min"]
    ].max(axis=1)

    q_low = daily["extreme_diff_vol"].quantile(bottom_pct)
    q_high = daily["extreme_diff_vol"].quantile(1 - top_pct)

    daily["regime"] = "normal"
    daily.loc[daily["extreme_diff_vol"] <= q_low, "regime"] = "low"
    daily.loc[daily["extreme_diff_vol"] >= q_high, "regime"] = "high"

    x = x.merge(
        daily[["date", "extreme_diff_vol", "regime"]],
        on="date",
        how="left",
    )

    return x, daily



    summary = (
        df_regime.groupby("regime")[
            ["trade_volume_at_mid", "trade_volume_below_mid", "trade_volume_above_mid"]
        ]
        .mean()
        .reindex(["low", "normal", "high"])
    )

    for col in ["trade_volume_at_mid", "trade_volume_below_mid", "trade_volume_above_mid"]:
        plt.figure(figsize=(8, 5))
        plt.bar(summary.index, summary[col])
        plt.xlabel("Regime")
        plt.ylabel(f"Average {col}")
        plt.title(f"Average {col} by Regime")
        plt.tight_layout()
        plt.show()

# ===== Cell 2 =====
def prepare_intraday_abnormal_features_multiq(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    seq_windows=(5, 10, 20, 30),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=390 * 20,
):
    x = df.copy()
    x["minute_dt"] = pd.to_datetime(x["minute_dt"])
    x = x.sort_values("minute_dt").reset_index(drop=True)

    if "date" not in x.columns:
        x["date"] = x["minute_dt"].dt.date

    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    x["price"] = x[price_col]

    if {"ask", "bid"}.issubset(x.columns):
        x["spread"] = pd.to_numeric(x["ask"], errors="coerce") - pd.to_numeric(x["bid"], errors="coerce")
    else:
        x["spread"] = np.nan

    min_hist = max(200, baseline_window // 5)
    new_cols = {}

    for col in vol_cols:
        s = pd.to_numeric(x[col], errors="coerce")

        for w in seq_windows:
            cum_col = f"cumseq_{col}_{w}m"
            base_mean_col = f"baseline_mean_{col}_{w}m"
            rel_col = f"rel_to_mean_{col}_{w}m"

            cum_s = s.rolling(w, min_periods=w).sum()
            base_mean_s = cum_s.rolling(baseline_window, min_periods=min_hist).mean()
            rel_s = cum_s / base_mean_s

            new_cols[cum_col] = cum_s
            new_cols[base_mean_col] = base_mean_s
            new_cols[rel_col] = rel_s

            for q in abnormal_qs:
                tag = _qtag(q)
                q_col = f"baseline_q_{col}_{w}m_{tag}"
                abn_col = f"abn_{col}_{w}m_{tag}"

                q_s = cum_s.rolling(baseline_window, min_periods=min_hist).quantile(q)
                abn_s = (cum_s >= q_s).astype(int)

                new_cols[q_col] = q_s
                new_cols[abn_col] = abn_s

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_intraday_sequence_table_multiq(
    df_feat,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    seq_windows=(5, 10, 20, 30),
    abnormal_qs=(0.90, 0.95, 0.99),
    min_run=2,
):
    out = []

    for col in vol_cols:
        for w in seq_windows:
            cum_col = f"cumseq_{col}_{w}m"
            base_mean_col = f"baseline_mean_{col}_{w}m"
            rel_col = f"rel_to_mean_{col}_{w}m"

            for q in abnormal_qs:
                tag = _qtag(q)
                flag_col = f"abn_{col}_{w}m_{tag}"

                z = df_feat.copy()
                z["_flag"] = z[flag_col].fillna(0).astype(int)
                z["_grp"] = (z["_flag"] != z["_flag"].shift()).cumsum()

                seq = (
                    z.groupby("_grp", as_index=False)
                    .agg(
                        flag_value=("_flag", "first"),
                        start_idx=("minute_dt", lambda s: s.index[0]),
                        end_idx=("minute_dt", lambda s: s.index[-1]),
                        start_dt=("minute_dt", "first"),
                        end_dt=("minute_dt", "last"),
                        start_date=("date", "first"),
                        run_len_mins=("minute_dt", "size"),
                        avg_spread_dollar=("spread", "mean"),
                        avg_seq_cumvol=(cum_col, "mean"),
                        avg_baseline_cumvol=(base_mean_col, "mean"),
                        avg_rel_to_baseline=(rel_col, "mean"),
                        seq_start_price=("price", "first"),
                        seq_high_price=("price", "max"),
                        seq_low_price=("price", "min"),
                    )
                )

                seq = seq[(seq["flag_value"] == 1) & (seq["run_len_mins"] >= min_run)].copy()
                if len(seq) == 0:
                    continue

                with np.errstate(divide="ignore", invalid="ignore"):
                    seq["current_window_max_up_pct"] = 100.0 * (seq["seq_high_price"] - seq["seq_start_price"]) / seq["seq_start_price"]
                    seq["current_window_max_down_pct"] = 100.0 * (seq["seq_low_price"] - seq["seq_start_price"]) / seq["seq_start_price"]
                seq["current_window_max_abs_pct"] = seq[["current_window_max_up_pct", "current_window_max_down_pct"]].abs().max(axis=1)

                seq["pattern"] = col
                seq["seq_window_mins"] = w
                seq["q_label"] = int(round(q * 100))
                out.append(seq)

    return pd.concat(out, ignore_index=True) if len(out) > 0 else pd.DataFrame()


def attach_intraday_future_excursions_multiq(
    seq_df,
    df_feat,
    horizons=DEFAULT_INTRADAY_HORIZONS,
    excursion_cache=None,
):
    if len(seq_df) == 0:
        return seq_df.copy()

    out = seq_df.copy()
    price_arr = df_feat["price"].to_numpy(dtype="float64")
    cache = excursion_cache if excursion_cache is not None else _compute_future_excursion_arrays(price_arr, horizons)
    end_indices = out["end_idx"].astype(int).to_numpy()

    for h in horizons:
        out[f"future_max_up_{h}m_pct"] = cache[h]["up"][end_indices]
        out[f"future_max_down_{h}m_pct"] = cache[h]["down"][end_indices]
        out[f"future_max_abs_{h}m_pct"] = cache[h]["abs"][end_indices]

    return out


def build_intraday_event_baselines_multiq(
    seq_outcome_df,
    df_feat,
    horizons=DEFAULT_INTRADAY_HORIZONS,
    baseline_window=20,
    excursion_cache=None,
    progress_prefix=None,
):
    if len(seq_outcome_df) == 0:
        return seq_outcome_df.copy()

    out = seq_outcome_df.copy()
    price_arr = df_feat["price"].to_numpy(dtype="float64")
    cache = excursion_cache if excursion_cache is not None else _compute_future_excursion_arrays(price_arr, horizons)

    init_cols = {}
    for h in horizons:
        init_cols[f"baseline_max_up_{h}m_pct"] = np.nan
        init_cols[f"baseline_max_down_{h}m_pct"] = np.nan
        init_cols[f"baseline_max_abs_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_up_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_down_{h}m_pct"] = np.nan
        init_cols[f"baseline_sd_abs_{h}m_pct"] = np.nan
    out = pd.concat([out, pd.DataFrame(init_cols, index=out.index)], axis=1).copy()

    combos = list(out[["pattern", "seq_window_mins", "q_label"]].drop_duplicates().itertuples(index=False, name=None))
    combo_total = len(combos)

    for combo_idx, (pattern, seq_window, q_label) in enumerate(combos, start=1):
        if progress_prefix:
            _render_progress(combo_idx - 1, combo_total, prefix=progress_prefix)

        abn_col = f"abn_{pattern}_{int(seq_window)}m_q{int(q_label)}"
        normal_mask = df_feat[abn_col].fillna(0).to_numpy() == 0
        combo_mask = (
            (out["pattern"] == pattern)
            & (out["seq_window_mins"] == seq_window)
            & (out["q_label"] == q_label)
        )
        event_indices = out.loc[combo_mask, "end_idx"].astype(int).to_numpy()
        target_rows = out.index[combo_mask]

        for h in horizons:
            mean_up, std_up = _rolling_baseline_stats(cache[h]["up"], normal_mask, baseline_window)
            mean_down, std_down = _rolling_baseline_stats(cache[h]["down"], normal_mask, baseline_window)
            mean_abs, std_abs = _rolling_baseline_stats(cache[h]["abs"], normal_mask, baseline_window)

            out.loc[target_rows, f"baseline_max_up_{h}m_pct"] = mean_up[event_indices]
            out.loc[target_rows, f"baseline_max_down_{h}m_pct"] = mean_down[event_indices]
            out.loc[target_rows, f"baseline_max_abs_{h}m_pct"] = mean_abs[event_indices]
            out.loc[target_rows, f"baseline_sd_up_{h}m_pct"] = std_up[event_indices]
            out.loc[target_rows, f"baseline_sd_down_{h}m_pct"] = std_down[event_indices]
            out.loc[target_rows, f"baseline_sd_abs_{h}m_pct"] = std_abs[event_indices]

    if progress_prefix:
        _render_progress(combo_total, combo_total, prefix=progress_prefix, final=True)

    return out


def summarize_intraday_multiq(
    seq_outcome_df,
    horizons=DEFAULT_INTRADAY_HORIZONS,
    global_future_means=None,
):
    rows = []

    if len(seq_outcome_df) == 0:
        return pd.DataFrame()

    for keys, g in seq_outcome_df.groupby(["pattern", "seq_window_mins", "q_label"]):
        pattern, seq_window, q_label = keys

        row = {
            "pattern": pattern,
            "seq_window_mins": seq_window,
            "q_label": q_label,
            "n_sequences": len(g),
            "avg_run_len_mins": g["run_len_mins"].mean(),
            "avg_seq_cumvol": g["avg_seq_cumvol"].mean(),
            "avg_baseline_cumvol": g["avg_baseline_cumvol"].mean(),
            "avg_rel_to_baseline": g["avg_rel_to_baseline"].mean(),
            "current_window_max_up_pct": g["current_window_max_up_pct"].mean(),
            "current_window_max_down_pct": g["current_window_max_down_pct"].mean(),
            "current_window_max_abs_pct": g["current_window_max_abs_pct"].mean(),
        }

        for h in horizons:
            for side in ["up", "down", "abs"]:
                abn_col = f"future_max_{side}_{h}m_pct"
                base_col = f"baseline_max_{side}_{h}m_pct"
                diff = g[abn_col] - g[base_col]
                tt = _safe_ttest_zero(diff)

                row[f"abn_max_{side}_{h}m_pct"] = g[abn_col].mean()
                row[f"baseline_max_{side}_{h}m_pct"] = g[base_col].mean()
                row[f"excess_max_{side}_{h}m_pct"] = tt["mean"]
                row[f"excess_sd_{side}_{h}m_pct"] = tt["sd"]
                row[f"excess_t_{side}_{h}m_pct"] = tt["tstat"]
                row[f"excess_p_{side}_{h}m_pct"] = tt["pvalue"]
                row[f"global_future_max_{side}_{h}m_pct"] = (
                    global_future_means.get((side, h), np.nan) if global_future_means is not None else np.nan
                )

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["pattern", "q_label", "seq_window_mins"]).reset_index(drop=True)


def plot_intraday_multiq_to_pdf(summary_df, pdf, ticker, horizons=DEFAULT_INTRADAY_HORIZONS, annotate_lower_than_baseline=False):
    if len(summary_df) == 0:
        return

    for q_label in sorted(summary_df["q_label"].unique()):
        for pattern in ["trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"]:
            sub = summary_df[(summary_df["q_label"] == q_label) & (summary_df["pattern"] == pattern)].sort_values("seq_window_mins")
            if len(sub) == 0:
                continue

            fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
            side_specs = [
                ("up", "Max Up Drift (%)"),
                ("down", "Max Down Drift (%)"),
                ("abs", "Max Absolute Drift (%)"),
            ]
            for ax, (side, ylabel) in zip(axes, side_specs):
                for _, row in sub.iterrows():
                    y1 = [row[f"abn_max_{side}_{h}m_pct"] for h in horizons]
                    y2 = [row[f"baseline_max_{side}_{h}m_pct"] for h in horizons]
                    y3 = [row.get(f"global_future_max_{side}_{h}m_pct", np.nan) for h in horizons]
                    y4 = [row.get(f"current_window_max_{side}_pct", np.nan)] * len(horizons)
                    ax.plot(horizons, y1, marker="o", label=f"post-abnormal future drift {int(row['seq_window_mins'])}m")
                    ax.plot(horizons, y2, marker="o", linestyle="--", label=f"matched baseline future drift {int(row['seq_window_mins'])}m")
                    ax.plot(horizons, y3, linestyle=":", linewidth=1.2, color="black", alpha=0.8, label=f"global average future drift {int(row['seq_window_mins'])}m")
                    ax.plot(horizons, y4, linestyle="-.", linewidth=1.0, color="gray", alpha=0.9, label=f"realized drift inside abnormal window {int(row['seq_window_mins'])}m")
                    if annotate_lower_than_baseline:
                        y1_arr = np.array(y1, dtype="float64")
                        y2_arr = np.array(y2, dtype="float64")
                        mask = np.isfinite(y1_arr) & np.isfinite(y2_arr) & (y1_arr < y2_arr)
                        if mask.any():
                            ax.scatter(
                                np.array(horizons, dtype="float64")[mask],
                                y1_arr[mask],
                                s=58,
                                marker="o",
                                facecolors="black",
                                edgecolors="black",
                                linewidths=0.8,
                                zorder=4,
                            )
                ax.set_ylabel(ylabel)
                ax.set_xlabel("Future horizon (minutes)")
                ax.set_title(f"{ticker} - Intraday q{q_label} - {pattern} - {side.upper()}")
                if side == "down":
                    ax.axhline(0, linewidth=1, color="black")
            axes[0].legend()
            _save_current_figure_to_pdf(pdf)


def run_intraday_effect_module_multiq(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    seq_windows=(5, 10, 20, 30),
    abnormal_qs=(0.90, 0.95, 0.99),
    horizons=DEFAULT_INTRADAY_HORIZONS,
    baseline_window=20,
    min_run=2,
    progress_prefix=None,
):
    df_feat = prepare_intraday_abnormal_features_multiq(
        df=df,
        price_col=price_col,
        vol_cols=vol_cols,
        seq_windows=seq_windows,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    seq_df = build_intraday_sequence_table_multiq(
        df_feat=df_feat,
        vol_cols=vol_cols,
        seq_windows=seq_windows,
        abnormal_qs=abnormal_qs,
        min_run=min_run,
    )

    excursion_cache = _compute_future_excursion_arrays(df_feat["price"].to_numpy(dtype="float64"), horizons)

    seq_outcome_df = attach_intraday_future_excursions_multiq(
        seq_df=seq_df,
        df_feat=df_feat,
        horizons=horizons,
        excursion_cache=excursion_cache,
    )

    seq_outcome_df = build_intraday_event_baselines_multiq(
        seq_outcome_df=seq_outcome_df,
        df_feat=df_feat,
        horizons=horizons,
        baseline_window=baseline_window,
        excursion_cache=excursion_cache,
        progress_prefix=progress_prefix,
    )

    global_future_means = {
        (side, h): pd.Series(excursion_cache[h][side]).dropna().mean()
        for h in horizons
        for side in ["up", "down", "abs"]
    }
    summary_df = summarize_intraday_multiq(
        seq_outcome_df=seq_outcome_df,
        horizons=horizons,
        global_future_means=global_future_means,
    )

    return df_feat, seq_df, seq_outcome_df, summary_df

# ===== Cell 3 =====
def build_daily_panel_for_excursion(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
):
    x = df.copy()
    x["minute_dt"] = pd.to_datetime(x["minute_dt"])
    x = x.sort_values("minute_dt").reset_index(drop=True)
    x["date"] = x["minute_dt"].dt.date

    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    for c in vol_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    daily = (
        x.groupby("date", as_index=False)
        .agg(
            close_price=(price_col, "last"),
            **{c: (c, "sum") for c in vol_cols}
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    day_hilo = (
        x.groupby("date", as_index=False)
        .agg(day_high=(price_col, "max"), day_low=(price_col, "min"))
        .sort_values("date")
        .reset_index(drop=True)
    )

    daily = daily.merge(day_hilo, on="date", how="left")
    daily["next_day_high"] = daily["day_high"].shift(-1)
    daily["next_day_low"] = daily["day_low"].shift(-1)

    daily["next_day_max_up_pct"] = 100.0 * (daily["next_day_high"] - daily["close_price"]) / daily["close_price"]
    daily["next_day_max_down_pct"] = 100.0 * (daily["next_day_low"] - daily["close_price"]) / daily["close_price"]
    daily["next_day_max_abs_pct"] = daily[["next_day_max_up_pct", "next_day_max_down_pct"]].abs().max(axis=1)
    prev_close = daily["close_price"].shift(1)
    daily["current_window_max_up_pct"] = 100.0 * (daily["day_high"] - prev_close) / prev_close
    daily["current_window_max_down_pct"] = 100.0 * (daily["day_low"] - prev_close) / prev_close
    daily["current_window_max_abs_pct"] = daily[["current_window_max_up_pct", "current_window_max_down_pct"]].abs().max(axis=1)

    return daily


def add_daily_abnormal_flags_multiq(
    daily_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = daily_df.copy()
    min_hist = max(5, baseline_window // 4)
    new_cols = {}

    for col in vol_cols:
        s = pd.to_numeric(x[col], errors="coerce")
        base_mean = s.shift(1).rolling(baseline_window, min_periods=min_hist).mean()
        new_cols[f"baseline_mean_{col}"] = base_mean
        new_cols[f"rel_{col}"] = s / base_mean

        for q in abnormal_qs:
            tag = _qtag(q)
            q_s = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(q)
            new_cols[f"baseline_q_{col}_{tag}"] = q_s
            new_cols[f"abn_{col}_{tag}"] = (s >= q_s).astype(int)

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_daily_event_baselines_multiq(
    daily_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = daily_df.copy()

    init_cols = {}
    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)
            init_cols[f"baseline_next_day_max_up_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_max_down_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_max_abs_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_up_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_down_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_day_sd_abs_{col}_{tag}_pct"] = np.nan
    x = pd.concat([x, pd.DataFrame(init_cols, index=x.index)], axis=1).copy()

    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)

            abn_col = f"abn_{col}_{tag}"

            for i in x.index:
                start = max(0, i - baseline_window)
                hist = x.iloc[start:i].copy()
                if len(hist) == 0:
                    continue

                hist = hist[hist[abn_col] == 0].copy()
                if len(hist) == 0:
                    continue

                x.at[i, f"baseline_next_day_max_up_{col}_{tag}_pct"] = hist["next_day_max_up_pct"].mean()
                x.at[i, f"baseline_next_day_max_down_{col}_{tag}_pct"] = hist["next_day_max_down_pct"].mean()
                x.at[i, f"baseline_next_day_max_abs_{col}_{tag}_pct"] = hist["next_day_max_abs_pct"].mean()

                x.at[i, f"baseline_next_day_sd_up_{col}_{tag}_pct"] = hist["next_day_max_up_pct"].std(ddof=1)
                x.at[i, f"baseline_next_day_sd_down_{col}_{tag}_pct"] = hist["next_day_max_down_pct"].std(ddof=1)
                x.at[i, f"baseline_next_day_sd_abs_{col}_{tag}_pct"] = hist["next_day_max_abs_pct"].std(ddof=1)

    return x


def summarize_daily_multiq(
    daily_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
):
    rows = []

    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)
            abn = daily_df[daily_df[f"abn_{col}_{tag}"] == 1].copy()
            if len(abn) == 0:
                continue

            row = {
                "pattern": col,
                "q_label": int(round(q * 100)),
                "n_abnormal_days": len(abn),
                "avg_rel_to_baseline": abn[f"rel_{col}"].mean(),
                "current_window_max_up_pct": abn["current_window_max_up_pct"].mean(),
                "current_window_max_down_pct": abn["current_window_max_down_pct"].mean(),
                "current_window_max_abs_pct": abn["current_window_max_abs_pct"].mean(),
            }

            for side in ["up", "down", "abs"]:
                abn_col = f"next_day_max_{side}_pct"
                base_col = f"baseline_next_day_max_{side}_{col}_{tag}_pct"
                diff = abn[abn_col] - abn[base_col]
                tt = _safe_ttest_zero(diff)

                row[f"abn_next_day_max_{side}_pct"] = abn[abn_col].mean()
                row[f"baseline_next_day_max_{side}_pct"] = abn[base_col].mean()
                row[f"excess_next_day_max_{side}_pct"] = tt["mean"]
                row[f"excess_sd_next_day_max_{side}_pct"] = tt["sd"]
                row[f"excess_t_next_day_max_{side}_pct"] = tt["tstat"]
                row[f"excess_p_next_day_max_{side}_pct"] = tt["pvalue"]
                row[f"global_future_max_{side}_pct"] = daily_df[abn_col].mean()

            rows.append(row)

    return pd.DataFrame(rows).sort_values(["pattern", "q_label"]).reset_index(drop=True)


def plot_daily_multiq_to_pdf(daily_summary, pdf, ticker, annotate_lower_than_baseline=False):
    if len(daily_summary) == 0:
        return

    for q_label in sorted(daily_summary["q_label"].unique()):
        sub = daily_summary[daily_summary["q_label"] == q_label].copy()

        side_specs = [
            ("up", "Max Up Drift (%)"),
            ("down", "Max Down Drift (%)"),
            ("abs", "Max Absolute Drift (%)"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
        for ax, (side, ylabel) in zip(axes, side_specs):
            x = np.arange(len(sub["pattern"]))
            abn_vals = sub[f"abn_next_day_max_{side}_pct"].to_numpy(dtype="float64")
            base_vals = sub[f"baseline_next_day_max_{side}_pct"].to_numpy(dtype="float64")
            global_vals = sub[f"global_future_max_{side}_pct"].to_numpy(dtype="float64")
            current_vals = sub[f"current_window_max_{side}_pct"].to_numpy(dtype="float64")
            width = 0.2
            bars = ax.bar(x - 1.5 * width, abn_vals, width=width, facecolor="white", edgecolor="black", linewidth=1.0, label="post-abnormal future drift")
            if annotate_lower_than_baseline:
                lower_mask = np.isfinite(abn_vals) & np.isfinite(base_vals) & (abn_vals < base_vals)
                for bar, is_lower in zip(bars, lower_mask):
                    if is_lower:
                        bar.set_hatch("xx")
                        bar.set_facecolor("#D9D9D9")
            ax.bar(x - 0.5 * width, base_vals, width=width, facecolor="white", edgecolor="black", linewidth=1.0, hatch="//", label="matched baseline future drift")
            ax.bar(x + 0.5 * width, global_vals, width=width, facecolor="#F0F0F0", edgecolor="black", linewidth=1.0, hatch="..", label="global average future drift")
            ax.bar(x + 1.5 * width, current_vals, width=width, facecolor="#BFBFBF", edgecolor="black", linewidth=1.0, label="realized drift inside abnormal window")
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Pattern")
            ax.set_xticks(x)
            ax.set_xticklabels(sub["pattern"], rotation=15)
            ax.set_title(f"{ticker} - Daily q{q_label} - {side.upper()}")
            ax.grid(axis="y", alpha=0.25)
            if side == "down":
                ax.axhline(0, linewidth=1, color="black")
        axes[0].legend()
        _save_current_figure_to_pdf(pdf)


def plot_daily_multiq_lines_to_pdf(daily_summary, pdf, ticker):
    if len(daily_summary) == 0:
        return

    for pattern in daily_summary["pattern"].drop_duplicates():
        sub = daily_summary[daily_summary["pattern"] == pattern].sort_values("q_label")
        q_vals = sub["q_label"].to_numpy(dtype="float64")
        side_specs = [
            ("up", "Max Up Drift (%)"),
            ("down", "Max Down Drift (%)"),
            ("abs", "Max Absolute Drift (%)"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for ax, (side, ylabel) in zip(axes, side_specs):
            ax.plot(q_vals, sub[f"abn_next_day_max_{side}_pct"], marker="o", color="black", label="post-abnormal future drift")
            ax.plot(q_vals, sub[f"baseline_next_day_max_{side}_pct"], marker="o", linestyle="--", color="black", label="matched baseline future drift")
            ax.plot(q_vals, sub[f"global_future_max_{side}_pct"], marker="o", linestyle=":", color="black", alpha=0.8, label="global average future drift")
            ax.plot(q_vals, sub[f"current_window_max_{side}_pct"], marker="o", linestyle="-.", color="gray", label="realized drift inside abnormal window")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ticker} - Daily Line View - {pattern} - {side.upper()}")
            ax.grid(alpha=0.25)
            if side == "down":
                ax.axhline(0, linewidth=1, color="black")
        axes[-1].set_xlabel("Quantile")
        axes[-1].set_xticks(q_vals)
        axes[-1].set_xticklabels([f"q{int(v)}" for v in q_vals])
        axes[0].legend()
        _save_current_figure_to_pdf(pdf)


def run_daily_effect_module_multiq(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    daily_df = build_daily_panel_for_excursion(
        df=df,
        price_col=price_col,
        vol_cols=vol_cols,
    )

    daily_df = add_daily_abnormal_flags_multiq(
        daily_df=daily_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    daily_df = build_daily_event_baselines_multiq(
        daily_df=daily_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    daily_summary = summarize_daily_multiq(
        daily_df=daily_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
    )

    return daily_df, daily_summary

# ===== Cell 4 =====
def build_weekly_panel_for_excursion(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
):
    x = df.copy()
    x["minute_dt"] = pd.to_datetime(x["minute_dt"])
    x = x.sort_values("minute_dt").reset_index(drop=True)
    x["week"] = x["minute_dt"].dt.to_period("W").astype(str)

    x[price_col] = pd.to_numeric(x[price_col], errors="coerce")
    for c in vol_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    weekly = (
        x.groupby("week", as_index=False)
        .agg(
            close_price=(price_col, "last"),
            **{c: (c, "sum") for c in vol_cols}
        )
        .sort_values("week")
        .reset_index(drop=True)
    )

    week_hilo = (
        x.groupby("week", as_index=False)
        .agg(week_high=(price_col, "max"), week_low=(price_col, "min"))
        .sort_values("week")
        .reset_index(drop=True)
    )

    weekly = weekly.merge(week_hilo, on="week", how="left")
    weekly["next_week_high"] = weekly["week_high"].shift(-1)
    weekly["next_week_low"] = weekly["week_low"].shift(-1)

    weekly["next_week_max_up_pct"] = 100.0 * (weekly["next_week_high"] - weekly["close_price"]) / weekly["close_price"]
    weekly["next_week_max_down_pct"] = 100.0 * (weekly["next_week_low"] - weekly["close_price"]) / weekly["close_price"]
    weekly["next_week_max_abs_pct"] = weekly[["next_week_max_up_pct", "next_week_max_down_pct"]].abs().max(axis=1)
    prev_close = weekly["close_price"].shift(1)
    weekly["current_window_max_up_pct"] = 100.0 * (weekly["week_high"] - prev_close) / prev_close
    weekly["current_window_max_down_pct"] = 100.0 * (weekly["week_low"] - prev_close) / prev_close
    weekly["current_window_max_abs_pct"] = weekly[["current_window_max_up_pct", "current_window_max_down_pct"]].abs().max(axis=1)

    return weekly


def add_weekly_abnormal_flags_multiq(
    weekly_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = weekly_df.copy()
    min_hist = max(5, baseline_window // 4)
    new_cols = {}

    for col in vol_cols:
        s = pd.to_numeric(x[col], errors="coerce")
        base_mean = s.shift(1).rolling(baseline_window, min_periods=min_hist).mean()
        new_cols[f"baseline_mean_{col}"] = base_mean
        new_cols[f"rel_{col}"] = s / base_mean

        for q in abnormal_qs:
            tag = _qtag(q)
            q_s = s.shift(1).rolling(baseline_window, min_periods=min_hist).quantile(q)
            new_cols[f"baseline_q_{col}_{tag}"] = q_s
            new_cols[f"abn_{col}_{tag}"] = (s >= q_s).astype(int)

    x = pd.concat([x, pd.DataFrame(new_cols, index=x.index)], axis=1).copy()
    return x


def build_weekly_event_baselines_multiq(
    weekly_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    x = weekly_df.copy()

    init_cols = {}
    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)
            init_cols[f"baseline_next_week_max_up_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_max_down_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_max_abs_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_up_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_down_{col}_{tag}_pct"] = np.nan
            init_cols[f"baseline_next_week_sd_abs_{col}_{tag}_pct"] = np.nan
    x = pd.concat([x, pd.DataFrame(init_cols, index=x.index)], axis=1).copy()

    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)

            abn_col = f"abn_{col}_{tag}"

            for i in x.index:
                start = max(0, i - baseline_window)
                hist = x.iloc[start:i].copy()
                if len(hist) == 0:
                    continue

                hist = hist[hist[abn_col] == 0].copy()
                if len(hist) == 0:
                    continue

                x.at[i, f"baseline_next_week_max_up_{col}_{tag}_pct"] = hist["next_week_max_up_pct"].mean()
                x.at[i, f"baseline_next_week_max_down_{col}_{tag}_pct"] = hist["next_week_max_down_pct"].mean()
                x.at[i, f"baseline_next_week_max_abs_{col}_{tag}_pct"] = hist["next_week_max_abs_pct"].mean()

                x.at[i, f"baseline_next_week_sd_up_{col}_{tag}_pct"] = hist["next_week_max_up_pct"].std(ddof=1)
                x.at[i, f"baseline_next_week_sd_down_{col}_{tag}_pct"] = hist["next_week_max_down_pct"].std(ddof=1)
                x.at[i, f"baseline_next_week_sd_abs_{col}_{tag}_pct"] = hist["next_week_max_abs_pct"].std(ddof=1)

    return x


def summarize_weekly_multiq(
    weekly_df,
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
):
    rows = []

    for col in vol_cols:
        for q in abnormal_qs:
            tag = _qtag(q)
            abn = weekly_df[weekly_df[f"abn_{col}_{tag}"] == 1].copy()
            if len(abn) == 0:
                continue

            row = {
                "pattern": col,
                "q_label": int(round(q * 100)),
                "n_abnormal_weeks": len(abn),
                "avg_rel_to_baseline": abn[f"rel_{col}"].mean(),
                "current_window_max_up_pct": abn["current_window_max_up_pct"].mean(),
                "current_window_max_down_pct": abn["current_window_max_down_pct"].mean(),
                "current_window_max_abs_pct": abn["current_window_max_abs_pct"].mean(),
            }

            for side in ["up", "down", "abs"]:
                abn_col = f"next_week_max_{side}_pct"
                base_col = f"baseline_next_week_max_{side}_{col}_{tag}_pct"
                diff = abn[abn_col] - abn[base_col]
                tt = _safe_ttest_zero(diff)

                row[f"abn_next_week_max_{side}_pct"] = abn[abn_col].mean()
                row[f"baseline_next_week_max_{side}_pct"] = abn[base_col].mean()
                row[f"excess_next_week_max_{side}_pct"] = tt["mean"]
                row[f"excess_sd_next_week_max_{side}_pct"] = tt["sd"]
                row[f"excess_t_next_week_max_{side}_pct"] = tt["tstat"]
                row[f"excess_p_next_week_max_{side}_pct"] = tt["pvalue"]
                row[f"global_future_max_{side}_pct"] = weekly_df[abn_col].mean()

            rows.append(row)

    return pd.DataFrame(rows).sort_values(["pattern", "q_label"]).reset_index(drop=True)


def plot_weekly_multiq_to_pdf(weekly_summary, pdf, ticker, annotate_lower_than_baseline=False):
    if len(weekly_summary) == 0:
        return

    x = weekly_summary.copy()
    patterns = list(x["pattern"].drop_duplicates())
    q_labels = sorted(x["q_label"].unique())
    xpos = np.arange(len(patterns), dtype=float)
    width = 0.8 / max(len(q_labels), 1)

    side_specs = [
        ("up", "Max Up Drift (%)"),
        ("down", "Max Down Drift (%)"),
        ("abs", "Max Absolute Drift (%)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=False)
    fill_handles = [
        Patch(facecolor="white", edgecolor="black", label="post-abnormal future drift"),
        Patch(facecolor="white", edgecolor="black", hatch="//", label="matched baseline future drift"),
        Patch(facecolor="#F0F0F0", edgecolor="black", hatch="..", label="global average future drift"),
        Patch(facecolor="#BFBFBF", edgecolor="black", label="realized drift inside abnormal window"),
    ]
    if annotate_lower_than_baseline:
        fill_handles.append(Patch(facecolor="#D9D9D9", edgecolor="black", hatch="xx", label="post-abnormal future drift below matched baseline"))

    for ax, (side, ylabel) in zip(axes, side_specs):
        for j, q_label in enumerate(q_labels):
            sub = x[x["q_label"] == q_label].set_index("pattern").reindex(patterns)
            offsets = xpos - 0.4 + (j + 0.5) * width
            abn_vals = sub[f"abn_next_week_max_{side}_pct"].to_numpy(dtype="float64")
            base_vals = sub[f"baseline_next_week_max_{side}_pct"].to_numpy(dtype="float64")
            global_vals = sub[f"global_future_max_{side}_pct"].to_numpy(dtype="float64")
            current_vals = sub[f"current_window_max_{side}_pct"].to_numpy(dtype="float64")
            sub_width = width / 4.0
            bars = ax.bar(offsets - 1.5 * sub_width, abn_vals, width=sub_width, facecolor="white", edgecolor="black", linewidth=1.0)
            if annotate_lower_than_baseline:
                lower_mask = np.isfinite(abn_vals) & np.isfinite(base_vals) & (abn_vals < base_vals)
                for bar, is_lower in zip(bars, lower_mask):
                    if is_lower:
                        bar.set_hatch("xx")
                        bar.set_facecolor("#D9D9D9")
            ax.bar(offsets - 0.5 * sub_width, base_vals, width=sub_width, facecolor="white", edgecolor="black", linewidth=1.0, hatch="//")
            ax.bar(offsets + 0.5 * sub_width, global_vals, width=sub_width, facecolor="#F0F0F0", edgecolor="black", linewidth=1.0, hatch="..")
            ax.bar(offsets + 1.5 * sub_width, current_vals, width=sub_width, facecolor="#BFBFBF", edgecolor="black", linewidth=1.0)

        ax.set_ylabel(ylabel)
        ax.set_xlabel("Pattern group (within each group: q90, q95, q99 from left to right)")
        ax.set_xticks(xpos)
        ax.set_xticklabels(patterns, rotation=15)
        ax.set_title(f"{ticker} - Weekly Summary - {side.upper()} (within each pattern: q90, q95, q99 from left to right)")
        ax.grid(axis="y", alpha=0.25)
        if side == "down":
            ax.axhline(0, color="black", linewidth=1)

    axes[0].legend(handles=fill_handles, title="Fill Meaning", loc="best")
    fig.text(0.5, 0.01, "For each pattern group, bars are ordered left to right as q90, q95, q99.", ha="center", fontsize=10)
    _save_current_figure_to_pdf(pdf)


def plot_weekly_multiq_lines_to_pdf(weekly_summary, pdf, ticker):
    if len(weekly_summary) == 0:
        return

    for pattern in weekly_summary["pattern"].drop_duplicates():
        sub = weekly_summary[weekly_summary["pattern"] == pattern].sort_values("q_label")
        q_vals = sub["q_label"].to_numpy(dtype="float64")
        side_specs = [
            ("up", "Max Up Drift (%)"),
            ("down", "Max Down Drift (%)"),
            ("abs", "Max Absolute Drift (%)"),
        ]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
        for ax, (side, ylabel) in zip(axes, side_specs):
            ax.plot(q_vals, sub[f"abn_next_week_max_{side}_pct"], marker="o", color="black", label="post-abnormal future drift")
            ax.plot(q_vals, sub[f"baseline_next_week_max_{side}_pct"], marker="o", linestyle="--", color="black", label="matched baseline future drift")
            ax.plot(q_vals, sub[f"global_future_max_{side}_pct"], marker="o", linestyle=":", color="black", alpha=0.8, label="global average future drift")
            ax.plot(q_vals, sub[f"current_window_max_{side}_pct"], marker="o", linestyle="-.", color="gray", label="realized drift inside abnormal window")
            ax.set_ylabel(ylabel)
            ax.set_title(f"{ticker} - Weekly Line View - {pattern} - {side.upper()}")
            ax.grid(alpha=0.25)
            if side == "down":
                ax.axhline(0, linewidth=1, color="black")
        axes[-1].set_xlabel("Quantile")
        axes[-1].set_xticks(q_vals)
        axes[-1].set_xticklabels([f"q{int(v)}" for v in q_vals])
        axes[0].legend()
        _save_current_figure_to_pdf(pdf)


def run_weekly_effect_module_multiq(
    df,
    price_col="vwap",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    abnormal_qs=(0.90, 0.95, 0.99),
    baseline_window=20,
):
    weekly_df = build_weekly_panel_for_excursion(
        df=df,
        price_col=price_col,
        vol_cols=vol_cols,
    )

    weekly_df = add_weekly_abnormal_flags_multiq(
        weekly_df=weekly_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    weekly_df = build_weekly_event_baselines_multiq(
        weekly_df=weekly_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
        baseline_window=baseline_window,
    )

    weekly_summary = summarize_weekly_multiq(
        weekly_df=weekly_df,
        vol_cols=vol_cols,
        abnormal_qs=abnormal_qs,
    )

    return weekly_df, weekly_summary

# ===== Cell 5 =====
def plot_all_microstructure_summary_to_pdf(df, pdf, ticker):
    x = _copy_and_prepare(df)

    cols = [
        "trade_volume_at_mid",
        "trade_volume_below_mid",
        "trade_volume_above_mid",
    ]

    for col in cols:
        s = _sample_series(x[col], max_points=300000, clip_q=0.995)
        s = np.log1p(s)

        plt.figure(figsize=(10, 5))
        plt.hist(s, bins=120)
        plt.xlabel(f"log(1 + {col})")
        plt.ylabel("Frequency")
        plt.title(f"{ticker} - Distribution of {col}")
        _save_current_figure_to_pdf(pdf)

    s = _sample_series(x["vwap_diff"], max_points=300000, clip_q=0.995)
    plt.figure(figsize=(10, 5))
    plt.hist(s, bins=120)
    plt.xlabel("Minute-to-minute VWAP difference")
    plt.ylabel("Frequency")
    plt.title(f"{ticker} - Distribution of VWAP Differences")
    _save_current_figure_to_pdf(pdf)

    s = _sample_series(x["spread"], max_points=300000, clip_q=0.995)
    s = s[s > 0]
    plt.figure(figsize=(10, 5))
    if len(s) > 0:
        plt.hist(s, bins=np.logspace(np.log10(s.min()), np.log10(s.max()), 120))
        plt.xscale("log")
    else:
        plt.text(0.5, 0.5, "No positive bid-ask spreads available", ha="center", va="center")
        plt.xticks([])
    plt.xlabel("Ask - Bid Spread")
    plt.ylabel("Frequency")
    plt.title(f"{ticker} - Distribution of Bid-Ask Spread (Log Scale)")
    _save_current_figure_to_pdf(pdf)

    monthly_spread = x.groupby("month", as_index=False)["spread"].mean()
    plt.figure(figsize=(12, 5))
    plt.plot(monthly_spread["month"], monthly_spread["spread"])
    plt.xlabel("Month")
    plt.ylabel("Average Spread")
    plt.title(f"{ticker} - Monthly Average Bid-Ask Spread")
    plt.xticks(rotation=45)
    _save_current_figure_to_pdf(pdf)


def run_full_batch_analysis_multiq(
    count_base_dir="data_2020_2025_count",
    main_base_dir="data_2020_2025",
    pdf_path="all_ticker_results_multiq.pdf",
    abnormal_qs=(0.90, 0.95, 0.99),
    intraday_seq_windows=(5, 10, 20, 30, 60),
    intraday_horizons=DEFAULT_INTRADAY_HORIZONS,
    intraday_baseline_window=390 * 20,
    intraday_min_run=2,
    daily_baseline_window=20,
    weekly_baseline_window=20,
    pdf_assets_per_file=1,
    show_progress=True,
):
    tickers = _get_common_tickers(
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
    )

    results = {}
    total_steps = max(len(tickers), 1) * 12
    step = 0

    def advance_progress(stage_label):
        nonlocal step
        step += 1
        if show_progress:
            _render_progress(step, total_steps, prefix=stage_label, final=(step == total_steps))

    for ticker in tickers:
        ticker_pdf_path = _derive_ticker_pdf_path(pdf_path, ticker)
        with PdfPages(ticker_pdf_path) as pdf:
            df = load_count_aligned_dataset(
                ticker=ticker,
                count_base_dir=count_base_dir,
                main_base_dir=main_base_dir,
                first_time="09:30:00",
            )
            advance_progress(f"{ticker} load")

            df_feat, seq_df, seq_outcome_df, intraday_summary = run_intraday_effect_module_multiq(
                df=df,
                price_col="vwap",
                vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
                seq_windows=intraday_seq_windows,
                abnormal_qs=abnormal_qs,
                horizons=intraday_horizons,
                baseline_window=intraday_baseline_window,
                min_run=intraday_min_run,
                progress_prefix=f"{ticker} intraday baseline" if show_progress else None,
            )
            advance_progress(f"{ticker} intraday")

            daily_df, daily_summary = run_daily_effect_module_multiq(
                df=df,
                price_col="vwap",
                vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
                abnormal_qs=abnormal_qs,
                baseline_window=daily_baseline_window,
            )
            advance_progress(f"{ticker} daily")

            weekly_df, weekly_summary = run_weekly_effect_module_multiq(
                df=df,
                price_col="vwap",
                vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
                abnormal_qs=abnormal_qs,
                baseline_window=weekly_baseline_window,
            )
            advance_progress(f"{ticker} weekly")

            imbalance_intraday_df, imbalance_seq_df, imbalance_seq_outcome_df, imbalance_intraday_summary = run_intraday_imbalance_effect_module_multiq(
                df=df,
                seq_windows=intraday_seq_windows,
                abnormal_qs=abnormal_qs,
                horizons=intraday_horizons,
                baseline_window=intraday_baseline_window,
                min_run=intraday_min_run,
                progress_prefix=f"{ticker} imbalance baseline" if show_progress else None,
            )

            imbalance_daily_df, imbalance_daily_summary = run_daily_imbalance_effect_module_multiq(
                df=df,
                price_col="vwap",
                abnormal_qs=abnormal_qs,
                baseline_window=daily_baseline_window,
            )

            imbalance_weekly_df, imbalance_weekly_summary = run_weekly_imbalance_effect_module_multiq(
                df=df,
                price_col="vwap",
                abnormal_qs=abnormal_qs,
                baseline_window=weekly_baseline_window,
            )

            drift_volume = run_drift_volume_link_analysis(
                df=df,
                price_col="vwap",
                intraday_unit_mins=max(intraday_seq_windows),
                rolling_window=20,
            )
            volume_max_drift = run_volume_max_drift_analysis(
                df=df,
                price_col="vwap",
                intraday_unit_mins=(30, 60),
            )
            corr_pvalue_table = _prettify_pvalue_table(build_volume_drift_pvalue_table(volume_max_drift))

            results[ticker] = {
                "df": df,
                "intraday": {
                    "df_feat": df_feat,
                    "seq_df": seq_df,
                    "seq_outcome_df": seq_outcome_df,
                    "summary_df": intraday_summary,
                },
                "daily": {
                    "daily_df": daily_df,
                    "summary_df": daily_summary,
                },
                "weekly": {
                    "weekly_df": weekly_df,
                    "summary_df": weekly_summary,
                },
                "imbalance": {
                    "intraday": {
                        "df_feat": imbalance_intraday_df,
                        "seq_df": imbalance_seq_df,
                        "seq_outcome_df": imbalance_seq_outcome_df,
                        "summary_df": imbalance_intraday_summary,
                    },
                    "daily": {
                        "daily_df": imbalance_daily_df,
                        "summary_df": imbalance_daily_summary,
                    },
                    "weekly": {
                        "weekly_df": imbalance_weekly_df,
                        "summary_df": imbalance_weekly_summary,
                    },
                },
                "drift_volume": drift_volume,
                "volume_max_drift": volume_max_drift,
                "corr_pvalue_table": corr_pvalue_table,
            }

            trigger_count_lines = _format_trigger_count_tables(results[ticker])

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Overview",
                lines=[
                    f"Ticker: {ticker}",
                    f"Rows in minute df: {len(df)}",
                    f"Intraday abnormal sequences: {len(seq_df)}",
                ],
            )
            advance_progress(f"{ticker} overview pdf")

            _pdf_fixed_count_pages(
                pdf,
                title=f"{ticker} - Trigger Counts",
                tables=trigger_count_lines,
                n_pages=3,
                body_fontsize=8.5,
                line_height=0.024,
            )
            advance_progress(f"{ticker} trigger-count pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Microstructure",
                lines=[
                    "This section contains distribution and monthly plots for merged minute data."
                ],
            )
            plot_all_microstructure_summary_to_pdf(df, pdf, ticker)
            advance_progress(f"{ticker} microstructure pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Intraday Multi-Quantile Results",
                lines=[
                    "q90 / q95 / q99 intraday excessive-sequence results versus baseline.",
                    "Post-abnormal future drift means the future Up / Down / Abs excursion after the abnormal sequence has been confirmed.",
                    "Matched baseline future drift is the corresponding non-abnormal baseline reference.",
                    "Dotted lines show global average future drift; dash-dot lines show realized drift already inside the abnormal window itself.",
                    "Down-side values are signed and are typically negative.",
                    "P-values are from t-tests on event-level excess versus rolling normal baseline excursion.",
                ],
            )
            plot_intraday_multiq_to_pdf(intraday_summary, pdf, ticker, horizons=intraday_horizons)
            advance_progress(f"{ticker} intraday pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Daily Multi-Quantile Results",
                lines=[
                    "q90 / q95 / q99 daily excessive-volume results versus baseline.",
                    "Post-abnormal future drift means next-day Up / Down / Abs excursion after the abnormal day has been identified.",
                    "Matched baseline future drift is the corresponding non-abnormal baseline reference.",
                    "Additional bars show global average next-day drift and realized drift already inside the abnormal day itself.",
                    "Independent line-view charts are also included, using quantile on the x-axis.",
                    "Down-side values are signed and are typically negative.",
                    "P-values are from excessive-day excess over rolling normal-day baseline.",
                ],
            )
            plot_daily_multiq_to_pdf(daily_summary, pdf, ticker)
            plot_daily_multiq_lines_to_pdf(daily_summary, pdf, ticker)
            advance_progress(f"{ticker} daily pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Weekly Summary Results",
                lines=[
                    "Weekly results are shown as combined Up / Down / Abs overview charts.",
                    "Post-abnormal future drift means next-week Up / Down / Abs excursion after the abnormal week has been identified.",
                    "Matched baseline future drift is the corresponding non-abnormal baseline reference.",
                    "Additional bars show global average next-week drift and realized drift already inside the abnormal week itself.",
                    "Within each pattern group, the weekly bars are ordered from left to right as q90, q95, q99.",
                    "Legend fill and hatch styles explain which bars are post-abnormal future drift, matched baseline, global average future drift, and realized abnormal-window drift.",
                    "Independent line-view charts are also included, using quantile on the x-axis.",
                    "Values are next-week excursions (%) relative to current week close.",
                    "Down-side values are signed and are typically negative.",
                ],
            )
            plot_weekly_multiq_to_pdf(weekly_summary, pdf, ticker)
            plot_weekly_multiq_lines_to_pdf(weekly_summary, pdf, ticker)
            advance_progress(f"{ticker} weekly pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Drift and Volume-Change Link",
                lines=[
                    "This section focuses on direct volume-drift links without repeating lower-value rolling-correlation charts.",
                    "Scatter plots compare total / above-mid / at-mid / below-mid volume against max up/down drift.",
                    "Correlation p-values for these plots are summarized in the P-Value Summary tables.",
                    "Intraday volume-drift scatter plots are only produced for 30-minute and 60-minute windows.",
                ],
            )
            _pdf_text_page(
                pdf,
                title=f"{ticker} - P-Value Summary",
                lines=[
                    "This section reports correlation p-values for the scatter/regression plots.",
                    "Intraday rows are restricted to 30m and 60m windows for readability.",
                ],
            )
            _pdf_dataframe_table_pages(
                pdf,
                base_title=f"{ticker} - P-Value Summary",
                section_title="Correlation p-values",
                df=results[ticker]["corr_pvalue_table"],
                max_rows_per_page=12,
                max_cols_per_page=5,
                empty_message="No correlation p-values",
                body_fontsize=7.8,
            )
            for intraday_window in sorted(volume_max_drift["intraday"]):
                for volume_item in volume_max_drift["intraday"][intraday_window].values():
                    plot_volume_max_drift_layer_to_pdf(
                        volume_item["panel"],
                        pdf,
                        ticker,
                        f"intraday {intraday_window}m",
                        volume_item["label"],
                    )
            for volume_item in volume_max_drift["daily"].values():
                plot_volume_max_drift_layer_to_pdf(
                    volume_item["panel"],
                    pdf,
                    ticker,
                    "daily",
                    volume_item["label"],
                )
            for volume_item in volume_max_drift["weekly"].values():
                plot_volume_max_drift_layer_to_pdf(
                    volume_item["panel"],
                    pdf,
                    ticker,
                    "weekly",
                    volume_item["label"],
                )
            advance_progress(f"{ticker} drift-volume pdf")

            _pdf_text_page(
                pdf,
                title=f"{ticker} - Above/Below Mid Imbalance",
                lines=[
                    "This section isolates bid/ask-side imbalance from overall volume level.",
                    "The trigger statistic is the signed count-volume difference: above_mid minus below_mid.",
                    "Intraday uses rolling cumulative imbalance over each sequence window; daily and weekly use aggregated imbalance at their own layer.",
                    "Upper-tail q90 / q95 / q99 events mark above-mid-dominant imbalance; lower-tail events mark below-mid-dominant imbalance.",
                    "Baselines are rolling non-imbalanced observations at the same layer.",
                    "Post-abnormal future drift is distinct from realized drift already inside the abnormal window.",
                    "Global-drift and realized-abnormal-window references are also shown to check whether imbalance is just capturing existing drift segments.",
                    "Filled imbalance markers/bars indicate abnormal outcomes that are smaller than their matched baseline.",
                ],
            )
            plot_intraday_multiq_to_pdf(
                imbalance_intraday_summary,
                pdf,
                ticker,
                horizons=intraday_horizons,
                annotate_lower_than_baseline=True,
            )
            plot_daily_multiq_to_pdf(
                imbalance_daily_summary,
                pdf,
                ticker,
                annotate_lower_than_baseline=True,
            )
            plot_daily_multiq_lines_to_pdf(imbalance_daily_summary, pdf, ticker)
            plot_weekly_multiq_to_pdf(
                imbalance_weekly_summary,
                pdf,
                ticker,
                annotate_lower_than_baseline=True,
            )
            plot_weekly_multiq_lines_to_pdf(imbalance_weekly_summary, pdf, ticker)
            advance_progress(f"{ticker} imbalance pdf")

    if show_progress and step < total_steps:
        _render_progress(total_steps, total_steps, prefix="Progress", final=True)

    return results

# ===== Cell 6 =====
def count_abnormal_triggers_all_tickers(
    count_base_dir="data_2020_2025_count",
    main_base_dir="data_2020_2025",
    vol_cols=("trade_volume_below_mid", "trade_volume_above_mid", "trade_volume_at_mid"),
    seq_windows=(5, 10, 20, 30, 60),
    intraday_baseline_window=390 * 20,
    daily_baseline_window=20,
    weekly_baseline_window=20,
    abnormal_q=0.99,
):
    tickers = _get_common_tickers(
        count_base_dir=count_base_dir,
        main_base_dir=main_base_dir,
    )

    rows = []

    for ticker in tickers:
        df = load_count_aligned_dataset(
            ticker=ticker,
            count_base_dir=count_base_dir,
            main_base_dir=main_base_dir,
            first_time="09:30:00",
        ).copy()

        df["minute_dt"] = pd.to_datetime(df["minute_dt"])
        df = df.sort_values("minute_dt").reset_index(drop=True)
        df["date"] = df["minute_dt"].dt.date
        df["week"] = df["minute_dt"].dt.to_period("W").astype(str)

        for c in vol_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        minute_counts = {}
        min_hist_intraday = max(200, intraday_baseline_window // 5)

        for col in vol_cols:
            for w in seq_windows:
                cum_col = df[col].rolling(w, min_periods=w).sum()
                q_col = cum_col.rolling(intraday_baseline_window, min_periods=min_hist_intraday).quantile(abnormal_q)
                abn_flag = (cum_col >= q_col).astype(int)
                minute_counts[f"intraday_{col}_{w}m_count"] = int(abn_flag.sum())

        daily = (
            df.groupby("date", as_index=False)
            .agg(**{c: (c, "sum") for c in vol_cols})
            .sort_values("date")
            .reset_index(drop=True)
        )

        daily_counts = {}
        min_hist_daily = max(5, daily_baseline_window // 4)

        for col in vol_cols:
            q_col = daily[col].rolling(daily_baseline_window, min_periods=min_hist_daily).quantile(abnormal_q)
            abn_flag = (daily[col] >= q_col).astype(int)
            daily_counts[f"daily_{col}_count"] = int(abn_flag.sum())

        weekly = (
            df.groupby("week", as_index=False)
            .agg(**{c: (c, "sum") for c in vol_cols})
            .sort_values("week")
            .reset_index(drop=True)
        )

        weekly_counts = {}
        min_hist_weekly = max(5, weekly_baseline_window // 4)

        for col in vol_cols:
            q_col = weekly[col].rolling(weekly_baseline_window, min_periods=min_hist_weekly).quantile(abnormal_q)
            abn_flag = (weekly[col] >= q_col).astype(int)
            weekly_counts[f"weekly_{col}_count"] = int(abn_flag.sum())

        row = {
            "ticker": ticker,
            "n_minute_rows": len(df),
            "n_days": len(daily),
            "n_weeks": len(weekly),
        }
        row.update(minute_counts)
        row.update(daily_counts)
        row.update(weekly_counts)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)

# ===== Cell 7 =====
if __name__ == "__main__":
    results = run_full_batch_analysis_multiq(
        count_base_dir="data_2020_2025_count",
        main_base_dir="data_2020_2025",
        pdf_path="all_ticker_results_multiq.pdf",
        abnormal_qs=(0.90, 0.95, 0.99),
        intraday_seq_windows=(5, 10, 20, 30, 60),
        intraday_horizons=DEFAULT_INTRADAY_HORIZONS,
        intraday_baseline_window=390 * 20,
        intraday_min_run=2,
        daily_baseline_window=20,
        weekly_baseline_window=20,
        show_progress=True,
    )

