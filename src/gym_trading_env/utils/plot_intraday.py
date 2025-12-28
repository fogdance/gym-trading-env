# src/gym_trading_env/utils/plot_intraday.py

from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def save_intraday_html(
    df_market: pd.DataFrame,
    title: str,
    out_path: str,
    start_pos: int | None = None,
    end_pos: int | None = None,
    enforce_mask: bool = True,
    debug: bool = True,
    *,
    market_features: list[str] | None = None,
    market_features_obs: list[str] | None = None,
    agent_features: list[str] | None = None,
    agent_features_obs: list[str] | None = None,
    agent_raw: dict | None = None,
    agent_obs: dict | None = None,
    focus_pos: int | None = None,
    focus_ts=None,
):
    """
    输出：
      - Market RAW 图（Close/VWAP/Ref + Volume；OI 叠加在 Volume 子图右轴，折线）
      - Market OBS 图（obs_Close/obs_VWAP/obs_Ref + obs_Volume；obs_OI 同上）
      - Market 表格：focus 点 raw vs obs 全量指标
      - Agent 表格：focus 点 raw vs obs 全量指标（如果提供 agent_raw/agent_obs）
    """
    if not isinstance(df_market, pd.DataFrame):
        raise TypeError(f"df_market must be pd.DataFrame, got {type(df_market)}")

    # 防止把 DataFrame 误当作 features 传进来
    if market_features is not None and not isinstance(market_features, (list, tuple)):
        raise TypeError(f"market_features must be list[str] | None, got {type(market_features)}")
    if market_features_obs is not None and not isinstance(market_features_obs, (list, tuple)):
        raise TypeError(f"market_features_obs must be list[str] | None, got {type(market_features_obs)}")

    df = df_market.copy()

    # 1) 截取窗口（init: 0..DAY_LEN；obs: start..end 只画 window_size）
    if start_pos is not None and end_pos is not None:
        df = df.iloc[int(start_pos): int(end_pos)]

    if df.empty:
        print(f"Warning: {title} has no rows to plot.")
        return

    # 2) 默认 features（不传则自动 import）
    if market_features is None or market_features_obs is None or agent_features is None or agent_features_obs is None:
        try:
            from gym_trading_env.utils.market_features import FEATURES_MARKET, FEATURES_MARKET_OBS
        except Exception:
            FEATURES_MARKET, FEATURES_MARKET_OBS = [], []

        try:
            from gym_trading_env.utils.agent_features import FEATURES_AGENT
        except Exception:
            FEATURES_AGENT = []

        try:
            from gym_trading_env.utils.agent_features import FEATURES_AGENT_OBS
        except Exception:
            FEATURES_AGENT_OBS = []

        market_features = list(FEATURES_MARKET) if market_features is None else list(market_features)
        market_features_obs = list(FEATURES_MARKET_OBS) if market_features_obs is None else list(market_features_obs)
        agent_features = list(FEATURES_AGENT) if agent_features is None else list(agent_features)
        agent_features_obs = list(FEATURES_AGENT_OBS) if agent_features_obs is None else list(agent_features_obs)
    else:
        market_features = list(market_features)
        market_features_obs = list(market_features_obs)
        agent_features = list(agent_features) if agent_features is not None else []
        agent_features_obs = list(agent_features_obs) if agent_features_obs is not None else []

    # 3) x 轴（优先 raw minute_index_t）
    if "minute_index_t" in df.columns:
        x_col = "minute_index_t"
    elif "obs_minute_index_t" in df.columns:
        x_col = "obs_minute_index_t"
    else:
        raise ValueError("df_market missing minute_index_t/obs_minute_index_t for plotting x-axis")

    # 4) focus 点（用于表格）
    if focus_ts is not None and focus_ts in df.index:
        focus_idx = focus_ts
    elif focus_pos is not None and 0 <= int(focus_pos) < len(df):
        focus_idx = df.index[int(focus_pos)]
    else:
        # 最后一个有效点优先
        focus_idx = df.index[-1]
        for mcol in ["mask_t", "obs_mask_t"]:
            if mcol in df.columns:
                vv = df.index[(pd.to_numeric(df[mcol], errors="coerce").fillna(0.0) > 0.0).to_numpy()]
                if len(vv) > 0:
                    focus_idx = vv[-1]
                    break

    # 5) dtype 清洗
    def _to_float_cols(cols: list[str]):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    _to_float_cols([
        x_col,
        "mask_t", "C_t", "cumVWAP_t", "ref_close_t", "V_t", "I_t",
        "obs_mask_t", "obs_C_t", "obs_cumVWAP_t", "obs_ref_close_t", "obs_V_t", "obs_I_t",
    ])

    # 6) mask：不删行；price -> NaN（断线），volume -> 0；OI(line) -> NaN（断线）
    def apply_mask(mask_col: str, price_cols: list[str], vol_cols: list[str], line_cols: list[str]):
        if mask_col not in df.columns:
            return
        m = (df[mask_col].fillna(0.0).to_numpy() > 0.0)
        if debug:
            print(f"[Debug] apply_mask_to_series: mask_col={mask_col}, invalid_rows={(~m).sum()}/{len(m)}")

        for c in price_cols:
            if c in df.columns:
                a = df[c].to_numpy(dtype=float, copy=False)
                a[~m] = np.nan
                df[c] = a

        for c in vol_cols:
            if c in df.columns:
                a = df[c].to_numpy(dtype=float, copy=False)
                a[~m] = 0.0
                df[c] = a

        for c in line_cols:
            if c in df.columns:
                a = df[c].to_numpy(dtype=float, copy=False)
                a[~m] = np.nan
                df[c] = a

    if enforce_mask:
        apply_mask(
            "mask_t",
            price_cols=["C_t", "cumVWAP_t", "ref_close_t"],
            vol_cols=["V_t"],
            line_cols=["I_t"],
        )
        apply_mask(
            "obs_mask_t",
            price_cols=["obs_C_t", "obs_cumVWAP_t", "obs_ref_close_t"],
            vol_cols=["obs_V_t"],
            line_cols=["obs_I_t"],
        )

    x = pd.to_numeric(df[x_col], errors="coerce").fillna(0).to_numpy(dtype=float)

    # 7) 画图：RAW（row2 开 secondary_y：Volume 左轴 + OI 右轴折线）
    fig_raw = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.06,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=(f"{title} (RAW)", "Volume / OI (RAW)")
    )

    if "C_t" in df.columns:
        fig_raw.add_trace(go.Scattergl(x=x, y=df["C_t"], name="Close", mode="lines", connectgaps=False), row=1, col=1)
    if "cumVWAP_t" in df.columns:
        fig_raw.add_trace(go.Scattergl(x=x, y=df["cumVWAP_t"], name="VWAP", mode="lines", connectgaps=False), row=1, col=1)
    if "ref_close_t" in df.columns:
        fig_raw.add_trace(
            go.Scattergl(
                x=x, y=df["ref_close_t"],
                name="Ref Close",
                mode="lines",
                line=dict(dash="dash", color="gray"),
                connectgaps=True,
            ),
            row=1, col=1
        )

    # row2: Volume (left)
    if "V_t" in df.columns:
        fig_raw.add_trace(go.Bar(x=x, y=df["V_t"], name="Volume"), row=2, col=1, secondary_y=False)

    # row2: OI line (right) —— 折线（不是直方图）
    if "I_t" in df.columns:
        s = df["I_t"].to_numpy(dtype=float, copy=False)
        if np.isfinite(s).any() and np.nanmax(np.abs(s)) > 0:
            fig_raw.add_trace(
                go.Scattergl(
                    x=x, y=df["I_t"],
                    name="OI",
                    mode="lines",
                    connectgaps=False,
                    line=dict(width=1.5, color="white"),
                ),
                row=2, col=1, secondary_y=True
            )

    fig_raw.update_layout(
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark",
        height=560,
        hovermode="x unified"
    )

    # 8) 画图：OBS（row2 same）
    fig_obs = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.06,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
        subplot_titles=(f"{title} (OBS)", "Volume / OI (OBS)")
    )

    if "obs_C_t" in df.columns:
        fig_obs.add_trace(go.Scattergl(x=x, y=df["obs_C_t"], name="obs_Close", mode="lines", connectgaps=False), row=1, col=1)
    if "obs_cumVWAP_t" in df.columns:
        fig_obs.add_trace(go.Scattergl(x=x, y=df["obs_cumVWAP_t"], name="obs_VWAP", mode="lines", connectgaps=False), row=1, col=1)
    if "obs_ref_close_t" in df.columns:
        fig_obs.add_trace(
            go.Scattergl(
                x=x, y=df["obs_ref_close_t"],
                name="obs_Ref Close",
                mode="lines",
                line=dict(dash="dash", color="gray"),
                connectgaps=True,
            ),
            row=1, col=1
        )

    # row2: Volume (left)
    if "obs_V_t" in df.columns:
        fig_obs.add_trace(go.Bar(x=x, y=df["obs_V_t"], name="obs_Volume"), row=2, col=1, secondary_y=False)

    # row2: OI line (right)
    if "obs_I_t" in df.columns:
        s = df["obs_I_t"].to_numpy(dtype=float, copy=False)
        if np.isfinite(s).any() and np.nanmax(np.abs(s)) > 0:
            fig_obs.add_trace(
                go.Scattergl(
                    x=x, y=df["obs_I_t"],
                    name="obs_OI",
                    mode="lines",
                    connectgaps=False,
                    line=dict(width=1.5, color="white"),
                ),
                row=2, col=1, secondary_y=True
            )

    fig_obs.update_layout(
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark",
        height=560,
        hovermode="x unified"
    )

    # 9) 表格：market（raw vs obs）
    def _fmt(v):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return ""
        if isinstance(v, (np.integer, int)):
            return str(int(v))
        try:
            fv = float(v)
            if abs(fv) >= 1e6:
                return f"{fv:.0f}"
            if abs(fv) >= 1e2:
                return f"{fv:.4f}"
            return f"{fv:.6f}"
        except Exception:
            return str(v)

    def _value_at(col: str):
        if col in df.columns and focus_idx in df.index:
            return df.loc[focus_idx, col]
        return np.nan

    # raw->obs 的映射：优先 "obs_"+raw
    pairs = []
    used_obs = set()

    for raw in market_features:
        if raw.startswith("obs_"):
            continue
        cand = f"obs_{raw}"
        if cand in market_features_obs or cand in df.columns:
            pairs.append((raw, cand))
            used_obs.add(cand)
        else:
            pairs.append((raw, None))

    # obs-only 的也补上
    for obs in market_features_obs:
        if obs in used_obs:
            continue
        base = obs.removeprefix("obs_")
        if base in market_features:
            continue
        pairs.append((None, obs))

    market_rows = []
    for raw, obs in pairs:
        raw_v = _value_at(raw) if raw else np.nan
        obs_v = _value_at(obs) if obs else np.nan
        market_rows.append((raw or "", _fmt(raw_v), obs or "", _fmt(obs_v)))

    market_table = (
        "<h3 style='margin:6px 0 10px 0;'>Market Features @ focus</h3>"
        f"<div style='color:#9fb3c8;font-size:12px;margin-bottom:10px;'>focus_idx: <code>{focus_idx}</code></div>"
        "<table class='feat-table'>"
        "<thead><tr><th>raw</th><th>raw_val</th><th>obs</th><th>obs_val</th></tr></thead><tbody>"
        + "".join([f"<tr><td>{r}</td><td>{rv}</td><td>{o}</td><td>{ov}</td></tr>" for r, rv, o, ov in market_rows])
        + "</tbody></table>"
    )

    # 10) 表格：agent（raw / obs 分开显示，避免错误 “obs_ + raw” 映射）
    agent_table = ""
    if agent_raw is not None or agent_obs is not None:
        agent_raw = {} if agent_raw is None else dict(agent_raw)
        agent_obs = {} if agent_obs is None else dict(agent_obs)

        raw_rows = []
        for k in agent_features:
            raw_rows.append((k, _fmt(agent_raw.get(k, None))))

        obs_rows = []
        for k in agent_features_obs:
            obs_rows.append((k, _fmt(agent_obs.get(k, None))))

        agent_table = (
            "<h3 style='margin:18px 0 10px 0;'>Agent Features</h3>"
            "<div style='display:flex; gap:12px; flex-wrap:wrap;'>"
            "<div style='flex:1; min-width:360px;'>"
            "<div style='color:#9fb3c8;font-size:12px;margin:0 0 6px 0;'>RAW</div>"
            "<table class='feat-table'>"
            "<thead><tr><th>name</th><th>val</th></tr></thead><tbody>"
            + "".join([f"<tr><td>{n}</td><td>{v}</td></tr>" for n, v in raw_rows])
            + "</tbody></table></div>"
            "<div style='flex:1; min-width:360px;'>"
            "<div style='color:#9fb3c8;font-size:12px;margin:0 0 6px 0;'>OBS</div>"
            "<table class='feat-table'>"
            "<thead><tr><th>name</th><th>val</th></tr></thead><tbody>"
            + "".join([f"<tr><td>{n}</td><td>{v}</td></tr>" for n, v in obs_rows])
            + "</tbody></table></div>"
            "</div>"
        )


    # 11) 拼 HTML
    style = """
    <style>
      body { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; background: #0b0f14; color: #e6edf3; }
      .container { max-width: 1200px; margin: 12px auto; padding: 0 12px; }
      .card { background: #0f1722; border: 1px solid #1f2a37; border-radius: 12px; padding: 12px; margin-bottom: 14px; }
      .feat-table { width: 100%; border-collapse: collapse; }
      .feat-table th, .feat-table td { padding: 6px 8px; border-bottom: 1px solid #223044; font-size: 12px; }
      .feat-table th { text-align: left; color: #9fb3c8; font-weight: 600; }
      code { color: #a5d6ff; }
    </style>
    """

    html = (
        "<html><head><meta charset='utf-8'/>"
        f"<title>{title}</title>{style}</head><body><div class='container'>"
        f"<div class='card'>{fig_raw.to_html(include_plotlyjs='cdn', full_html=False)}</div>"
        f"<div class='card'>{fig_obs.to_html(include_plotlyjs=False, full_html=False)}</div>"
        f"<div class='card'>{market_table}{agent_table}</div>"
        "</div></body></html>"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(html, encoding="utf-8")

    if debug:
        print(f"\n--- DEBUG INFO for: {title} ---")
        print(f"Rows={len(df)}, focus_idx={focus_idx}")
        if "mask_t" in df.columns:
            print(f"mask_t valid rows: {int((df['mask_t'].fillna(0.0) > 0.0).sum())}")
        if "obs_mask_t" in df.columns:
            print(f"obs_mask_t valid rows: {int((df['obs_mask_t'].fillna(0.0) > 0.0).sum())}")
        print(f"Plot saved to {out_path}\n")
