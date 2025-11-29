# utils/plot_intraday.py
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def save_intraday_html(df_market, title, out_path, start_pos=None, end_pos=None, enforce_mask=True, debug=False):
    """
    绘制分时图，并包含调试信息的打印功能。
    """
    # 1. 完整性检查
    need = {"C_t","cumVWAP_t","ref_close_t","V_t","minute_index_t","mask_t"}
    miss = need - set(df_market.columns)
    if miss:
        raise ValueError(f"df_market missing columns: {miss}")

    df = df_market.copy()

    # 2. 窗口截取
    if start_pos is not None and end_pos is not None:
        df = df.iloc[int(start_pos): int(end_pos)]

    # 3. 强制类型转换（防止 Decimal 或 Object 类型导致 0 替换失败）
    cols_to_float = ["C_t", "cumVWAP_t", "ref_close_t", "V_t", "mask_t"]
    for col in cols_to_float:
        df[col] = df[col].astype(float)

    # 4. Mask 过滤
    if enforce_mask:
        # 记录过滤前的行数
        rows_before = len(df)
        df = df[df["mask_t"] > 0.0]
        rows_after = len(df)
        if debug and rows_before != rows_after:
            print(f"[Debug] Mask filtering removed {rows_before - rows_after} rows (mask_t=0).")

    if df.empty:
        print(f"Warning: {title} has no valid rows to plot.")
        return

    # 5. 清洗 0 值 (关键步骤)
    # 将价格相关的 0.0 或极小值 替换为 NaN
    price_cols = ["C_t", "cumVWAP_t", "ref_close_t"]
    for col in price_cols:
        # 将 0.0 替换为 NaN
        df[col] = df[col].replace(0.0, np.nan)
        # 可选：如果你怀疑是极小值（如 1e-8）导致的，可以取消下面这行的注释
        # df.loc[df[col] < 1e-4, col] = np.nan

    # ================= [DEBUG START] =================
    if debug:
        print(f"\n--- DEBUG INFO for: {title} ---")
        print(f"Plotting {len(df)} rows.")
        
        # 检查是否还有 0 或 NaN
        zeros_c = (df["C_t"] == 0).sum()
        nans_c = df["C_t"].isna().sum()
        zeros_vwap = (df["cumVWAP_t"] == 0).sum()
        print(f"Stats -> C_t zeros: {zeros_c}, C_t NaNs: {nans_c}")
        print(f"Stats -> cumVWAP_t zeros: {zeros_vwap}")

        print("\n[Head 5 Rows]:")
        print(df[["minute_index_t", "C_t", "cumVWAP_t", "mask_t", "V_t"]].head(5).to_string())
        
        print("\n[Tail 5 Rows]:")
        print(df[["minute_index_t", "C_t", "cumVWAP_t", "mask_t", "V_t"]].tail(5).to_string())
        
        # 检查是否存在价格骤降点（斜线元凶）
        # 找出价格非常低（例如小于 10，假设正常价格是 800）的行
        low_price_rows = df[df["C_t"] < 10]
        if not low_price_rows.empty:
            print(f"\n[WARNING] Found {len(low_price_rows)} rows with suspiciously low Price (<10):")
            print(low_price_rows[["minute_index_t", "C_t", "cumVWAP_t", "mask_t"]].head(10).to_string())
        else:
            print("\n[Check] No suspiciously low prices (<10) found in cleaned data.")
    # ================= [DEBUG END] =================

    # 6. 画图
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.7, 0.3], vertical_spacing=0.06,
        subplot_titles=(title, "Volume")
    )

    x = df["minute_index_t"].to_numpy()

    # 主图：connectgaps=False 确保 NaN 处断开
    fig.add_trace(
        go.Scattergl(x=x, y=df["C_t"], name="Close", mode="lines", connectgaps=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(x=x, y=df["cumVWAP_t"], name="VWAP", mode="lines", connectgaps=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Scattergl(
            x=x, y=df["ref_close_t"], name="Ref Close",
            mode="lines", line=dict(dash="dash", color="gray"), connectgaps=True
        ),
        row=1, col=1
    )

    # 副图
    fig.add_trace(
        go.Bar(x=x, y=df["V_t"], name="Volume", marker_color="teal"),
        row=2, col=1
    )

    fig.update_layout(
        title=f"{title}",
        legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
        margin=dict(l=40, r=20, t=60, b=40),
        template="plotly_dark",
        height=600,
        hovermode="x unified"
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_path, include_plotlyjs="cdn")
    if debug:
        print(f"Plot saved to {out_path}\n")