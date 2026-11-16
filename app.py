# -*- coding: utf-8 -*-
"""电商复购预测（Repurchase Prediction）

基于 Online Retail II 公开交易数据的复购预测应用（Streamlit），
核心演示：双时点切分防数据泄漏 → RFM+行为特征 → 逻辑回归/梯度提升树 →
诚实评估（基线对比 + AUC/校准/提升度 + 排列重要性）。

运行：streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import src  # noqa: F401,E402  ← 必须最先加载（见 src/__init__.py 的环境说明）

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.features import CUTOFFS, FEATURE_CN, FEATURE_COLS, get_datasets, LABEL_WINDOW
from src.modeling import (
    calibration_points,
    confusion_at_threshold,
    generate_insight,
    roc_curve_points,
    train_all,
)

st.set_page_config(page_title="电商复购预测", page_icon="🔁", layout="wide")

INK, INDIGO, PINK, ORANGE, TEAL, AMBER = "#1c1917", "#4f46e5", "#FF2E7E", "#FF6A00", "#0d9488", "#d97706"
PALETTE = [INDIGO, PINK, ORANGE, TEAL, AMBER]

st.markdown(
    """
    <style>
    .block-container{padding-top:2rem;max-width:1280px}
    div[data-testid="stMetric"]{background:#fffef9;border:1.5px solid #1c1917;border-radius:10px;
      box-shadow:3px 3px 0 rgba(28,25,23,.85);padding:14px 18px}
    main table{border-collapse:collapse;font-size:13px;width:100%;margin:8px 0}
    main table th,main table td{border:1px solid #e5decf;padding:6px 10px;text-align:left}
    main table th{background:#f0ebdd;font-family:Georgia,'Songti SC',serif}
    main table tr:nth-child(even) td{background:#faf7ee}
    </style>
    """,
    unsafe_allow_html=True,
)


def show_chart(fig):
    try:
        st.plotly_chart(fig, use_container_width=True)
    except TypeError:
        st.plotly_chart(fig)


def show_table(df):
    try:
        st.table(df)
    except Exception:
        st.markdown(df.to_html(index=False, border=0), unsafe_allow_html=True)


# ---------------- 数据与模型（缓存） ----------------
DATA = get_datasets()
RESULTS, MODELS = st.cache_resource(show_spinner="正在训练模型（时序切分）…")(
    lambda: train_all(DATA["train"], DATA["test"], FEATURE_COLS)
)()
ORDERS, TRAIN, TEST = DATA["orders"], DATA["train"], DATA["test"]

# 月度订单量（单键聚合，安全模式）
_monthly = (
    ORDERS.groupby(ORDERS["InvoiceDate"].dt.to_period("M").astype(str))
    .agg(orders=("Invoice", "nunique"), customers=("CustomerID", "nunique"), amount=("amount", "sum"))
    .reset_index()
    .rename(columns={"InvoiceDate": "month"})
)

st.sidebar.title("🔁 电商复购预测")
st.sidebar.caption("Online Retail II 公开数据 · 时序切分防泄漏")
page = st.sidebar.radio(
    "分析页",
    ["🧾 数据与问题定义", "🛠️ 特征工程", "📈 模型评估", "🔍 解释与结论"],
    label_visibility="collapsed",
)


# ================= 页 1：数据与问题定义 =================
if page == "🧾 数据与问题定义":
    st.title("🧾 数据与问题定义")
    st.caption("先把「预测什么、用什么时刻的信息预测」定义清楚——这是防泄漏的第一步")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("交易行数", f"{len(ORDERS):,}")
    with c2:
        st.metric("客户数", f"{ORDERS['CustomerID'].nunique():,}")
    with c3:
        st.metric("时间跨度", "2009-12 ~ 2011-12", help="约 24 个月的英国电商交易")
    with c4:
        st.metric("累计交易额", f"£{ORDERS['amount'].sum()/1e6:.1f}M")

    st.subheader("月度交易活跃度")
    fig = go.Figure()
    fig.add_bar(x=_monthly["month"], y=_monthly["orders"], name="订单数", marker_color=ORANGE, opacity=0.85)
    fig.add_scatter(x=_monthly["month"], y=_monthly["customers"], name="活跃客户数", mode="lines",
                    line=dict(color=INK, width=2))
    fig.update_layout(height=340, margin=dict(t=10, l=10, r=10, b=10), barmode="overlay",
                      legend=dict(orientation="h", y=1.15), yaxis_title="数量")
    show_chart(fig)
    st.caption("2011-12 的骤降是数据集本身在 12-09 截止（不满月），非业务异动——读数据先读边界。")

    st.subheader("双时点切分设计（本项目的核心方法论）")
    t_train, t_test = CUTOFFS["train"], CUTOFFS["test"]
    with st.container(border=True):
        st.markdown(
            f"""
            | 快照 | 时点 cutoff | 特征来源 | 标签窗口（未来 90 天） | 客户数 | 复购率 |
            |---|---|---|---|---|---|
            | **训练集** | {t_train.date()} | cutoff 之前的交易 | {t_train.date()} ~ {(t_train + LABEL_WINDOW).date()} | {len(TRAIN):,} | {TRAIN['repurchase'].mean():.1%} |
            | **评估集** | {t_test.date()} | cutoff 之前的交易 | {t_test.date()} ~ {(t_test + LABEL_WINDOW).date()} | {len(TEST):,} | {TEST['repurchase'].mean():.1%} |
            """
        )
        st.markdown(
            f"""
            **防泄漏三原则**：
            1. **时点隔离**：特征只看 cutoff 之前，标签只看之后——特征表里不存在任何未来信息；
            2. **时序切分**：评估发生在训练之后的时点上（{t_test.date()} > {t_train.date()}），
               模拟真实上线「用过去训练、预测未来」；
            3. **对照朴素基线**：模型必须赢过「拍脑袋规则」才算有价值（见模型评估页）。
            """
        )
    st.caption(
        f"两个 cutoff 的复购率（{TRAIN['repurchase'].mean():.0%} vs {TEST['repurchase'].mean():.0%}）"
        "差异来自圣诞季前真实的季节性回升——这正是时序切分（而非随机切分）才能暴露的分布漂移。"
    )

    st.subheader("交易明细预览")
    show_table(ORDERS.head(8))


# ================= 页 2：特征工程 =================
elif page == "🛠️ 特征工程":
    st.title("🛠️ 特征工程")
    st.caption("11 个 cutoff 时点可观测特征——全部满足「预测时刻拿得到」的约束")

    snap = st.radio("查看快照", ["训练集（cutoff 2011-06-01）", "评估集（cutoff 2011-09-01）"], horizontal=True)
    df = TRAIN if snap.startswith("训练") else TEST

    st.subheader("特征定义")
    show_table(pd.DataFrame([
        {"特征": FEATURE_CN[c], "英文名": c, "含义": d} for c, d in [
            ("recency_days", "距最近一次购买的天数（越小越活跃）"),
            ("tenure_days", "距首次购买的天数（客户在册时长）"),
            ("n_orders", "历史购买订单数（按 Invoice 去重）"),
            ("n_items", "历史购买总件数"),
            ("monetary", "累计消费金额（Quantity×Price）"),
            ("distinct_products", "购买过的不同商品数"),
            ("avg_order_value", "客单价 = 累计消费 / 订单数"),
            ("avg_gap_days", "平均购买间隔天数"),
            ("n_orders_30d", "近 30 天订单数（近期活跃）"),
            ("n_orders_90d", "近 90 天订单数"),
            ("active_month_ratio", "有购买月份占比（活跃稳定性）"),
        ]
    ]))

    st.subheader("复购 vs 未复购：关键特征对比（均值）")
    cmp_rows = []
    g = df.groupby("repurchase")[FEATURE_COLS].mean()
    for c in FEATURE_COLS:
        cmp_rows.append({
            "特征": FEATURE_CN[c],
            "未复购均值": g.loc[0, c],
            "复购均值": g.loc[1, c],
            "差异方向": "↑ 复购者更高" if g.loc[1, c] > g.loc[0, c] else "↓ 复购者更低",
        })
    show_table(pd.DataFrame(cmp_rows))
    st.caption("recency 的巨大差异（复购者显著更低）是天然的强信号——但必须由模型在防泄漏框架下量化。")

    st.subheader("特征分布对比（箱线，对数轴）")
    feat = st.selectbox("选择特征", FEATURE_COLS, format_func=lambda c: FEATURE_CN[c], index=0)
    fig = px.box(df, x="repurchase", y=feat, color="repurchase",
                 color_discrete_sequence=[AMBER, INDIGO],
                 labels={"repurchase": "标签"})
    fig.update_xaxes(tickvals=[0, 1], ticktext=["未复购", "复购"])
    fig.update_layout(height=340, margin=dict(t=10, l=10, r=10, b=10), showlegend=False,
                      yaxis_type="log" if feat in ("n_items", "monetary", "avg_order_value", "distinct_products", "n_orders") else "linear")
    show_chart(fig)

    st.subheader("单特征视角：recency 分桶复购率（教学图）")
    bins = [-1, 30, 60, 90, 180, 365, 10_000]
    labels = ["0-30天", "31-60天", "61-90天", "91-180天", "181-365天", "365天+"]
    b = pd.cut(df["recency_days"], bins=bins, labels=labels)
    rate = df.groupby(b, observed=False)["repurchase"].agg(["mean", "count"]).reset_index()
    rate.columns = ["距上次购买", "复购率", "客户数"]
    fig = px.bar(rate, x="距上次购买", y="复购率", text="客户数",
                 color_discrete_sequence=[INDIGO])
    fig.update_traces(textposition="outside")
    fig.update_layout(height=330, margin=dict(t=10, l=10, r=10, b=10), yaxis_tickformat=".0%")
    show_chart(fig)
    st.caption("柱上数字为客户数。单调下降的阶梯是「沉默即流失」的经典形态——也是业务规则基线的依据。")


# ================= 页 3：模型评估 =================
elif page == "📈 模型评估":
    st.title("📈 模型评估")
    st.caption("时序外样本 · 三个基线先行 · 排序 / 校准 / 业务价值三视角")

    st.subheader("指标总表（评估集：cutoff 2011-09-01 的 4,788 名客户）")
    rows = []
    for name, r in RESULTS.items():
        rows.append({"模型": name, "AUC-ROC": f"{r.auc:.3f}", "PR-AUC": f"{r.pr_auc:.3f}",
                     "Brier ↓": f"{r.brier:.3f}",
                     "说明": {"B0 概率基线": "一律输出训练集正类率（排序能力=随机）",
                              "B1 业务规则基线": "recency<30 天→预测复购（可解释规则）",
                              "M1 逻辑回归": "标准化特征，线性可解释",
                              "M2 梯度提升树": "非线性，捕捉交互效应"}[name]})
    show_table(pd.DataFrame(rows))
    st.caption("AUC 看排序能力（谁更可能复购排前面）；Brier 看概率校准（说 70% 的真的有 70% 发生）。")

    l, r_col = st.columns(2)
    with l:
        st.subheader("ROC 曲线")
        fig = go.Figure()
        colors = {"B0 概率基线": "#a8a29e", "B1 业务规则基线": AMBER, "M1 逻辑回归": TEAL, "M2 梯度提升树": INDIGO}
        for name, res in RESULTS.items():
            fpr, tpr = roc_curve_points(res.y_true, res.y_prob)
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={res.auc:.3f})",
                            line=dict(color=colors.get(name, INK), width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#78716c", width=1),
                        showlegend=False)
        fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                          xaxis_title="假阳性率", yaxis_title="真阳性率",
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        show_chart(fig)
    with r_col:
        st.subheader("校准曲线（提升树）")
        gb = RESULTS["M2 梯度提升树"]
        mean_pred, frac_pos = calibration_points(gb.y_true, gb.y_prob)
        fig = go.Figure()
        fig.add_scatter(x=mean_pred, y=frac_pos, mode="markers+lines", name="提升树",
                        line=dict(color=INDIGO, width=2))
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dot", color="#78716c", width=1),
                        name="完美校准", showlegend=True)
        fig.update_layout(height=380, margin=dict(t=10, l=10, r=10, b=10),
                          xaxis_title="预测复购概率", yaxis_title="实际复购频率",
                          xaxis_range=[0, 1], yaxis_range=[0, 1],
                          legend=dict(orientation="h", y=-0.25, font=dict(size=10)))
        show_chart(fig)

    st.subheader("十分位提升度表（业务价值视角）")
    show_table(gb.lift_table.round(3))
    st.caption(
        "读法：对预测概率 Top 20% 的客户做唤醒营销，可覆盖 39% 的真实复购者；"
        "Top 10% 组的复购率是整体的 2.1 倍。预算有限时，「打给谁」由这张表决定。"
    )

    st.subheader("阈值模拟器")
    thr = st.slider("判定阈值（预测概率 ≥ 阈值 → 预测复购）", 0.1, 0.9, 0.5, 0.05, format="%.2f")
    y_true, y_prob = gb.y_true, gb.y_prob
    cm = confusion_at_threshold(y_true, y_prob, thr)
    c1, c2, c3 = st.columns(3)
    y_pred = (y_prob >= thr).astype(int)
    with c1:
        st.metric("精确率（预测复购中真复购）", f"{y_true[y_pred==1].mean()*100:.0f}%" if (y_pred==1).any() else "—")
    with c2:
        st.metric("召回率（真复购被找到）", f"{y_pred[y_true==1].mean()*100:.0f}%")
    with c3:
        st.metric("营销触达量", f"{(y_pred==1).sum():,}", help="阈值越低触达越多但越不精准")
    show_table(cm)


# ================= 页 4：解释与结论 =================
elif page == "🔍 解释与结论":
    st.title("🔍 解释与结论")
    st.caption("模型怎么想的（排列重要性 + 线性系数）→ 业务怎么用（自动结论）")

    l, r_col = st.columns(2)
    with l:
        st.subheader("排列重要性（打乱特征后的 AUC 降幅）")
        imp = RESULTS["M2 梯度提升树"].extra["permutation_importance"].copy()
        imp["特征CN"] = imp["特征"].map(FEATURE_CN)
        fig = px.bar(imp.sort_values("AUC 降幅").iloc[::-1].head(8), x="AUC 降幅", y="特征CN",
                     orientation="h", color_discrete_sequence=[INDIGO])
        fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10), yaxis_title=None,
                          xaxis_title="AUC 降幅（越大越重要）")
        show_chart(fig)
    with r_col:
        st.subheader("逻辑回归系数（标准化后，可比大小）")
        coef = RESULTS["M1 逻辑回归"].extra["coefficients"].copy()
        coef["特征CN"] = coef["特征"].map(FEATURE_CN)
        coef["颜色"] = coef["标准化系数"].map(lambda v: "正向（越大约率越高）" if v > 0 else "负向（越大概率越低）")
        fig = px.bar(coef.head(8).sort_values("标准化系数"), x="标准化系数", y="特征CN",
                     orientation="h", color="颜色",
                     color_discrete_map={"正向（越大约率越高）": TEAL, "负向（越大概率越低）": PINK})
        fig.update_layout(height=400, margin=dict(t=10, l=10, r=10, b=10), yaxis_title=None,
                          legend=dict(font=dict(size=10)))
        show_chart(fig)

    st.subheader("自动结论")
    with st.container(border=True):
        for s in generate_insight(RESULTS, TRAIN, TEST):
            st.markdown(f"- {s}")
    st.caption(
        "方法论提醒：本应用的全部指标都来自 2011-09-01 时点之后的真实数据（外样本）；"
        "任何「用 12 月数据预测 12 月」式的随机切分都会虚高 AUC——那是泄漏，不是能力。"
    )


st.divider()
st.caption(
    "电商复购预测 · 数据：Online Retail II 公开交易数据（经源仓库预清洗的英国客户子集，来源见 README）· "
    "方法：双时点切分防泄漏 / 标准化逻辑回归 / HistGradientBoosting / 排列重要性 · 结论均由数据计算生成"
)
