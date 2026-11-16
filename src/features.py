# -*- coding: utf-8 -*-
"""特征与标签构建（防泄漏设计）。

方法红线（本文件核心）：
1. 时点隔离：特征只用 cutoff 之前的数据；标签只用 cutoff 之后 90 天窗口的数据。
   训练与评估集来自不同 cutoff（训练 2011-06-01，评估 2011-09-01），
   模拟真实上线时序：用「过去」训练，预测「未来」。
2. 全程纯 Python 单遍循环聚合（70 万行 → 客户级小表），无大表布尔选择——
   生产日志管道同风格，稳定且内存友好。
3. 退货/负值在源数据已被剔除（见 README 数据说明），此处只做防御式过滤。

特征定义（均为 cutoff 时点可观测值）：
  recency_days     距最近一次购买的天数
  tenure_days      距首次购买的天数（客户年龄）
  n_orders         购买订单数（按 Invoice 去重）
  n_items          购买总件数
  monetary         累计消费金额（Quantity×Price）
  distinct_products 购买过的不同商品数
  avg_order_value  客单价（monetary / n_orders）
  avg_gap_days     平均购买间隔（tenure / max(n_orders-1,1)）
  n_orders_30d     近 30 天订单数（近期活跃度）
  n_orders_90d     近 90 天订单数
  months_active    有购买的月份数 / 在册月份数（活跃稳定性）

标签：
  repurchase = 1  若该客户在 [cutoff, cutoff+90d) 内有购买
"""

from pathlib import Path

import pandas as pd
import streamlit as st

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "online_retail2.csv.gz"

# 双时点设计：训练与评估使用不同 cutoff，避免「用未来预测过去」
CUTOFFS = {
    "train": pd.Timestamp("2011-06-01"),
    "test": pd.Timestamp("2011-09-01"),
}
LABEL_WINDOW = pd.Timedelta(days=90)


@st.cache_data(ttl=600, show_spinner="正在解析 70 万行交易数据…")
def load_orders() -> pd.DataFrame:
    """读取交易明细（数据源已在准备阶段完成预清洗，详见 README）。"""
    df = pd.read_csv(
        DATA_PATH,
        usecols=["Invoice", "StockCode", "Quantity", "InvoiceDate", "Price", "CustomerID"],
        dtype={"Invoice": str, "StockCode": str},
        parse_dates=["InvoiceDate"],
    )
    df = df.dropna(subset=["InvoiceDate", "Quantity", "Price", "CustomerID"])
    # 向量化金额计算（安全模式）；数据源已剔除退货与负值（构建时验证过 0 行）
    qty = df["Quantity"].to_numpy()
    price = df["Price"].to_numpy()
    df["amount"] = qty * price
    df = df.sort_values("InvoiceDate", kind="mergesort").reset_index(drop=True)
    return df


def build_snapshot(orders: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """单遍循环构建 cutoff 时点的客户特征快照 + 90 天复购标签。

    返回客户级小表（约数千行），列见模块 docstring。
    实现：列转 Python 列表（C 层转换）后纯 Python 循环，无大表物化。
    """
    label_end = cutoff + LABEL_WINDOW

    inv_list = orders["Invoice"].tolist()
    ts_list = orders["InvoiceDate"].tolist()      # pd.Timestamp 对象列表
    cust_list = orders["CustomerID"].tolist()
    prod_list = orders["StockCode"].tolist()
    qty_list = orders["Quantity"].tolist()
    amt_list = orders["amount"].tolist()

    cust = {}
    for i in range(len(inv_list)):
        ts = ts_list[i]
        c = cust_list[i]
        if ts < cutoff:
            # ---- 特征侧（仅 cutoff 前）----
            st_ = cust.get(c)
            if st_ is None:
                cust[c] = {
                    "first": ts, "last": ts, "n_items": 0,
                    "monetary": 0.0, "products": set(), "invoices": set(),
                    "orders_30d": set(), "orders_90d": set(), "months": set(),
                }
                st_ = cust[c]
            st_["last"] = ts
            st_["n_items"] += qty_list[i]
            st_["monetary"] += amt_list[i]
            st_["products"].add(prod_list[i])
            st_["invoices"].add(inv_list[i])
            if ts >= cutoff - pd.Timedelta(days=30):
                st_["orders_30d"].add(inv_list[i])
            if ts >= cutoff - pd.Timedelta(days=90):
                st_["orders_90d"].add(inv_list[i])
            st_["months"].add((ts.year, ts.month))
        elif ts < label_end:
            # ---- 标签侧（未来窗口，绝不进入特征）----
            if c in cust:  # 只对 cutoff 前已存在的客户打标签
                cust[c]["future_buy"] = True

    # ---- 组装特征表 ----
    rows = []
    for c, st_ in cust.items():
        first, last = st_["first"], st_["last"]
        tenure = (cutoff - first).days
        n_orders = len(st_["invoices"])
        if n_orders == 0 or tenure <= 0:
            continue
        months_span = max(tenure // 30 + 1, 1)
        rows.append({
            "CustomerID": c,
            "recency_days": (cutoff - last).days,
            "tenure_days": tenure,
            "n_orders": n_orders,
            "n_items": st_["n_items"],
            "monetary": st_["monetary"],
            "distinct_products": len(st_["products"]),
            "avg_order_value": st_["monetary"] / n_orders,
            "avg_gap_days": tenure / max(n_orders - 1, 1),
            "n_orders_30d": len(st_.get("orders_30d", set())),
            "n_orders_90d": len(st_.get("orders_90d", set())),
            "active_month_ratio": len(st_["months"]) / months_span,
            "repurchase": int(st_.get("future_buy", False)),
        })
    snap = pd.DataFrame(rows)
    snap["CustomerID"] = snap["CustomerID"].astype("int64")
    return snap


def get_datasets() -> dict:
    """构建训练与评估快照（缓存）。"""
    orders = load_orders()
    train = build_snapshot(orders, CUTOFFS["train"])
    test = build_snapshot(orders, CUTOFFS["test"])
    return {"orders": orders, "train": train, "test": test}


FEATURE_COLS = [
    "recency_days", "tenure_days", "n_orders", "n_items", "monetary",
    "distinct_products", "avg_order_value", "avg_gap_days",
    "n_orders_30d", "n_orders_90d", "active_month_ratio",
]
FEATURE_CN = {
    "recency_days": "最近购买距今天数", "tenure_days": "客户在册天数", "n_orders": "历史订单数",
    "n_items": "历史购买件数", "monetary": "累计消费金额", "distinct_products": "购买过的商品数",
    "avg_order_value": "客单价", "avg_gap_days": "平均购买间隔(天)",
    "n_orders_30d": "近30天订单数", "n_orders_90d": "近90天订单数", "active_month_ratio": "月度活跃比",
}
