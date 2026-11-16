# 电商复购预测 Repurchase Prediction

基于 **Online Retail II 公开交易数据**的复购预测应用（Streamlit + scikit-learn），
核心演示**双时点切分防数据泄漏**的完整建模方法链路。

## 数据来源（公开，可自由迁移）

| 项 | 说明 |
|---|---|
| 数据集 | Online Retail II（英国线上礼品零售商真实交易，2009-12 ~ 2011-12） |
| 本仓库子集 | 经公开仓库 [coletangsy/Customer-Analysis-Online-Retails](https://github.com/coletangsy/Customer-Analysis-Online-Retails)（MIT License）预清洗：保留英国客户、含 CustomerID、剔除退货 → 700,386 行 / 5,350 客户 |
| 位置 | `data/online_retail2.csv.gz`（4.1MB，**离线可运行**，仅保留核心交易列） |
| 预清洗声明 | 原始 Online Retail II 为 106 万行（UCI 公开）；本子集覆盖其可识别客户的 65%，结论适用于该总体 |

## 运行

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 方法设计（本项目的核心教学点）

**双时点切分防泄漏**：

| 快照 | cutoff | 特征来源 | 标签窗口（90 天） | 客户 | 复购率 |
|---|---|---|---|---|---|
| 训练集 | 2011-06-01 | cutoff 之前 | 06-01 ~ 08-31 | 4,518 | 32.4% |
| 评估集 | 2011-09-01 | cutoff 之前 | 09-01 ~ 11-30 | 4,788 | 42.4% |

- **特征**（11 维，全部 cutoff 时点可观测）：RFM（recency/frequency/monetary）+
  在册时长、商品宽度、客单价、平均间隔 + 近 30/90 天活跃 + 月度活跃比
- **标签**：cutoff 后 90 天内是否购买（时点隔离，绝无特征穿越）
- **基线先行**：B0 概率基线（AUC 0.500）、B1 业务规则 recency<30 天（AUC 0.611）
- **模型**：M1 标准化逻辑回归（AUC 0.792，Brier 0.192 最优）、
  M2 梯度提升树（AUC 0.788）
- **实现**：纯 Python 单遍扫描构建客户快照（70 万行 → 数千行小表），
  特征/标签在循环内按 cutoff 分流累计

## 功能（4 页）

| 页 | 内容 |
|---|---|
| 🧾 数据与问题定义 | 月度活跃度、双时点设计表、防泄漏三原则、预览 |
| 🛠️ 特征工程 | 特征定义表、复购 vs 未复购对比、箱线分布、recency 分桶复购率（教学图） |
| 📈 模型评估 | 四模型指标总表、ROC、校准曲线、十分位提升度表、阈值模拟器 |
| 🔍 解释与结论 | 排列重要性、逻辑回归系数、自动结论 |

配套 **`讲解.html`**（学术论文风，浏览器直接打开）：摘要 / 问题形式化（编号公式）/
双时点切分示意（图 1）/ 三线表结果 / 讨论（预清洗影响、窗口选择、校准含义、诚实边界）/ 参考文献。

## 核心结论（外时间样本，可复算）

- 复购可预测：AUC 0.79（vs 业务规则 0.61，vs 随机 0.50）
- 业务价值：Top 10% 客户实际复购率 91%（整体 2.1 倍）；Top 20% 覆盖 39% 真实复购者
- 最强信号：最近购买间隔（recency，排列重要性第一）；「累计消费高」并非强信号

## 目录结构

```
repurchase-prediction/
├── app.py               # Streamlit 应用（4 页）
├── 讲解.html             # 学术论文风方法论文（独立视觉风格）
├── requirements.txt
├── data/online_retail2.csv.gz
└── src/
    ├── __init__.py      # 环境加载顺序说明（sklearn 先于 pandas）
    ├── features.py      # 单遍扫描特征/标签引擎（防泄漏实现）
    └── modeling.py      # 基线/模型/评估（AUC/PR/Brier/提升度/排列重要性）
```

## 推上 GitHub

```bash
git init
git add .
git commit -m "feat: 电商复购预测 v1.0（双时点切分防泄漏）"
git branch -M main
git remote add origin https://github.com/<你的用户名>/repurchase-prediction.git
git push -u origin main
```

## License

MIT
