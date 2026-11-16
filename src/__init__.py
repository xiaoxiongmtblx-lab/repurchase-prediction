# -*- coding: utf-8 -*-
"""src 包初始化。

⚠️ 重要（本容器环境特有）：sklearn 与 pandas 的动态库存在加载顺序冲突——
若 pandas 先加载，随后导入 sklearn 会段错误；反过来则安全。
因此在本包加载任何子模块之前，先显式导入 sklearn。
（顺序问题仅影响本容器；此写法在正常环境中无副作用。）
"""
import sklearn.linear_model  # noqa: F401  必须最先
import sklearn.ensemble  # noqa: F401
import sklearn.calibration  # noqa: F401
import sklearn.metrics  # noqa: F401
