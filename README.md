# PPG + ECG 免袖带血压估计实验

用 16 路多波长 PPG（红外、红、绿、蓝光，各 4 个位置）、单导联 ECG、三轴加速度和人口学信息，估计每轮 1 分钟记录的收缩压（SBP）和舒张压（DBP）。

*An experiment in snapshot cuffless blood-pressure estimation from 16-channel multi-wavelength PPG, single-lead ECG, accelerometry and demographics. It covers beat-level signal processing and physiological features, subject-level CatBoost and CNN models, subject-grouped cross-validation reweighted to an AAMI/ISO 81060-2-style BP distribution, a label-shift-aware posterior-median correction, and a comparison with pyPPG biomarkers. The key findings are listed below in Chinese.*

## 主要结论

1. **这是受试者层面的问题。** SBP 在人与人之间的标准差是 14.8 mmHg，同一人各轮之间只有 2.3 mmHg。所以模型按人拟合，再加回每轮的平均偏移。
2. **信号在人口学之外只多出一点信息。** 在较年轻的招募队列（编号 <1000，约 700 人）里，和只用人口学相比：
   - SBP 的 R² 只从 0.541 提到 0.577；
   - DBP 从 0.216 提到 0.462，提升是实在的。

   三类完全不同的模型（CatBoost + 生理特征、节拍模板网络、原始波形网络）收敛到同一个上限。
3. **pyPPG 生物标志物没有多出信息。** 每拍 102 项生物标志物取中位数和四分位距、按人汇总后加入我们的特征，对照组是打乱人头的同一张表：
   - wMAE 变化 +0.020，95% 区间 [−0.066, +0.106]；
   - 和混合模型残差的相关都低于置换阈值；
   - 在年轻队列里，单用 pyPPG 和我们的特征持平。
4. **离临床精度还很远。** ISO 81060-2 准则 1 要求平均误差在 ±5 mmHg 以内、标准差不超过 8 mmHg：
   - 按训练集的血压分布：SBP 误差标准差 9.5，不达标；DBP 6.8，达标。
   - 按 AAMI/ISO 81060-2 式的血压分布（高、低血压更多）：SBP 平均误差 −6.5、标准差 13.6；DBP −2.0、9.2，都不达标。

   误差主要来自模型向均值收缩。按目标分布做标签偏移校正（后验中位数）能减轻两端的误差，但最佳校正强度取决于对目标分布的假设。
5. **个人校准是一个很强的参照。** 每人只用第 1 轮的袖带读数、完全不用信号，去预测后面各轮：wMAE 3.15（误差标准差 SBP 4.1、DBP 2.9），而免校准模型是 6.63。所以校准后的设备看起来准，可能只是因为血压本来就没怎么变；信号自身的贡献，要在血压真的变化时才看得出来。
6. **和近期文献一致。** Di Achille et al.（*Nature* 2026）在多天聚合的数据上筛查长期高血压状态，效果接近诊室血压；但在只有几分钟数据时，信息明显有限。单次几分钟的免校准绝对血压估计，基本上是"人口学再加一点点"。

## 方法

| 环节 | 做法 | 代码 |
|---|---|---|
| 节拍 | ECG 检测 R 波；16 路 PPG 按 R 波对齐；按节律、运动和模板相关性剔除坏拍；得到稳健模板 | `ppgbp/beats.py` |
| 特征 | 脉搏到达时间、上升时间、宽度、面积、拐点和反射、APG 比值、谐波、波形采样；波长之间和位置之间的延迟与对比；ECG 间期；心率变异、呼吸调制、运动 | `ppgbp/features.py` |
| 按人汇总 | 各轮特征取中位数，按人拟合，再加回每轮的平均偏移 | `ppgbp/tables.py` |
| 模型 | CatBoost、节拍模板卷积网络、原始波形卷积网络，按权重混合（`scripts/final_pipeline.py` 默认 CatBoost 0.7 + 网络 0.3） | `ppgbp/models/` |
| 验证 | 按人分组、重复的分层 5 折交叉验证；把训练段按 AAMI/ISO 81060-2 的血压配额和保留集的人口学边际加权，模拟目标分布 | `ppgbp/cv.py`、`ppgbp/validate.py` |
| 偏移校正 | 后验中位数：残差分布 × 目标分布与训练分布的档位比^τ | `ppgbp/postprocess.py`、`ppgbp/shift.py` |

## 目录

| 目录 | 内容 |
|---|---|
| `ppgbp/` | 库：读数据、节拍、特征、交叉验证、模型、偏移校正、写出预测 |
| `scripts/` | 数据检查、缓存构建、各类交叉验证实验、网络训练、端到端流水线、校准和 ISO 统计 |
| `analysis/pyppg/` | pyPPG 生物标志物的提取和对比 |

## 数据

本仓库不包含数据。实验用的是 MISSION1BP–GMU-KIME 数据集（Zhang, 2026）：1,000 名成年受试者，每人 4–5 轮，每轮 1 分钟同步记录，前后各测一次袖带血压，标签是两次读数的平均。数据分为有标签的训练集（900 人）和不带标签的保留集（100 人）。

请按数据提供方的使用协议获取数据，放到仓库根目录下的 `data/`（其中有 `train/` 和 `test/` 两个子目录），或者用环境变量 `PPGBP_DATA` 指定位置。

## 运行

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/check_data.py          # 标签、轮次顺序、ID、重复记录检查
.venv/bin/python scripts/build_cache.py         # 节拍、特征、网络模板 -> cache/
.venv/bin/python scripts/exp_base.py            # 基础模型的交叉验证对比
.venv/bin/python scripts/run_nn.py --name b6_r3 --repeats 3 --kw '{"n_blocks": 6}'   # 训练一个网络配置（建议 GPU）
.venv/bin/python scripts/final_pipeline.py --repeats 5 --tag r5 --out predictions/final.csv   # 端到端，写出保留集预测
.venv/bin/python scripts/calibration_iso.py     # 个人校准参照和 ISO 81060-2 统计
```

- **pyPPG 对比**：pyPPG 1.0.73 钉死的依赖装不进 Python 3.13，需要单独建一个 pandas 2.2 的环境（pandas 3 会让它的链式赋值静默失效，基准点全部为空）。具体版本见 `analysis/pyppg/ppg_lib.py` 开头。用这个环境运行 `analysis/pyppg/extract.py` 提取生物标志物，再用主环境运行 `analysis/pyppg/evaluate.py --fit`。
- **已知问题**：频谱法心率特征 `hr_ppg` 约 6.5% 的片段卡在 0.6 Hz 的搜索下限，另有约 8% 偏差超过 10%，尚未修正。

## 引用

- 数据集：Y. Zhang, "Equitable and Interpretable Cuffless Blood Pressure Monitoring Using Distributed Wearable Spectral Sensors in a Multi-Cohort Trial," 2026. https://doi.org/10.21203/rs.3.rs-9176759/v1
- pyPPG：M. Á. Goda, P. H. Charlton, J. A. Behar, "pyPPG: a Python toolbox for comprehensive photoplethysmography signal analysis," *Physiological Measurement* 45(4):045001, 2024.
- P. Di Achille et al., "Opportunistically Detecting Signs of Hypertension on a Consumer Smartwatch," *Nature*, 2026. https://doi.org/10.1038/s41586-026-11336-3
