# 信用违约风险评分系统

基于多源数据表特征工程 + LightGBM 梯度提升的信用违约预测流水线。系统从多张异构金融数据表（贷款申请、征信记录、历史申请、分期还款、POS/现金余额、信用卡流水）中提取数千维行为特征，通过时间窗口聚合与跨表衍生构建用户风险画像，最终使用分层 K 折交叉验证训练 LightGBM 模型。

核心思路是**风险排序**而非硬分类：模型输出连续的风险分数，用于将申请人从低风险到高风险排序，阈值决策留给下游业务逻辑。

## 流水线概览

```
train_main / test_main（主申请表）
        │
        ▼
┌─────────────────┐
│  申请表特征       │  预处理、比率特征、外部评分统计、
│  (app_features)  │  独热编码、分组中位数编码
└────────┬────────┘
         │
    ┌────┴────┬──────────┬──────────┬──────────┐
    ▼         ▼          ▼          ▼          ▼
  征信记录   历史申请    POS余额    分期还款    信用卡
  bureau     prev_app   pos        inst       cc
    │         │          │          │          │
    └────┬────┴──────────┴──────────┴──────────┘
         │
         ▼
┌─────────────────┐
│  跨表衍生特征     │  关联比率、时间窗口对比、
│  (cross_ratios)  │  Top-K 近期分期特征
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  列筛选          │  移除 ~600 个低信号特征
│  (col_filter)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  LightGBM 训练   │  5折分层CV、早停、
│  (cv_trainer)    │  OOF预测、排名融合
└─────────────────┘
```

## 项目结构

```
├── dataset/                    # 原始CSV数据表（不纳入git）
├── core/
│   ├── config.py               # 全局配置：LightGBM超参数、随机种子
│   ├── feature_eng/
│   │   ├── app_features.py     # 申请表特征工程
│   │   ├── bureau_features.py  # 征信记录 + 月度余额聚合
│   │   ├── prev_app_features.py # 历史申请特征
│   │   ├── pos_features.py     # POS/现金余额特征
│   │   ├── inst_features.py    # 分期还款特征
│   │   ├── cc_features.py      # 信用卡余额特征
│   │   ├── cross_ratios.py     # 跨表比率特征
│   │   ├── recency_features.py # Top-K近期分期特征
│   │   ├── col_filter.py       # 待移除的低效特征列名
│   │   └── builder.py          # 端到端特征构建编排
│   ├── training/
│   │   └── cv_trainer.py       # LightGBM 交叉验证训练
│   └── viz/
│       └── charts.py           # 特征重要性、ROC/PR曲线可视化
├── run.py                      # 入口脚本
├── output/                     # 生成的提交文件
├── plots/                      # 生成的图表
└── requirements.txt
```

## 特征工程亮点

- **时间窗口聚合**：每张辅助表在多个时间窗口（30/90/120/365天、最近3/10笔记录）下分别聚合，捕捉短期与长期行为变化
- **透视表特征**：POS/现金和信用卡表按月份区间（2/4/12/24/36+月）做透视，构建细粒度时间画像
- **跨表关联**：辅助表聚合值与申请表字段的比率（如征信年金/申请年金），捕捉相对行为模式
- **分组编码**：学历、职业、收入类型等分类字段通过分组中位数编码
- **列筛选**：通过迭代重要性分析识别并移除约600个低信号特征

## 环境搭建与运行

```bash
# 创建虚拟环境
python -m venv .venv
.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/Mac

# 安装依赖
pip install -r requirements.txt

# 运行训练
python run.py
```

训练脚本会：
1. 从7张源表构建约3200维特征
2. 使用5折分层交叉验证训练LightGBM（200轮早停）
3. 将提交预测保存到 `output/`
4. 将特征重要性和ROC曲线图保存到 `plots/`

## 模型配置

`core/config.py` 中提供三组参数：
- **params[0]**：GPU加速，lr=0.01，强正则化（默认）
- **params[1]**：CPU友好，lr=0.015，高采样率
- **params[2]**：CPU友好，lr=0.02，bagging为主

在 `run.py` 中通过 `train_lgbm_cv(..., params_idx=1)` 切换。
