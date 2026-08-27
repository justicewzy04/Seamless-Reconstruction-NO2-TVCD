# UFill：卫星 NO₂ 柱浓度缺测填补

UFill 是一个面向卫星对流层 NO₂ 垂直柱浓度（VCD）的时空缺测填补项目。主模型融合
POMINO-GEMS、POMINO-TROPOMI、GEOS-CF、空间位置和日期特征，生成连续的日尺度
NO₂ VCD 结果。

## 数据来源

本仓库不附带原始卫星和再分析数据，需要从数据提供方下载：

- **POMINO-TROPOMI_PAI NO₂**：数据将在北京大学大气化学模型组（ACM）网站提供。
  该网站目前已经提供 POMINO-TROPOMI 和 POMINO-GEMS NO₂ 产品：
  [ACM 产品页面](https://www.pku-atmos-acm.org/acmProduct.php/)。
- **GEOS-CF v1.0**：可从 NASA NCCS 数据共享平台下载：
  [GEOS-CF 数据目录](https://portal.nccs.nasa.gov/datashare/gmao/geos-cf/)（NASA，2025）。

下载后的原始数据需要预处理为当前数据加载器使用的 NetCDF 和 NumPy 文件。默认目录为
`data/raw`，推荐结构如下：

```text
data/
├─ raw/
│  ├─ GEMS/hourly/                    POMINO-GEMS 小时产品（NetCDF，变量 VCD）
│  ├─ TROPOMI/daily/                  POMINO-TROPOMI 日产品（NetCDF，变量 VCD）
│  ├─ GEOS-CF_VM_daily/               日尺度 GEOS-CF 特征（.npy）
│  └─ space/distance/
│     └─ space_distance_normal.npy    归一化空间特征
└─ splits/
   ├─ train.txt
   ├─ val.txt
   ├─ test.txt
   └─ all.txt
```

日期划分文件每行使用 `YYYYMMDD` 格式。数据文件名必须包含对应日期；GEMS 小时文件还需
包含 `0245`、`0345`、`0445` 或 `0545` 时次标识。

## 模型输入与输出

主时序 U-Net 使用 55 个输入通道：

- 当日四个 POMINO-GEMS 时次；
- 当日 GEOS-CF；
- 空间位置和日期特征；
- 前一日及后一日 POMINO-TROPOMI，以及各自的空间和日期特征。

训练目标是当日 POMINO-TROPOMI VCD。

## 项目结构

```text
src/ufill/
├─ config.py                 可移植路径配置
├─ models/                   U-Net、ConvLSTM、SegFormer 及骨干网络
├─ data/                     主模型和七通道基线数据集
├─ training/                 损失函数、训练循环和损失记录
├─ inference/                模型权重加载与张量推理
└─ utils/                    图像、文件和通用工具
scripts/
├─ train_geo.py              主模型训练入口
└─ predict_geo.py            主模型批量预测入口
experiments/baselines/
├─ deep/                     深度学习基线
└─ classical/                传统机器学习基线
checkpoints/                 模型权重
data/splits/                 日期划分
outputs/                     训练日志、权重和预测结果
```


## 训练与预测

训练主时序 U-Net：

```powershell
python scripts/train_geo.py
```

批量预测：

```powershell
python scripts/predict_geo.py
```

训练输出默认写入 `outputs/training/geo_unet`，预测 NetCDF 默认写入
`outputs/predictions/geo_unet`。运行前请确认输入数据、空间特征、日期划分和模型权重均已
准备完成。

## 预训练权重

仓库中的 `checkpoints/unet_vgg_voc.pth` 是原项目保留的初始化权重。预测脚本默认查找
`checkpoints/best_epoch_weights.pth`；如果权重位于其他位置，请设置
`UFILL_MODEL_PATH`，或在创建预测器时显式传入 `model_path`。

