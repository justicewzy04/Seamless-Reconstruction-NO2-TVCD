import os
import numpy as np
import lightgbm as lgb  # 引入 LightGBM
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ufill.utils.files import find_nc_files, find_npy_files, find_files_with_keyword
from ufill.config import DATA_ROOT, OUTPUT_ROOT, SPLIT_ROOT, env_path
import netCDF4 as nc
import time
from datetime import datetime, timedelta

# 数据处理函数
def find_files_with_keywords_2(file_list, keyword1, keyword2):
    matching_files = []
    for file_name in file_list:
        if keyword1 in file_name and keyword2 in file_name:
            matching_files.append(file_name)
    if len(matching_files) == 0:
        print(f"错误：文件不存在: {keyword1},{keyword2}")
        sys.exit(1)
    return matching_files

def GEMS_process(GEMS):
    GEMS = nc.Dataset(GEMS[0])
    VCD_GEMS = GEMS.variables['VCD'][:]
    VCD_GEMS[VCD_GEMS > 100] = np.nan
    VCD_GEMS = np.divide(np.subtract(VCD_GEMS, 2.3492), 4.0094)
    VCD_GEMS[np.isnan(VCD_GEMS)] = 0
    VCD_GEMS = VCD_GEMS.flatten()
    return VCD_GEMS

def TROPOMI_process(TROPOMI_file_path):
    TROPOMI = nc.Dataset(TROPOMI_file_path[0])
    VCD_TROPOMI = TROPOMI.variables['VCD'][:]
    VCD_TROPOMI[VCD_TROPOMI > 100] = np.nan
    VCD_TROPOMI = np.divide(np.subtract(VCD_TROPOMI, 1.5439), 2.5269)
    VCD_TROPOMI[np.isnan(VCD_TROPOMI)] = 0
    VCD_TROPOMI = VCD_TROPOMI.flatten()
    return VCD_TROPOMI

def GEOS_process(GEOS_file_path):
    GEOS = np.load(GEOS_file_path[0])
    GEOS = GEOS[0:1400, 0:800]
    GEOS = np.divide(np.subtract(GEOS, 2.0366), 4.1030)
    GEOS_CF = GEOS.flatten()
    return GEOS_CF

def is_leap_year(year):
    """判断是否为闰年"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)

def convert_date_to_day(year, month, day):
    """将年月日转换成年份和年中的第几天"""
    months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    if is_leap_year(year):
        months_days[1] = 29
    day_of_year = sum(months_days[:month - 1]) + day
    return f"{year}{day_of_year:03d}"

def calculate_previous_date(year, month, day):
    """计算给定日期的前一天"""
    given_date = datetime(year, month, day)
    previous_date = given_date - timedelta(days=1)
    return previous_date.strftime("%Y%m%d")

def calculate_next_date(year, month, day):
    """计算给定日期的后一天"""
    given_date = datetime(year, month, day)
    next_date = given_date + timedelta(days=1)
    return next_date.strftime("%Y%m%d")

# 主程序
start_time = time.time()

# 加载图像数据和目标值
GEMS_path = env_path("UFILL_GEMS_ROOT", DATA_ROOT / "GEMS" / "hourly")
TROPOMI_path = env_path("UFILL_TROPOMI_ROOT", DATA_ROOT / "TROPOMI" / "daily")
GEOS_path = env_path("UFILL_GEOS_ROOT", DATA_ROOT / "GEOS-CF_VM_daily")
GEMS_flist = find_nc_files(GEMS_path)
TROPOMI_list = find_nc_files(TROPOMI_path)
GEOS_flist = find_npy_files(GEOS_path)
train_list = SPLIT_ROOT / "train.txt"
with open(train_list, "r") as f:
    train_lines = f.readlines()

# 定义空的特征矩阵和目标向量
features = []
covariates = []
target_vector = []

# 提取图像特征
for line in train_lines:
    line = line.rstrip('\n')
    print(line)

    pre_date = calculate_previous_date(int(line[0:4]), int(line[4:6]), int(line[6:8]))
    nex_date = calculate_next_date(int(line[0:4]), int(line[4:6]), int(line[6:8]))

    GEMS_0245 = find_files_with_keywords_2(GEMS_flist, line, "0245")
    GEMS_0345 = find_files_with_keywords_2(GEMS_flist, line, "0345")
    GEMS_0445 = find_files_with_keywords_2(GEMS_flist, line, "0445")
    GEMS_0545 = find_files_with_keywords_2(GEMS_flist, line, "0545")

    # 处理GEMS文件
    VCD_GEMS_0245 = GEMS_process(GEMS_0245)
    VCD_GEMS_0345 = GEMS_process(GEMS_0345)
    VCD_GEMS_0445 = GEMS_process(GEMS_0445)
    VCD_GEMS_0545 = GEMS_process(GEMS_0545)
    VCD_GEMS = np.column_stack((VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545))

    TROPOMI_file_path = find_files_with_keyword(TROPOMI_list, line)
    VCD_TROPOMI = TROPOMI_process(TROPOMI_file_path)

    pre_TROPOMI = find_files_with_keyword(TROPOMI_list, pre_date)
    pre_VCD_TROPOMI = TROPOMI_process(pre_TROPOMI)

    nex_TROPOMI = find_files_with_keyword(TROPOMI_list, nex_date)
    nex_VCD_TROPOMI = TROPOMI_process(nex_TROPOMI)

    mask = (VCD_TROPOMI != 0)

    VCD_GEMS_0245 = VCD_GEMS_0245[mask]
    VCD_GEMS_0345 = VCD_GEMS_0345[mask]
    VCD_GEMS_0445 = VCD_GEMS_0445[mask]
    VCD_GEMS_0545 = VCD_GEMS_0545[mask]
    VCD_GEMS = np.column_stack((VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545))

    VCD_TROPOMI = VCD_TROPOMI[mask]
    pre_VCD_TROPOMI = pre_VCD_TROPOMI[mask]
    nex_VCD_TROPOMI = nex_VCD_TROPOMI[mask]

    GEOS_file_path = find_files_with_keyword(GEOS_flist, line)
    GEOS_CF = GEOS_process(GEOS_file_path)
    GEOS_CF = GEOS_CF[mask]

    TROPOMI_GEOS = np.column_stack((pre_VCD_TROPOMI, nex_VCD_TROPOMI, GEOS_CF))

    # 添加特征和目标值
    features.append(VCD_GEMS)
    covariates.append(TROPOMI_GEOS)
    target_vector.append(VCD_TROPOMI)

# 转换为特征矩阵和目标向量
features = np.concatenate(features, axis=0)
covariates = np.concatenate(covariates, axis=0)
X = np.column_stack((features, covariates))
target_vector = np.concatenate(target_vector, axis=0)

# 创建 LightGBM 数据集
train_data = lgb.Dataset(X, label=target_vector)

# 定义 LightGBM 参数
params = {
    'objective': 'regression',  # 回归任务
    'metric': 'rmse',           # 均方根误差
    'boosting_type': 'gbdt',    # 梯度提升决策树
    'num_leaves': 31,           # 叶子节点数
    'learning_rate': 0.05,      # 学习率
    'feature_fraction': 0.9,    # 特征采样比例
    'bagging_fraction': 0.8,    # 数据采样比例
    'bagging_freq': 5,          # 每5次迭代进行一次 bagging
    'verbose': 0                # 不打印日志
}

# 训练 LightGBM 模型
num_round = 100  # 迭代次数
bst = lgb.train(params, train_data, num_round)

# 保存模型
output_model = OUTPUT_ROOT / "baselines" / "lightgbm" / "model.txt"
output_model.parent.mkdir(parents=True, exist_ok=True)
bst.save_model(output_model)

# 加载模型进行推理
save_path = OUTPUT_ROOT / "baselines" / "lightgbm" / "predictions"
if not os.path.exists(save_path):
    os.makedirs(save_path)

test_list = SPLIT_ROOT / "all.txt"
with open(test_list, "r") as f:
    test_lines = f.readlines()

# 定义空的特征矩阵和目标向量
features = []
covariates = []

# 提取图像特征
for line in test_lines:
    line = line.rstrip('\n')
    print(line)

    pre_date = calculate_previous_date(int(line[0:4]), int(line[4:6]), int(line[6:8]))
    nex_date = calculate_next_date(int(line[0:4]), int(line[4:6]), int(line[6:8]))

    GEMS_0245 = find_files_with_keywords_2(GEMS_flist, line, "0245")
    GEMS_0345 = find_files_with_keywords_2(GEMS_flist, line, "0345")
    GEMS_0445 = find_files_with_keywords_2(GEMS_flist, line, "0445")
    GEMS_0545 = find_files_with_keywords_2(GEMS_flist, line, "0545")

    # 处理GEMS文件
    VCD_GEMS_0245 = GEMS_process(GEMS_0245)
    VCD_GEMS_0345 = GEMS_process(GEMS_0345)
    VCD_GEMS_0445 = GEMS_process(GEMS_0445)
    VCD_GEMS_0545 = GEMS_process(GEMS_0545)
    VCD_GEMS = np.column_stack((VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545))

    pre_TROPOMI = find_files_with_keyword(TROPOMI_list, pre_date)
    pre_VCD_TROPOMI = TROPOMI_process(pre_TROPOMI)

    nex_TROPOMI = find_files_with_keyword(TROPOMI_list, nex_date)
    nex_VCD_TROPOMI = TROPOMI_process(nex_TROPOMI)

    GEOS_file_path = find_files_with_keyword(GEOS_flist, line)
    GEOS_CF = GEOS_process(GEOS_file_path)
    TROPOMI_GEOS = np.column_stack((pre_VCD_TROPOMI, nex_VCD_TROPOMI, GEOS_CF))

    # 添加特征和目标值
    features.append(VCD_GEMS)
    covariates.append(TROPOMI_GEOS)

# 转换为特征矩阵和目标向量
features = np.concatenate(features, axis=0)
covariates = np.concatenate(covariates, axis=0)
X = np.column_stack((features, covariates))

# 加载模型
bst = lgb.Booster(model_file=output_model)

# 使用模型进行预测
y_pred = bst.predict(X)
y_pred = np.reshape(y_pred, (len(test_lines), 1400, 800))

# 保存预测结果
for i in range(len(test_lines)):
    pred = y_pred[i, :, :]
    pred = pred * 2.5269 + 1.5439
    nc_file = nc.Dataset(os.path.join(save_path, test_lines[i][0:8] + '_LightGBM.nc'), 'w', format='NETCDF4')
    num_rows, num_cols = pred.shape
    nc_file.createDimension('row', num_rows)
    nc_file.createDimension('col', num_cols)
    x_var = nc_file.createVariable('VCD', 'f4', ('row', 'col'))
    x_var[:] = pred
    nc_file.close()
    print(f'{test_lines[i][0:8]}已完成预测')

end_time = time.time()
elapsed_time = int(end_time - start_time)
print(f'一共用时{elapsed_time}秒')
