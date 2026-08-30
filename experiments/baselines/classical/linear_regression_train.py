import numpy as np
from sklearn.linear_model import LinearRegression
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ufill.utils.files import find_nc_files, find_npy_files, find_files_with_keyword
from ufill.config import DATA_ROOT, OUTPUT_ROOT, SPLIT_ROOT, env_path
import netCDF4 as nc
import pickle
import time
from datetime import datetime, timedelta

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
    VCD_GEMS=GEMS.variables['VCD'][:]
    VCD_GEMS [VCD_GEMS > 100] = np.nan
    VCD_GEMS= np.divide(np.subtract(VCD_GEMS, 2.3492), 4.0094)
    VCD_GEMS[np.isnan(VCD_GEMS)] = 0  
    # VCD_GEMS = np.array(VCD_GEMS)  
    VCD_GEMS= VCD_GEMS.flatten()    
    # VCD_GEMS= VCD_GEMS[np.newaxis,:] 
    return VCD_GEMS
def TROPOMI_process(TROPOMI_file_path):

    TROPOMI      = nc.Dataset(TROPOMI_file_path[0])
    # TROPOMI_name   =  os.path.basename(img_path)
    VCD_TROPOMI=TROPOMI.variables['VCD'][:]
    #将TROPOMI数据中大于100的异常值去掉
    VCD_TROPOMI [VCD_TROPOMI > 100] = np.nan
    VCD_TROPOMI=np.divide(np.subtract(VCD_TROPOMI, 1.5439), 2.5269)
    VCD_TROPOMI[np.isnan(VCD_TROPOMI)] = 0    
    VCD_TROPOMI= VCD_TROPOMI.flatten()
    # VCD_TROPOMI =VCD_TROPOMI[np.newaxis,: ]

    return VCD_TROPOMI
def GEOS_process(GEOS_file_path):
    
    GEOS = np.load(GEOS_file_path[0])
    GEOS = GEOS[0:1400,0:800]
    GEOS=np.divide(np.subtract(GEOS, 2.0366), 4.1030)
    GEOS_CF= GEOS.flatten()

    return GEOS_CF
def is_leap_year(year):
    """判断是否为闰年"""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
def convert_date_to_day(year, month, day):
    """将年月日转换成年份和年中的第几天"""
    # 每个月的天数，考虑平年
    months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 如果是闰年，二月为29天
    if is_leap_year(year):
        months_days[1] = 29
    
    # 计算给定日期是年中的第几天
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


start_time=time.time()
# 加载图像数据和目标值
GEMS_path = env_path("UFILL_GEMS_ROOT", DATA_ROOT / "GEMS" / "hourly")
TROPOMI_path = env_path("UFILL_TROPOMI_ROOT", DATA_ROOT / "TROPOMI" / "daily")
GEOS_path = env_path("UFILL_GEOS_ROOT", DATA_ROOT / "GEOS-CF_VM_daily")
GEMS_flist= find_nc_files(GEMS_path)
TROPOMI_list= find_nc_files(TROPOMI_path)
GEOS_flist= find_npy_files(GEOS_path)
train_list = SPLIT_ROOT / "train.txt"
with open(train_list,"r") as f:
    train_lines = f.readlines()
# 定义空的特征矩阵和目标向量
features = []
covariates=[]
target_vector = []

# 提取图像特征
for line in train_lines:
    line  = line.rstrip('\n')
    print(line)

    pre_date=calculate_previous_date(int(line[0:4]),int(line[4:6]),int(line[6:8]))
    nex_date=calculate_next_date(int(line[0:4]),int(line[4:6]),int(line[6:8]))



    GEMS_0245 = find_files_with_keywords_2(GEMS_flist,line,"0245")
    GEMS_0345 = find_files_with_keywords_2(GEMS_flist,line,"0345")
    GEMS_0445 = find_files_with_keywords_2(GEMS_flist,line,"0445")
    GEMS_0545 = find_files_with_keywords_2(GEMS_flist,line,"0545")
    #处理GEMS文件
    VCD_GEMS_0245 = GEMS_process(GEMS_0245)
    VCD_GEMS_0345 = GEMS_process(GEMS_0345)
    VCD_GEMS_0445 = GEMS_process(GEMS_0445)
    VCD_GEMS_0545 = GEMS_process(GEMS_0545)
    # VCD_GEMS=np.column_stack((VCD_GEMS_0245, VCD_GEMS_0345,VCD_GEMS_0445,VCD_GEMS_0545))


    TROPOMI_file_path = find_files_with_keyword(TROPOMI_list,line )
    VCD_TROPOMI=TROPOMI_process(TROPOMI_file_path)

    pre_TROPOMI = find_files_with_keyword(TROPOMI_list,pre_date)
    pre_VCD_TROPOMI= TROPOMI_process(pre_TROPOMI)

    nex_TROPOMI = find_files_with_keyword(TROPOMI_list,nex_date)
    nex_VCD_TROPOMI= TROPOMI_process(nex_TROPOMI)


    mask =  (VCD_TROPOMI != 0)

    VCD_GEMS_0245 = VCD_GEMS_0245[mask]
    VCD_GEMS_0345 = VCD_GEMS_0345[mask]
    VCD_GEMS_0445 = VCD_GEMS_0445[mask]
    VCD_GEMS_0545 = VCD_GEMS_0545[mask]
    VCD_GEMS=np.column_stack((VCD_GEMS_0245, VCD_GEMS_0345,VCD_GEMS_0445,VCD_GEMS_0545))

    VCD_TROPOMI = VCD_TROPOMI[mask]
    pre_VCD_TROPOMI = pre_VCD_TROPOMI[mask]
    nex_VCD_TROPOMI = nex_VCD_TROPOMI[mask]


    GEOS_file_path = find_files_with_keyword(GEOS_flist,line)
    GEOS_CF = GEOS_process(GEOS_file_path)
    GEOS_CF = GEOS_CF[mask]

    TROPOMI_GEOS=np.column_stack((pre_VCD_TROPOMI,nex_VCD_TROPOMI,GEOS_CF))

    

    # 添加特征和目标值
    features.append(VCD_GEMS)
    covariates.append(TROPOMI_GEOS)
    target_vector.append(VCD_TROPOMI)


# 转换为特征矩阵和目标向量
features = np.concatenate(features, axis=0)
covariates = np.concatenate(covariates, axis=0)
X = np.column_stack((features, covariates))
target_vector = np.concatenate(target_vector, axis=0)



# 实例化线性回归模型
model = LinearRegression()

# 拟合模型
model.fit(X, target_vector)
# 保存模型
output_model = OUTPUT_ROOT / "baselines" / "linear_regression" / "model.pkl"
output_model.parent.mkdir(parents=True, exist_ok=True)
with open(output_model, 'wb') as f:
    pickle.dump(model, f)

end_time=time.time()
elapsed_time=int(end_time-start_time)
print(f'一共花了{elapsed_time}秒')
