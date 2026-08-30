import netCDF4 as nc
from netCDF4 import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ufill.config import DATA_ROOT, OUTPUT_ROOT, SPLIT_ROOT, env_path
from ufill.inference.unet_predictor import Unet

def find_nc_files(folder_path):
    nc_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    return nc_files


def find_files_with_keywords_2(file_list, keyword1, keyword2):
    matching_files = []
    for file_name in file_list:
        if keyword1 in file_name and keyword2 in file_name:
            matching_files.append(file_name)
    if len(matching_files) == 0:
        print(f"错误：文件不存在: {keyword1},{keyword2}")
        # sys.exit(1)
    return matching_files


def GEMS_process(file_list, keyword1, keyword2):
    matching_files = []
    for file_name in file_list:
        if keyword1 in file_name and keyword2 in file_name:
            matching_files.append(file_name)

    if len(matching_files) != 0:    
        GEMS = nc.Dataset(matching_files[0])
        VCD_GEMS=GEMS.variables['VCD'][:]
        VCD_GEMS [VCD_GEMS > 100] = np.nan
        VCD_GEMS      = torch.from_numpy(np.array(VCD_GEMS)).type(torch.FloatTensor)
        VCD_GEMS=torch.div(torch.sub(VCD_GEMS,2.3587),4.0309)
        VCD_GEMS= torch.where(torch.isnan(VCD_GEMS),torch.tensor(0.0),VCD_GEMS)
        VCD_GEMS= VCD_GEMS[np.newaxis,:,:] 
        VCD_GEMS=F.pad(VCD_GEMS, (0, 0,4,4))

    if len(matching_files) == 0:
        VCD_GEMS = torch.zeros(1,1408, 800)

    return VCD_GEMS


def find_npy_files(folder_path):
    npy_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files
def GEOS_process(GEOS):
    GEOS = np.load(GEOS[0])
    GEOS = GEOS[0:1400,0:800]
    GEOS_CF      = torch.from_numpy(np.array(GEOS)).type(torch.FloatTensor)
    GEOS_CF=torch.div(torch.sub(GEOS_CF,2.0399),4.1086)
    GEOS_CF= GEOS_CF[np.newaxis,:,:] 
    GEOS_CF=F.pad(GEOS_CF, (0, 0,4,4))
    return GEOS_CF

def TROPOMI_process(file_list, keyword):
    matching_files = []
    for file_name in file_list:
        if keyword in file_name:
            matching_files.append(file_name)

    if len(matching_files) != 0:  
        TROPOMI = nc.Dataset(matching_files[0])
        VCD_TROPOMI=TROPOMI.variables['VCD'][:]
        #将GEMS和TROPOMI数据中大于100的异常值去掉
        VCD_TROPOMI [VCD_TROPOMI > 100] = np.nan
        VCD_TROPOMI      = torch.from_numpy(np.array(VCD_TROPOMI)).type(torch.FloatTensor)
        #归一化
        VCD_TROPOMI=torch.div(torch.sub(VCD_TROPOMI,1.5490),2.5403)
        #把GEMS和TROPOMI里的nan值替换为0
        VCD_TROPOMI= torch.where(torch.isnan(VCD_TROPOMI),torch.tensor(0.0),VCD_TROPOMI)
        #把[1400,800]变成[1,1400,800]  
        VCD_TROPOMI= VCD_TROPOMI[np.newaxis,:,:]
        #将影像从[1，1400,800]填充至[1,1408,800]，使其在卷积时不会发生错误
        VCD_TROPOMI= F.pad(VCD_TROPOMI, (0, 0,4,4)) 
    if len(matching_files) == 0:
        VCD_TROPOMI= torch.zeros(1,1408, 800)

    return VCD_TROPOMI

def find_files_with_keyword(file_list, keyword):
    matching_files = []
    for file_name in file_list:
        if keyword in file_name:
            matching_files.append(file_name)
    if len(matching_files) == 0:
        print(f"错误：文件不存在: {keyword}")
    return matching_files
def space_process(space):
    space     = torch.from_numpy(np.array(space)).type(torch.FloatTensor)
    # space= space[np.newaxis,:,:]
    space=F.pad(space, (0, 0,4,4))
    return space
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

def safe_divide(year, month, day,DOY):
    day_value = int(convert_date_to_day(year, month, day)[-3:])
    denominator = int(DOY) - day_value
    if denominator == 0:
        return 1  # 分母为0时返回1
    return 1 / denominator
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

def day_process(day):
    DOY=day[-3:]
    D1 = np.abs(safe_divide(int(day[0:4]), 3, 21,DOY))  # 计算 D1，第81天
    D2 = np.abs(safe_divide(int(day[0:4]), 6, 21,DOY)) # 计算 D2，第173天
    D3 = np.abs(safe_divide(int(day[0:4]), 9, 22,DOY))  # 计算 D3，第266天
    D4 = np.abs(safe_divide(int(day[0:4]), 12, 22,DOY))  # 计算 D4，第357天


    DOY = int(DOY)
    
    #归一化
    DOY=(DOY-1)/(365-1)
    D1=(D1-1/284)/(1-1/284)#最大日期间隔284天
    D2=(D2-1/192)/(1-1/192)#最大日期间隔192天
    D3=(D3-1/265)/(1-1/265)#最大日期间隔265天
    D4=(D4-1/356)/(1-1/356)#最大日期间隔356天

    DOY = torch.full((1408, 800), DOY, dtype=torch.int64)
    D1 = torch.full((1408, 800), D1, dtype=torch.int64)
    D2 = torch.full((1408, 800), D2, dtype=torch.int64)
    D3 = torch.full((1408, 800), D3, dtype=torch.int64)
    D4 = torch.full((1408, 800), D4, dtype=torch.int64)
    day_wei=torch.stack([DOY, D1, D2,D3,D4], dim=0)
    return day_wei

if __name__ == "__main__":
    mode = "dir_predict"
    # name_classes
    #-------------------------------------------------------------------------#
     
    # GEMS hourly
    dir_origin_path = env_path("UFILL_GEMS_ROOT", DATA_ROOT / "GEMS" / "hourly")
    #GEOS_path = os.path.join(relative_path,'data/air/GEOS_V2.1')

    # 逐月线性调整后的GEOS-CF数据
    GEOS_path = env_path("UFILL_GEOS_ROOT", DATA_ROOT / "GEOS-CF_VM_daily")
    TROPOMI_path = env_path("UFILL_TROPOMI_ROOT", DATA_ROOT / "TROPOMI" / "daily")

    # 保存路径
    dir_save_path = OUTPUT_ROOT / "predictions" / "geo_unet"
    file_list = env_path("UFILL_PREDICT_SPLIT", SPLIT_ROOT / "test.txt")
    with open(file_list,"r") as f:
        train_lines = f.readlines()
    
    #-------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    GEMS_flist = find_nc_files(dir_origin_path)
    GEOS_flist = find_npy_files(GEOS_path)
    TROPOMI_flist = find_nc_files(TROPOMI_path)
    unet = Unet()
    ps_path = env_path(
        "UFILL_SPACE_FEATURE",
        DATA_ROOT / "space" / "distance" / "space_distance_normal.npy",
    )
    ps=np.load(ps_path)
    ps=space_process(ps)
    # img_names = os.listdir(dir_origin_path)
    for annotation_line in train_lines:
        # if img_name.lower().endswith(('.nc')): 
            annotation_line = annotation_line[0:8]
            day=convert_date_to_day(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            pre_date=calculate_previous_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            pre_day=convert_date_to_day(int(pre_date[0:4]),int(pre_date[4:6]),int(pre_date[6:8]))
            nex_date=calculate_next_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            nex_day=convert_date_to_day(int(nex_date[0:4]),int(nex_date[4:6]),int(nex_date[6:8]))
            day_wei=day_process(day)
            pre_day_wei=day_process(pre_day)
            nex_day_wei=day_process(nex_day)

            #处理GEMS文件
            VCD_GEMS_0245 = GEMS_process(GEMS_flist,annotation_line,"0245")
            VCD_GEMS_0345 = GEMS_process(GEMS_flist,annotation_line,"0345")
            VCD_GEMS_0445 = GEMS_process(GEMS_flist,annotation_line,"0445")
            VCD_GEMS_0545 = GEMS_process(GEMS_flist,annotation_line,"0545")

            GEOS = find_files_with_keyword(GEOS_flist,annotation_line)
            GEOS_CF = GEOS_process(GEOS)

            pre_VCD_TROPOMI= TROPOMI_process(TROPOMI_flist,pre_date)
            nex_VCD_TROPOMI= TROPOMI_process(TROPOMI_flist,nex_date)


            GEMS_GEOS_space_date_TROPOMI = torch.cat([VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545,GEOS_CF,ps,day_wei,pre_VCD_TROPOMI,ps,pre_day_wei,nex_VCD_TROPOMI,ps,nex_day_wei], dim=0)
        

            #利用训练好的模型进行预测
            VCD_TROPOMI     = unet.detect_image(GEMS_GEOS_space_date_TROPOMI)
            
            #----将预测好的VCD_TROPOMI存成nc文件
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            VCD_TROPOMI = VCD_TROPOMI[:,4:-4,:]
            VCD_TROPOMI=VCD_TROPOMI.squeeze()
            VCD_TROPOMI=torch.add(torch.mul(VCD_TROPOMI,2.5403),1.5490) 
            nc_file = Dataset(os.path.join(dir_save_path, 'POMINO-GEMS_'+annotation_line+'_VT_geo_0926.nc'), 'w', format='NETCDF4')
            num_rows, num_cols = VCD_TROPOMI.size()
            nc_file.createDimension('row', num_rows)
            nc_file.createDimension('col', num_cols)
            x_var = nc_file.createVariable('data', 'f4', ('row', 'col'))
            x_var[:] = VCD_TROPOMI.numpy()
            nc_file.close()
            print(f'{annotation_line}已处理完成')
print('finish')
