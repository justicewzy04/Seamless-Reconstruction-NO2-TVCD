import os
import netCDF4 as nc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from ufill.config import DATA_ROOT, env_path
from datetime import datetime, timedelta

class APDataset(Dataset):
    def __init__(self, annotation_lines, GEMS_path, TROPOMI_path, GEOS_path):
        super(APDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.GEMS_flist=find_nc_files(GEMS_path)
        self.TROPOMI_flist=find_nc_files(TROPOMI_path)
        self.GEOS_flist=find_npy_files(GEOS_path)

    def __len__(self):
        return len(self.annotation_lines) 
 
    def __getitem__(self, index):
        # print(index)
        #选择对应样本的日期
        annotation_line = self.annotation_lines[index % len(self.TROPOMI_flist)][0:8] 
        # print(annotation_line)
        day=convert_date_to_day(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
        pre_date=calculate_previous_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
        pre_day=convert_date_to_day(int(pre_date[0:4]),int(pre_date[4:6]),int(pre_date[6:8]))
        nex_date=calculate_next_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
        nex_day=convert_date_to_day(int(nex_date[0:4]),int(nex_date[4:6]),int(nex_date[6:8]))

        day_wei=day_process(day)
        pre_day_wei=day_process(pre_day)
        nex_day_wei=day_process(nex_day)
        #找到对应文件
        # GEMS = find_files_with_keywords_2(self.GEMS_flist,annotation_line,"0245")  
        GEMS_0245 = find_files_with_keywords_2(self.GEMS_flist,annotation_line,"0245")
        GEMS_0345 = find_files_with_keywords_2(self.GEMS_flist,annotation_line,"0345")
        GEMS_0445 = find_files_with_keywords_2(self.GEMS_flist,annotation_line,"0445")
        GEMS_0545 = find_files_with_keywords_2(self.GEMS_flist,annotation_line,"0545")
        #处理GEMS文件
        VCD_GEMS_0245 = GEMS_process(GEMS_0245)
        VCD_GEMS_0345 = GEMS_process(GEMS_0345)
        VCD_GEMS_0445 = GEMS_process(GEMS_0445)
        VCD_GEMS_0545 = GEMS_process(GEMS_0545)

        GEOS = find_files_with_keyword(self.GEOS_flist,annotation_line)
        GEOS_CF = GEOS_process(GEOS)

        #找到TROPOMI文件
        TROPOMI = find_files_with_keyword(self.TROPOMI_flist,annotation_line)
        #处理TROPOMI文件
        VCD_TROPOMI= TROPOMI_process(TROPOMI)

        pre_TROPOMI = find_files_with_keyword(self.TROPOMI_flist,pre_date)
        pre_VCD_TROPOMI= TROPOMI_process(pre_TROPOMI)

        nex_TROPOMI = find_files_with_keyword(self.TROPOMI_flist,nex_date)
        nex_VCD_TROPOMI= TROPOMI_process(nex_TROPOMI)


        ps_path = env_path(
            "UFILL_SPACE_FEATURE",
            DATA_ROOT / "space" / "distance" / "space_distance_normal.npy",
        )
        ps=np.load(ps_path)
        ps=space_process(ps)
        flip_flag = torch.randint(0, 2, (1,))
        if flip_flag == 1:
            # 定义掩膜的大小
            mask_size = (192, 192)
            #生成随机掩膜
            _, height, width = VCD_GEMS_0245.size()
            # 创建与原始张量相同大小的真值掩膜
            mask = torch.ones((height, width))
            # 随机生成掩膜区域的左上角坐标
            x = np.random.randint(256, 440)
            y = np.random.randint(768, 1080)
            # 根据掩膜大小在掩膜区域内填充掩膜值
            mask[y:y+mask_size[0], x:x+mask_size[1]] = 0
            # 将掩膜应用于原始张量
            VCD_GEMS_0245 = VCD_GEMS_0245 * mask
            VCD_GEMS_0345 = VCD_GEMS_0345 * mask
            VCD_GEMS_0445 = VCD_GEMS_0445 * mask
            VCD_GEMS_0545 = VCD_GEMS_0545 * mask


        flip_flag = torch.randint(0, 2, (1,))
        if flip_flag == 1:
            # 定义掩膜的大小
            mask_size = (192, 192)
            #生成随机掩膜
            _, height, width = pre_VCD_TROPOMI.size()
            # 创建与原始张量相同大小的真值掩膜
            mask = torch.ones((height, width))
            # 随机生成掩膜区域的左上角坐标
            x = np.random.randint(256, 440)
            y = np.random.randint(768, 1080)
            # 根据掩膜大小在掩膜区域内填充掩膜值
            mask[y:y+mask_size[0], x:x+mask_size[1]] = 0
            # 将掩膜应用于原始张量
            pre_VCD_TROPOMI = pre_VCD_TROPOMI * mask
            nex_VCD_TROPOMI = nex_VCD_TROPOMI * mask
        GEMS_GEOS_space_date_TROPOMI = torch.cat([VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545,GEOS_CF,ps,day_wei,pre_VCD_TROPOMI,ps,pre_day_wei,nex_VCD_TROPOMI,ps,nex_day_wei], dim=0)
        return GEMS_GEOS_space_date_TROPOMI, VCD_TROPOMI
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

def find_nc_files(folder_path):
    nc_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    return nc_files
def find_npy_files(folder_path):
    npy_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                npy_files.append(os.path.join(root, file))
    return npy_files
def find_files_with_keywords_2(file_list, keyword1, keyword2):
    matching_files = []
    for file_name in file_list:
        if keyword1 in file_name and keyword2 in file_name:
            matching_files.append(file_name)
    if len(matching_files) == 0:
        print(f"错误：文件不存在: {keyword1},{keyword2}")
        # sys.exit(1)
    return matching_files

def find_files_with_keyword(file_list, keyword):
    matching_files = []
    for file_name in file_list:
        if keyword in file_name:
            matching_files.append(file_name)
    if len(matching_files) == 0:
        print(f"错误：文件不存在: {keyword}")
    return matching_files

def GEMS_process(GEMS):
    GEMS = nc.Dataset(GEMS[0])
    VCD_GEMS=GEMS.variables['VCD'][:]
    VCD_GEMS [VCD_GEMS > 100] = np.nan
    VCD_GEMS      = torch.from_numpy(np.array(VCD_GEMS)).type(torch.FloatTensor)
    VCD_GEMS=torch.div(torch.sub(VCD_GEMS,2.3492),4.0094)
    VCD_GEMS= torch.where(torch.isnan(VCD_GEMS),torch.tensor(0.0),VCD_GEMS)
    VCD_GEMS= VCD_GEMS[np.newaxis,:,:] 
    VCD_GEMS=F.pad(VCD_GEMS, (0, 0,4,4), mode='reflect')
    return VCD_GEMS
def GEOS_process(GEOS):
    GEOS = np.load(GEOS[0])
    GEOS = GEOS[0:1400,0:800]
    GEOS_CF      = torch.from_numpy(np.array(GEOS)).type(torch.FloatTensor)
    GEOS_CF=torch.div(torch.sub(GEOS_CF,1.5464),3.3543)
    GEOS_CF= GEOS_CF[np.newaxis,:,:] 
    GEOS_CF=F.pad(GEOS_CF, (0, 0,4,4), mode='reflect')
    return GEOS_CF
def TROPOMI_process(TROPOMI):
    TROPOMI = nc.Dataset(TROPOMI[0])
    VCD_TROPOMI=TROPOMI.variables['VCD'][:]
    #将GEMS和TROPOMI数据中大于100的异常值去掉
    VCD_TROPOMI [VCD_TROPOMI > 100] = np.nan
    VCD_TROPOMI      = torch.from_numpy(np.array(VCD_TROPOMI)).type(torch.FloatTensor)
    #归一化
    VCD_TROPOMI=torch.div(torch.sub(VCD_TROPOMI,1.5439),2.5269)
    #把GEMS和TROPOMI里的nan值替换为0
    VCD_TROPOMI= torch.where(torch.isnan(VCD_TROPOMI),torch.tensor(0.0),VCD_TROPOMI)
    #把[1400,800]变成[1,1400,800]  
    VCD_TROPOMI= VCD_TROPOMI[np.newaxis,:,:]
    #将影像从[1，1400,800]填充至[1,1408,800]，使其在卷积时不会发生错误
    VCD_TROPOMI= F.pad(VCD_TROPOMI, (0, 0,4,4), mode='reflect') 
    return VCD_TROPOMI
def space_process(space):
    space     = torch.from_numpy(np.array(space)).type(torch.FloatTensor)
    space= F.pad(space, (0, 0,4,4)) 
    return space
