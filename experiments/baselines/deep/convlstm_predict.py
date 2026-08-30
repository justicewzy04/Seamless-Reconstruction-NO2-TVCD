import netCDF4 as nc
from netCDF4 import Dataset
import torch
import torch.nn.functional as F
import numpy as np
import os
from torch import nn
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from ufill.models.convlstm import ConvLSTM
from ufill.config import CHECKPOINT_ROOT, DATA_ROOT, OUTPUT_ROOT, SPLIT_ROOT, env_path
from datetime import datetime, timedelta


class ConvLSTMPredictor:
    """Load a ConvLSTM checkpoint and run seven-channel tensor inference."""

    def __init__(self, model_path=None, num_classes=1, cuda=True):
        if model_path is None:
            model_path = env_path("UFILL_CONVLSTM_MODEL", CHECKPOINT_ROOT / "convlstm.pth")
        self.cuda = cuda and torch.cuda.is_available()
        self.net = ConvLSTM(num_classes=num_classes, in_channels=7)
        device = torch.device("cuda" if self.cuda else "cpu")
        state_dict = torch.load(model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        if self.cuda:
            self.net = nn.DataParallel(self.net).cuda()
        print(f"Loaded ConvLSTM checkpoint: {model_path}")

    def detect_image(self, image):
        image_data = np.expand_dims(image, 0)
        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            return self.net(images)[0].cpu()

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
        print(f"閿欒锛氭枃浠朵笉瀛樺湪: {keyword1},{keyword2}")
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
        #灏咷EMS鍜孴ROPOMI鏁版嵁涓ぇ浜?00鐨勫紓甯稿€煎幓鎺?
        VCD_TROPOMI [VCD_TROPOMI > 100] = np.nan
        VCD_TROPOMI      = torch.from_numpy(np.array(VCD_TROPOMI)).type(torch.FloatTensor)
        #褰掍竴鍖?
        VCD_TROPOMI=torch.div(torch.sub(VCD_TROPOMI,1.5490),2.5403)
        #鎶奊EMS鍜孴ROPOMI閲岀殑nan鍊兼浛鎹负0
        VCD_TROPOMI= torch.where(torch.isnan(VCD_TROPOMI),torch.tensor(0.0),VCD_TROPOMI)
        #鎶奫1400,800]鍙樻垚[1,1400,800]  
        VCD_TROPOMI= VCD_TROPOMI[np.newaxis,:,:]
        #灏嗗奖鍍忎粠[1锛?400,800]濉厖鑷砙1,1408,800]锛屼娇鍏跺湪鍗风Н鏃朵笉浼氬彂鐢熼敊璇?
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
        print(f"閿欒锛氭枃浠朵笉瀛樺湪: {keyword}")
    return matching_files
def space_process(space):
    space     = torch.from_numpy(np.array(space)).type(torch.FloatTensor)
    # space= space[np.newaxis,:,:]
    space=F.pad(space, (0, 0,4,4))
    return space
def is_leap_year(year):
    """Return whether year is a leap year."""
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
def convert_date_to_day(year, month, day):
    """Convert a calendar date to year plus day-of-year."""
    # 姣忎釜鏈堢殑澶╂暟锛岃€冭檻骞冲勾
    months_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    # 濡傛灉鏄棸骞达紝浜屾湀涓?9澶?
    if is_leap_year(year):
        months_days[1] = 29
    
    # 璁＄畻缁欏畾鏃ユ湡鏄勾涓殑绗嚑澶?
    day_of_year = sum(months_days[:month - 1]) + day
    
    return f"{year}{day_of_year:03d}"

def safe_divide(year, month, day,DOY):
    day_value = int(convert_date_to_day(year, month, day)[-3:])
    denominator = int(DOY) - day_value
    if denominator == 0:
        return 1  # 鍒嗘瘝涓?鏃惰繑鍥?
    return 1 / denominator
def calculate_previous_date(year, month, day):
    """Return the previous date in YYYYMMDD format."""
    given_date = datetime(year, month, day)
    previous_date = given_date - timedelta(days=1)
    return previous_date.strftime("%Y%m%d")

def calculate_next_date(year, month, day):
    """Return the next date in YYYYMMDD format."""
    given_date = datetime(year, month, day)
    next_date = given_date + timedelta(days=1)
    return next_date.strftime("%Y%m%d")

def day_process(day):
    DOY=day[-3:]
    D1 = np.abs(safe_divide(int(day[0:4]), 3, 21,DOY))  # 璁＄畻 D1锛岀81澶?
    D2 = np.abs(safe_divide(int(day[0:4]), 6, 21,DOY)) # 璁＄畻 D2锛岀173澶?
    D3 = np.abs(safe_divide(int(day[0:4]), 9, 22,DOY))  # 璁＄畻 D3锛岀266澶?
    D4 = np.abs(safe_divide(int(day[0:4]), 12, 22,DOY))  # 璁＄畻 D4锛岀357澶?


    DOY = int(DOY)
    
    #褰掍竴鍖?
    DOY=(DOY-1)/(365-1)
    D1=(D1-1/284)/(1-1/284)#鏈€澶ф棩鏈熼棿闅?84澶?
    D2=(D2-1/192)/(1-1/192)#鏈€澶ф棩鏈熼棿闅?92澶?
    D3=(D3-1/265)/(1-1/265)#鏈€澶ф棩鏈熼棿闅?65澶?
    D4=(D4-1/356)/(1-1/356)#鏈€澶ф棩鏈熼棿闅?56澶?

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

    # 閫愭湀绾挎€ц皟鏁村悗鐨凣EOS-CF鏁版嵁
    GEOS_path = env_path("UFILL_GEOS_ROOT", DATA_ROOT / "GEOS-CF_VM_daily")
    TROPOMI_path = env_path("UFILL_TROPOMI_ROOT", DATA_ROOT / "TROPOMI" / "daily")

    # 淇濆瓨璺緞
    dir_save_path = OUTPUT_ROOT / "baselines" / "convlstm"
    file_list = SPLIT_ROOT / "all.txt"
    with open(file_list,"r") as f:
        train_lines = f.readlines()
    
    #-------------------------------------------------------------------------#
    #   simplify            浣跨敤Simplify onnx
    #   onnx_save_path      鎸囧畾浜唎nnx鐨勪繚瀛樿矾寰?
    #-------------------------------------------------------------------------#
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"
    GEMS_flist = find_nc_files(dir_origin_path)
    GEOS_flist = find_npy_files(GEOS_path)
    TROPOMI_flist = find_nc_files(TROPOMI_path)
    model = ConvLSTMPredictor()
    # ps_path=os.path.join(relative_path,'data/air/space/distance/space_distance_normal.npy')
    # ps=np.load(ps_path)
    # ps=space_process(ps)
    # img_names = os.listdir(dir_origin_path)
    for annotation_line in train_lines:
        # if img_name.lower().endswith(('.nc')): 
            annotation_line = annotation_line[0:8]
            day=convert_date_to_day(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            pre_date=calculate_previous_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            pre_day=convert_date_to_day(int(pre_date[0:4]),int(pre_date[4:6]),int(pre_date[6:8]))
            nex_date=calculate_next_date(int(annotation_line[0:4]),int(annotation_line[4:6]),int(annotation_line[6:8]))
            nex_day=convert_date_to_day(int(nex_date[0:4]),int(nex_date[4:6]),int(nex_date[6:8]))
            # day_wei=day_process(day)
            # pre_day_wei=day_process(pre_day)
            # nex_day_wei=day_process(nex_day)

            #澶勭悊GEMS鏂囦欢
            VCD_GEMS_0245 = GEMS_process(GEMS_flist,annotation_line,"0245")
            VCD_GEMS_0345 = GEMS_process(GEMS_flist,annotation_line,"0345")
            VCD_GEMS_0445 = GEMS_process(GEMS_flist,annotation_line,"0445")
            VCD_GEMS_0545 = GEMS_process(GEMS_flist,annotation_line,"0545")

            GEOS = find_files_with_keyword(GEOS_flist,annotation_line)
            GEOS_CF = GEOS_process(GEOS)

            pre_VCD_TROPOMI= TROPOMI_process(TROPOMI_flist,pre_date)
            nex_VCD_TROPOMI= TROPOMI_process(TROPOMI_flist,nex_date)


            GEMS_GEOS_TROPOMI = torch.cat([VCD_GEMS_0245, VCD_GEMS_0345, VCD_GEMS_0445, VCD_GEMS_0545,GEOS_CF,pre_VCD_TROPOMI,nex_VCD_TROPOMI], dim=0)
        

            #鍒╃敤璁粌濂界殑妯″瀷杩涜棰勬祴
            VCD_TROPOMI     = model.detect_image(GEMS_GEOS_TROPOMI)
            
            #----灏嗛娴嬪ソ鐨刅CD_TROPOMI瀛樻垚nc鏂囦欢
            if not os.path.exists(dir_save_path):
                os.makedirs(dir_save_path)
            VCD_TROPOMI = VCD_TROPOMI[:,4:-4,:]
            VCD_TROPOMI=VCD_TROPOMI.squeeze()
            VCD_TROPOMI=torch.add(torch.mul(VCD_TROPOMI,2.5403),1.5490) 
            nc_file = Dataset(os.path.join(dir_save_path, annotation_line+'_ConvLSTM.nc'), 'w', format='NETCDF4')
            num_rows, num_cols = VCD_TROPOMI.size()
            nc_file.createDimension('row', num_rows)
            nc_file.createDimension('col', num_cols)
            x_var = nc_file.createVariable('VCD', 'f4', ('row', 'col'))
            x_var[:] = VCD_TROPOMI.numpy()
            nc_file.close()
            print(f'{annotation_line} processed')
print('finish')





