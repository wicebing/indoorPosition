import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json, pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, to_rgba

import utils

databank_filepath = "./guider20240808/databank/positions"
os.makedirs(databank_filepath,exist_ok=True)

local_timezone = pytz.timezone('Asia/Taipei')  

select_beacons =['N002', 'N003', 'N004', 'N005', 'N006', 'N007', 'N008', 'N017', 'N029']
beacon_ids = select_beacons #utils.get_beacons()
print('=== load beacons ids ===')

x_min=302491
x_max=302516
y_min=2770397
y_max=2770422
scale = 45
grid_size = 45
txyzPds = {}
txyzPds_origin = {}
txyzOutlier = {}

def detect_and_label_outliers(df, window='3s'):
    dfc = df.copy()  # Avoid modifying the original DataFrame

    dfc['x_avg'] = dfc['x'].shift(1).rolling(window).mean()
    dfc['y_avg'] = dfc['y'].shift(1).rolling(window).mean()

    dfc['x_m1'] = dfc['x'].shift(1)
    dfc['y_m1'] = dfc['y'].shift(1)
    
    dfc['time_diff'] = dfc['timesss'].diff().dt.total_seconds()

    dfc['group']=0
    dfc.loc[((dfc['x']-dfc['x_m1']).abs()>5) | ((dfc['y']-dfc['y_m1']).abs()>5), 'group'] +=1
    dfc.loc[(dfc['time_diff'] > 90), 'group'] +=1
    dfc.loc[(((dfc['x']-dfc['x_avg']).abs()>5) | ((dfc['y']-dfc['y_avg']).abs()>5)) & (dfc['time_diff'] < 60), 'fly'] = True
    
    dfc['fly'] = dfc['fly'].fillna(False) # handle cases where no outlier was detected.
    dfc['group'] = dfc['group'].cumsum()
    return dfc

for beacon in beacon_ids:
    recordName= f'{beacon}.pkl'
    pickle_filepath = os.path.join(databank_filepath,recordName)
    
    if os.path.isfile(pickle_filepath):
        print(f'=== {beacon}.pkl exist, loading ===')

        txyzPd_origin = pd.read_pickle(pickle_filepath)
        pd_pz = pd.json_normalize(txyzPd_origin['position'])
        pd_time = pd.to_datetime(txyzPd_origin['positionTime'],format='mixed').dt.tz_convert(local_timezone)
        df = pd.concat([pd_time,pd_pz],axis=1)
        
        df['x'] = df['x']-x_min
        df['y'] = df['y']-y_min
        
        txyzPds_origin[beacon]=df
        
        aao = df.dropna().copy().set_index('positionTime')
        aao['timesss'] = aao.index
        txyzOutlier[beacon] = {'origin':len(aao),'outlier':0}
        outliers = 1
        while(outliers>0):
            aa = detect_and_label_outliers(aao, window='3s')
            group_count = aa.value_counts('group')
            aa['group_num'] = group_count[aa.group].values
            drop = ((aa.fly) & (aa['group_num']<=3)) | (aa['group_num']<=2)
            aao = aao.loc[~drop]
            print(len(aa)-len(aao),len(aa),len(aao))
            outliers = len(aa)-len(aao)
            txyzOutlier[beacon]['outlier'] += outliers
        
        aa = detect_and_label_outliers(aao, window='3s')
        group_x = aa.groupby('group')['x'].mean()
        group_y = aa.groupby('group')['y'].mean()
        group_lapse = aa.groupby('group')['time_diff'].sum()
        group_count = aa.value_counts('group')
        
        fly_group = aa[aa.fly]
        
        drop_group = []
        
        for dgp in list(set(fly_group['group'])):
            now_x = group_x[int(dgp)]
            now_y = group_y[int(dgp)]
            if now_x < -2 or now_x >27:
                drop_group.append(dgp)
            if now_y < -2 or now_y >27:
                drop_group.append(dgp)
                
            try:
                if (abs(group_x[int(dgp-1)]-group_x[int(dgp+1)])<5) & \
                    (abs(now_x-group_x[int(dgp-1)])>5) & \
                    (group_count[int(dgp)]<(group_count[int(dgp-1)])) & \
                     (group_count[int(dgp)]<30) & \
                      (group_lapse[int(dgp)]<150):
                          drop_group.append(dgp)
                if (abs(group_y[int(dgp-1)]-group_y[int(dgp+1)])<5) & \
                    (abs(now_y-group_y[int(dgp-1)])>5)& \
                     (group_count[int(dgp)]<30) & \
                      (group_lapse[int(dgp)]<150):
                          drop_group.append(dgp)
                if (abs(now_x-group_x[int(dgp-1)])>5) & \
                    (group_count[int(dgp)]<(group_count[int(dgp-1)])) & \
                    (group_count[int(dgp)]<(group_count[int(dgp+1)])) & \
                     (group_count[int(dgp)]<60) & \
                      (group_lapse[int(dgp)]<120):
                          drop_group.append(dgp)
                if (abs(now_y-group_y[int(dgp-1)])>5)& \
                    (group_count[int(dgp)]<(group_count[int(dgp-1)])) & \
                    (group_count[int(dgp)]<(group_count[int(dgp+1)])) & \
                     (group_count[int(dgp)]<60) & \
                      (group_lapse[int(dgp)]<120):
                          drop_group.append(dgp)                          
            except:
                pass
        drop_group=list(set(drop_group))
        
        for dpg in drop_group:
            drop = aa['group']==dpg
            txyzOutlier[beacon]['outlier'] += drop.sum()
            aa = aa.loc[~drop]
        
        txyzPds[beacon]=aa.reset_index()
    

with open("./guider20240808/databank/pkl/origin.pkl", 'wb') as f:
    pickle.dump(txyzPds_origin, f)    
with open("./guider20240808/databank/pkl/filter01.pkl", 'wb') as f:
    pickle.dump(txyzPds, f)

pd.DataFrame(txyzOutlier).to_excel('./output/outliers.xlsx')
