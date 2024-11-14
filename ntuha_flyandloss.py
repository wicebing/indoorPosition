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
    dfc.loc[(((dfc['x']-dfc['x_m1']).abs()>5) | ((dfc['y']-dfc['y_m1']).abs()>5)), 'group'] +=1
    dfc.loc[(((dfc['x']-dfc['x_m1']).abs()>12) | ((dfc['y']-dfc['y_m1']).abs()>12)) & (dfc['time_diff']<5), 'group'] +=1
    dfc.loc[(((dfc['x']-dfc['x_m1']).abs()>3) | ((dfc['y']-dfc['y_m1']).abs()>3)) & (dfc['time_diff'] > 90), 'group'] +=1
    dfc.loc[(((dfc['x']-dfc['x_avg']).abs()>3) | ((dfc['y']-dfc['y_avg']).abs()>3)), 'fly'] = True
    dfc.loc[(((dfc['x']-dfc['x_avg']).abs()>7) | ((dfc['y']-dfc['y_avg']).abs()>7)) & (dfc['time_diff'] < 5), 'fly'] = True
    
    dfc['fly'] = dfc['fly'].astype(bool).fillna(False) # handle cases where no outlier was detected.
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
        out_boundary = (aao['x']<0)|((aao['x']>25))|(aao['y']<0)|((aao['y']>25))
        txyzOutlier[beacon] = {'origin':len(aao),'outlier':out_boundary.sum()}
        aao = aao.loc[~out_boundary]
        
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
        
            
        outliers_group = 1
        while_loop = 0
        while(outliers_group>0):
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
                if now_x < 1 or now_x >24:
                    drop_group.append(dgp)
                if now_y < 1 or now_y >26:
                    drop_group.append(dgp)
                    
                try:
                    if (abs(group_x[int(dgp-1)]-group_x[int(dgp+1)])<6) & \
                        (abs(group_y[int(dgp-1)]-group_y[int(dgp+1)])<6) & \
                        ((abs(now_y-group_y[int(dgp-1)])>4)| \
                        (abs(now_x-group_x[int(dgp-1)])>4)) & \
                        ((group_count[int(dgp)]<(group_count[int(dgp-1)])) | \
                        (group_count[int(dgp)]<(group_count[int(dgp+1)]))) & \
                          (group_lapse[int(dgp)]<300):
                              drop_group.append(dgp)                    
                except:
                    pass

                try:
                    if ((abs(now_x-group_x[int(dgp-1)])>6)|(abs(now_y-group_y[int(dgp-1)])>6)) &\
                        (group_count[int(dgp)]<(group_count[int(dgp-1)])) & \
                        (group_count[int(dgp)]<(group_count[int(dgp+1)])) & \
                         (group_count[int(dgp)]<75) & \
                          (group_lapse[int(dgp)]<300):
                              drop_group.append(dgp)                      
                except:
                    pass

                try:
                    if (abs(group_x[int(dgp-1)]-group_x[int(dgp+1)])<3) & \
                        (abs(group_y[int(dgp-1)]-group_y[int(dgp+1)])<3) & \
                          (group_lapse[int(dgp)]<300):
                              drop_group.append(dgp)                    
                except:
                    pass
            drop_group=list(set(drop_group))
            outliers_group = len(drop_group)
            
            drop_idx_ = []
            for dpg in drop_group:
                drop = aa['group']==dpg
                drop_idx_.append(drop)
                # txyzOutlier[beacon]['outlier'] += drop.sum()
                # print(f'drop {while_loop} group{dpg} {drop.sum()}')
                # aa = aa.loc[~drop]
            drop_idx = pd.concat(drop_idx_,axis=1)
            dropAll = drop_idx.sum(axis=1).astype(bool)
            txyzOutlier[beacon]['outlier'] += dropAll.sum()
            print(f'drop {while_loop} {dropAll.sum()}')
            aao = aao.loc[~dropAll]
            while_loop += 1
        
        txyzPds[beacon]=aao.reset_index()
    

with open("./guider20240808/databank/pkl/origin.pkl", 'wb') as f:
    pickle.dump(txyzPds_origin, f)    
with open("./guider20240808/databank/pkl/filter01.pkl", 'wb') as f:
    pickle.dump(txyzPds, f)

pd.DataFrame(txyzOutlier).to_excel('./output/outliers.xlsx')
