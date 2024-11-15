import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json, pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, to_rgba

from scipy.interpolate import interp1d
from pykalman import KalmanFilter
from scipy.signal import savgol_filter


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

def filter_single(df, time_col='positionTime'):
    dfc = df.copy().sort_values(by=time_col)

    # Calculate differences in position and time (in seconds)
    dfc['x_diff'] = dfc['x'].diff()
    dfc['y_diff'] = dfc['y'].diff()
    dfc['time_diff'] = dfc[time_col].diff().dt.total_seconds()
    dfc['position_diff'] = (dfc['x_diff']**2 + dfc['y_diff']**2)**0.5
    
    dfc['group'] = dfc['position_diff'] > 2.5
    # Handle cases with large missing time gaps
    dfc['skip'] = 0
    
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'skip'] +=1
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'group'] = True
    dfc['skip'] = dfc['skip'].cumsum()
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
        
        aao = df.dropna().copy()
        aao['timesss'] = aao.index
        out_boundary = (aao['x']<0)|((aao['x']>25))|(aao['y']<0)|((aao['y']>25))
        txyzOutlier[beacon] = {'origin':len(aao),'outlier':0, 'out_boundary':out_boundary.sum()}
        print(f'out_boundary {out_boundary.sum()}')
        aao = aao.loc[~out_boundary]
        
        outliers = 1
        while(outliers>0):
            aa = filter_single(aao)
            skip_count = aa.value_counts('skip')
            aa['skip_num'] = skip_count[aa.skip].values
            group_count = aa.value_counts('group')
            aa['group_num'] = group_count[aa.group].values
            drop = (aa['skip_num']<=1)
            aao = aao.loc[~drop]
            print(len(aa)-len(aao),len(aa),len(aao))
            outliers = len(aa)-len(aao)
            txyzOutlier[beacon]['outlier'] += outliers
        
        threshold = 2
        outliers = 1
        while(outliers>0):
            aa = filter_single(aao)
            group_lapse = aa.groupby('group')['time_diff'].sum()
            aa['group_lapse'] = group_lapse[aa.group].values
            skip_count = aa.value_counts('skip')
            aa['skip_num'] = skip_count[aa.skip].values
            group_count = aa.value_counts('group')
            aa['group_num'] = group_count[aa.group].values
            drop = (aa['group_num']<=threshold)
            aao = aao.loc[~drop]
            print(threshold,drop.sum(),len(aa)-len(aao),len(aa),len(aao))
            outliers = len(aa)-len(aao)
            txyzOutlier[beacon]['outlier'] += outliers


        outliers_group = 1
        while_loop = 0
        
        while(outliers_group>0):
            aa = filter_single(aao)
            groupInTime= aa.groupby('group')['skip'].value_counts()
            groupInTime = groupInTime.reset_index().set_index('group')
            group_lastxy = aa.groupby('group')[['x','y']].last()
            group_firstxy = aa.groupby('group')[['x','y']].first()
            group_diff = aa.groupby('group')[['time_diff','position_diff']].first()
            group_x = aa.groupby('group')['x'].mean()
            group_y = aa.groupby('group')['y'].mean()
            group_count = aa.value_counts('group')
    
            drop_group = []
            for gp in range(len(groupInTime)):
                now_x = group_x[gp]
                now_y = group_y[gp]
                if now_x < 1 or now_x >24:
                    drop_group.append(gp)
                if now_y < 1 or now_y >26:
                    drop_group.append(gp)
                    
                if gp==0: continue
            
                skip_now = groupInTime.loc[gp]['skip']
                skip_m1 = groupInTime.loc[gp-1]['skip']
                sss = group_diff.loc[gp]
                xm1,ym1 = group_lastxy.loc[gp-1]
                x1,y1 = group_firstxy.loc[gp] 
                
                if skip_now > skip_m1: continue        
                if sss['position_diff'] < sss['time_diff']: 
                    if xm1<=9.5 and (x1>10.5 or now_x>11):
                        if group_count[gp]<60:
                            drop_group.append(gp)
                    continue        
                if xm1<=9.5 and (x1>10.5 or now_x>11):
                    if group_count[gp]<60:
                        drop_group.append(gp)
                
                now_xm1 = group_x[gp-1]
                if x1<=9.5 and (xm1>10.5):
                    if group_count[gp]<60:
                        drop_group.append(gp-1)
                    
            drop_group=list(set(drop_group))
            outliers_group = len(drop_group)
            
            if outliers_group >0:
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
