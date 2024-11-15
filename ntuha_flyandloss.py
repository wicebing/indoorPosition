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
    
    dfc['group'] = dfc['position_diff'] > 1.5*dfc['time_diff']
    # Handle cases with large missing time gaps
    dfc['skip'] = 0
    
    dfc.loc[(dfc['time_diff']>333) & (dfc['position_diff']>2), 'skip'] +=1
    dfc.loc[(dfc['time_diff']>333) & (dfc['position_diff']>2), 'group'] = True
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
        
        for threshold in range(15):
            outliers = 1
            while(outliers>0):
                aa = filter_single(aao)
                group_lapse = aa.groupby('group')['time_diff'].sum()
                aa['group_lapse'] = group_lapse[aa.group].values
                skip_count = aa.value_counts('skip')
                aa['skip_num'] = skip_count[aa.skip].values
                group_count = aa.value_counts('group')
                aa['group_num'] = group_count[aa.group].values
                if threshold < 5:
                    drop = (aa['group_num']<=threshold)
                else:
                    drop = (aa['group_num']<=threshold) & (aa['group_lapse']<=60*threshold)
                aao = aao.loc[~drop]
                print(threshold,drop.sum(),len(aa)-len(aao),len(aa),len(aao))
                outliers = len(aa)-len(aao)
                txyzOutlier[beacon]['outlier'] += outliers

        
        txyzPds[beacon]=aao.reset_index()
    

with open("./guider20240808/databank/pkl/origin.pkl", 'wb') as f:
    pickle.dump(txyzPds_origin, f)    
with open("./guider20240808/databank/pkl/filter01.pkl", 'wb') as f:
    pickle.dump(txyzPds, f)

pd.DataFrame(txyzOutlier).to_excel('./output/outliers.xlsx')
