import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json, pickle
from functools import reduce
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
    dfc['skip_change'] = 0
    
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'skip_change'] +=1
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'group'] = True
    dfc['skip'] = dfc['skip_change'].cumsum()
    dfc['group'] = dfc['group'].cumsum()
    
    dfc['weekday'] = dfc[time_col].dt.weekday
    dfc['hour'] = dfc[time_col].dt.hour
    
    dfc['loss_tick'] = np.maximum(np.floor(dfc['time_diff'] - 1),0).fillna(0)
    dfc['id_hours'] = dfc['positionTime'].dt.round('min')
    
    temp = dfc[dfc['skip_change']>0]
    dfc.loc[temp.index,'time_diff']=0
    dfc.loc[temp.index,'position_diff']=0
    
    return dfc

# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]

# Load the beacon positionTime
with open("./guider20240808/databank/pkl/filter01.pkl", 'rb') as f:
    txyzPds = pickle.load(f) 

aao = txyzPds.copy()
aa={}
lossTick = {}
for k,v in aao.items():
    aa[k] = filter_single(v)
    
N029 = aa['N029'].set_index('positionTime')
N008 = aa['N008'].set_index('positionTime')

N008new = pd.concat([N008[:"2024-10-17 09:00:00"],N029["2024-10-17 09:01:00":]],
                    axis=0,
                    ignore_index=False)
aa.pop('N029')
aa['N008'] = filter_single(N008new.reset_index())

loadings = []
for k, v in aa.items():
    res = v.groupby(['id_hours', 'skip']).agg({'time_diff': 'sum', 'position_diff': 'sum'})
    load = res.reset_index().groupby(['id_hours']).agg({'time_diff': 'sum', 'position_diff': 'sum'})
    load['mps'] = load['position_diff']/load['time_diff']
    load = load.rename(columns={'time_diff': f'time_diff_{k}', 
                                'position_diff': f'position_diff_{k}',
                                'mps': f'mps_{k}',})
    loadings.append(load.reset_index())

load_all = reduce(lambda x, y: x.merge(y, how='outer', on='id_hours'), loadings)
# Sum time_diff columns (after renaming)
time_diff_cols = [col for col in load_all.columns if col.startswith('time_diff_')]
load_all['time_diff_all'] = load_all[time_diff_cols].sum(axis=1)
position_diff_cols = [col for col in load_all.columns if col.startswith('position_diff_')]
load_all['position_diff_all'] = load_all[position_diff_cols].sum(axis=1)
load_all['mps_all'] = -1*load_all['position_diff_all']/load_all['time_diff_all']

load_all['event'] = 0
load_all = load_all.set_index('id_hours')

mps_cols = [col for col in load_all.columns if col.startswith('mps_')]
plot_data = load_all[mps_cols].copy()


# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]

plot_data['event'] = 0
for i, evt in events.iterrows():
    print(f' == work on {i} event == ')
    positionTime = evt['positionTime']
    evt_x = evt['X']
    evt_y = evt['Y']
    evt_what = evt['事件分類']
    發生地點 = evt['發生地點']
    endtime = positionTime-datetime.timedelta(hours=0.)
    startTime = endtime-datetime.timedelta(hours=1)
    
    # load_all.loc[endtime,'event'] = 1 if evt_what=='轉重症' else 2


# consecutive_indices = np.where(load_all['event'].values==1)

# if consecutive_indices[0].size > 1:
#     cons_i = 0
#     for i, idx in enumerate(consecutive_indices[0]):
    fig, ax = plt.subplots(figsize=(20, 10))  # adjust figsize for better view
    
    x_consecutive = plot_data.loc[startTime:endtime]
    
    ax = x_consecutive.plot(figsize=(30,10),ylim=(-0.5,0.5))
    plt.savefig(fname=f'./output/loading/{i}_{evt_what}.png')

    plot_data.loc[startTime:endtime,['event']] = 1 if evt_what=='轉重症' else 2

plot_data['event'] = plot_data['event'].fillna(0)

jjj = plot_data.groupby('event').agg({k: ['mean','std'] for k in mps_cols})
jjj.to_csv('./output/loading_report_byCart.csv')

ana = plot_data.reset_index().copy()
ana['weekday'] = ana['id_hours'].dt.weekday
ana['hour'] = ana['id_hours'].dt.hour
analysis_ = [] 
for k in mps_cols:
    if k == 'mps_all': continue
    temp = ana.loc[:,[k,'event','weekday','hour']]
    temp.columns = ['mps','event','weekday','hour']
    analysis_.append(temp)

analysis = pd.concat(analysis_, axis=0, ignore_index=True)
jjj2 = analysis.groupby('event').agg({'mps': ['mean','std']})
print(jjj2)

analysis.dropna().to_csv('./analysis/loading_event.csv')


akk = analysis.groupby(['weekday','hour']).agg({'mps': ['mean','std']})
akk = akk.reset_index()
akk = akk.pivot(columns='weekday', index='hour')
akk.to_csv('./output/loading_report_bydayhour.csv')
