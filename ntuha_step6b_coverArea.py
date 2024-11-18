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
 
select_beacons =['N002', 'N003', 'N004', 'N005', 'N006', 'N007', 'N008', 'N017']
beacon_ids = select_beacons #utils.get_beacons()
print('=== load beacons ids ===')

x_min=302491
x_max=302516
y_min=2770397
y_max=2770422
scale = 45
grid_size = 45

# Define coordinates to remove
all_area_coords = set()
for i in range(25):   # grid_x <= 7 (0-7)
    for j in range(25):  # grid_y >= 16
        all_area_coords.add((i, j))
remove_coords = set()
for i in range(25):  # Or range(25) if your grid is actually 0-24
    remove_coords.add((0, i))     # grid_x = 0
    remove_coords.add((24, i))    # grid_x = 24
for j in range(25):  # Or range(25) if your grid is actually 0-24
    remove_coords.add((j, 0))     # grid_y = 0
    remove_coords.add((j, 24))    # grid_y = 24
for i in range(8):   # grid_x <= 7 (0-7)
    for j in range(16, 25):  # grid_y >= 16
        remove_coords.add((i, j))
for i in range(3):   # grid_x <= 7 (0-7)
    for j in range(13, 16):  # grid_y >= 16
        remove_coords.add((i, j))
for i in range(3):   # grid_x <= 7 (0-7)
    for j in range(13, 16):  # grid_y >= 16
        remove_coords.add((i, j))
for i in range(4,18):   # grid_x <= 7 (0-7)
    for j in range(5):  # grid_y >= 16
        remove_coords.add((i, j))
all_area_coords = all_area_coords-remove_coords

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
    
    dfc['grid_x'] = np.floor(dfc['x']).astype(int)
    dfc['grid_y'] = np.floor(dfc['y']).astype(int)
    
    # Create the 'axis' column
    def get_axis_values(row, heatmap_rows, heatmap_cols):
        i, j = row['grid_x'], row['grid_y']
        axis_values = []
        x_min_index = max(0, i-3)
        x_max_index = min(heatmap_cols - 1, i + 3)
        y_min_index = max(0, j-3)
        y_max_index = min(heatmap_rows - 1, j + 3)

        for x_index in range(x_min_index, x_max_index + 1):
            for y_index in range(y_min_index, y_max_index + 1):
                if (x_index == i-3 and y_index == j-3) or \
                   (x_index == i+3 and y_index == j-3) or \
                   (x_index == i-3 and y_index == j+3) or \
                   (x_index == i+3 and y_index == j+3):
                       continue  # Skip the corner cells.
                axis_values.append((x_index, y_index)) # Store as tuples
        return axis_values

    dfc['axis'] = dfc.apply(get_axis_values, args=(25, 25), axis=1)    
    
    return dfc

aao = []
# Load the beacon positionTime
for k in beacon_ids:
    print(f' == load {k} == ')
    with open(f"./guider20240808/databank/pkl/filter02_gridxy_{k}.pkl", 'rb') as f:
        kkk = pickle.load(f)
        print(f' == group {k} == ')
        jjj = kkk.groupby('id_hours')['axis'].sum().reset_index()
        print(f' == set & list {k} == ')
        jjj.loc[:,['axis']] = jjj['axis'].apply(set).apply(list)
        aao.append(jjj)
        
print(f' == concat == ')
combined_beacons = pd.concat(aao,axis=0, ignore_index=True)
print(f' == group_last == ')
byhour_coverArea =combined_beacons.groupby('id_hours')['axis'].sum()

byhour_coverArea = byhour_coverArea.sort_index().reset_index()
byhour_coverArea.loc[:,['axis']] = byhour_coverArea['axis'].apply(set)

aa3 = byhour_coverArea.copy()
aa3.loc[:,['axis']] = aa3['axis'].apply(lambda x: x - remove_coords)
aa3['corrd_number'] = aa3['axis'].apply(len)
aa3['cover_area_pct'] = aa3['corrd_number']/len(all_area_coords)

aa3['weekday'] = aa3['id_hours'].dt.weekday
aa3['hour'] = aa3['id_hours'].dt.hour

akk = aa3.groupby(['weekday','hour']).agg({'cover_area_pct': ['mean','std']})
akk = akk.reset_index()
akk = akk.pivot(columns='weekday', index='hour')
akk.to_csv('./output/areaPct_report_bydayhour.csv')



# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]


plot_data = aa3.copy().set_index('id_hours')
plot_data['event'] = 0
for i, evt in events.iterrows():
    print(f' == work on {i} event == ')
    positionTime = evt['positionTime']
    evt_x = evt['X']
    evt_y = evt['Y']
    evt_what = evt['事件分類']
    發生地點 = evt['發生地點']
    endtime = positionTime-datetime.timedelta(hours=0.25)
    startTime = endtime-datetime.timedelta(hours=1)
    
    fig, ax = plt.subplots(figsize=(20, 10))  # adjust figsize for better view
    
    x_consecutive = plot_data.loc[startTime:endtime,['cover_area_pct']]
    
    ax = x_consecutive.plot(figsize=(30,10),ylim=(0,1))
    plt.savefig(fname=f'./output/areaPct/{i}_{evt_what}.png')

    plot_data.loc[startTime:endtime,['event']] = 1 if evt_what=='轉重症' else 2



jjj2 = plot_data.groupby('event').agg({'cover_area_pct': ['mean','std']})
print(jjj2)




