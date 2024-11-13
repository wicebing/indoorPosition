import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json. pickle
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors

import utils

databank_filepath = "./guider20240808/databank/positions"
os.makedirs(databank_filepath,exist_ok=True)

local_timezone = pytz.timezone('Asia/Taipei')  

beacon_ids = utils.get_beacons()
print('=== load beacons ids ===')

txyzPds = {}

for beacon in beacon_ids:
    recordName= f'{beacon}.pkl'
    pickle_filepath = os.path.join(databank_filepath,recordName)
    
    if os.path.isfile(pickle_filepath):
        print(f'=== {beacon}.pkl exist, loading ===')

        txyzPd_origin = pd.read_pickle(pickle_filepath)
        pd_pz = pd.json_normalize(txyzPd_origin['position'])
        pd_time = pd.to_datetime(txyzPd_origin['positionTime'],format='mixed').dt.tz_convert(local_timezone)
        df = pd.concat([pd_time,pd_pz],axis=1)
        
        txyzPds[beacon]=df

events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]


def plot_trajectory(dfs, evt_x, evt_y, pic_name='evtTimePoint'):
    """Plots the trajectory of points from a DataFrame.

    Args:
        df: Pandas DataFrame with 'positionTime' (datetime) and 'position' (dict).
    """
    x_min=302491
    x_max=302516
    y_min=2770397
    y_max=2770422
    scale = 45
    grid_size = 45
    
    fig, ax = plt.subplots(figsize=(10, 10))  # adjust figsize for better view
    # Load the image
    img = Image.open('./guider20240808/databank/ED_Area.png')
    img_array = np.array(img)
    img_array = np.flipud(img_array)     
    ax.imshow(img_array)
    
    colors = ['blue','violet','limegreen','darkorange',
              'tomato','gold','peru','salmon','hotpink']
    
    for i, df in enumerate(dfs):
        # Extract x and y coordinates; handle potential errors.
        x = scale*(df['x']-x_min)
        y = scale*(df['y']-y_min)
        
        # Calculate alpha values for transparency (optional)
        alpha_values = np.linspace(1, 0.2, len(x)) # Adjust transparency as needed
        
        # Plot points with transparency
        ax.scatter(x, y, c = colors[i], alpha=0.3, s = 10)

    # Add a line connecting the points
    ax.plot(x, y, color=colors[i], alpha = 0.1, linestyle = '-') 
    
    # plot event point
    ax.scatter(scale*(evt_x-x_min),scale*(evt_y-y_min), marker='P', s =200, c='black')

    # Set plot limits
    major_ticks = np.arange(0, 1125, grid_size)
    ax.grid(which='major', alpha=0.5, linestyle='--')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    
    ax.set_xlim(0,scale*(x_max-x_min))
    ax.set_ylim(0,scale*(y_max-y_min))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Position Trajectory')

    # plt.axis('off')
    plt.grid(True)
    plt.show()

select_beacons =['N002', 'N003', 'N004', 'N005', 'N006', 'N007', 'N008', 'N017', 'N029']

for i, evt in events.iterrows():
    dfs = []
    positionTime = evt['positionTime']
    evt_x = evt['X']
    evt_y = evt['Y']
    evt_what = evt['事件分類']
    startTime = positionTime-datetime.timedelta(hours=1)
    
    for beacon in select_beacons:
        df = txyzPds[beacon].loc[(txyzPds[beacon]['positionTime'] >= startTime) & (txyzPds[beacon]['positionTime'] <= positionTime)]    
        if len(df) > 0:
            dfs.append(df)
    
    plot_trajectory(dfs, evt_x, evt_y, pic_name='evtTimePoint')
    
        
    
