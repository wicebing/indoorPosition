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
        
        txyzPds[beacon]=df

def plot_trajectory(dfs, evt_x, evt_y, evt_what, pic_name='evtTimePoint'):
    """Plots the trajectory of points from a DataFrame.

    Args:
        df: Pandas DataFrame with 'positionTime' (datetime) and 'position' (dict).
    """
    x_min=302491 - 302491
    x_max=302516 - 302491
    y_min=2770397 - 2770397
    y_max=2770422 - 2770397
    scale = 45
    grid_size = 45
    
    fig, ax = plt.subplots(figsize=(10, 10))  # adjust figsize for better view
    # Load the image
    img = Image.open('./guider20240808/databank/ED_Area.png')
    img_array = np.array(img)
    img_array = np.flipud(img_array)     
    ax.imshow(img_array)
    
    colors = {'N002':'blue',
              'N003':'violet',
              'N004':'limegreen',
              'N005':'darkorange',
              'N006':'tomato',
              'N007':'royalblue',
              'N008':'peru',
              'N017':'salmon',
              'N029':'peru',}
    
        
    for k,df in dfs.items():
        # Extract x and y coordinates; handle potential errors.
        x = scale*(df['x']-x_min)
        y = scale*(df['y']-y_min)
        
        df['time_diff'] = df['positionTime'].diff().dt.total_seconds()
        
        times = df['positionTime'].values
        
        # Calculate alpha values for transparency (optional)
        alpha_values = np.linspace(0.0, 0.75, len(x)) # Adjust transparency as needed
        r, g, b = to_rgb(colors[k])
        clr = [(r, g, b, alpha) for alpha in np.floor(alpha_values*10)/10]
        
        # Plot points with transparency
        ax.scatter(x, y, c = clr, alpha=0.5, s = 15)

        # Add a line connecting the points
        consecutive_indices = np.where(df['time_diff'].values > 60)
        if consecutive_indices[0].size > 1:
            cons_i = 0
            for i, idx in enumerate(consecutive_indices[0]):
                x_consecutive = x[cons_i:idx]
                y_consecutive = y[cons_i:idx]
                ax.plot(x_consecutive, y_consecutive, color=colors[k], alpha = 0.1, linestyle = '-')
                cons_i=idx
            if i == consecutive_indices[0].size-1:
                x_consecutive = x[cons_i:]
                y_consecutive = y[cons_i:]
                ax.plot(x_consecutive, y_consecutive, color=colors[k], alpha = 0.1, linestyle = '-')
        else:
            ax.plot(x, y, color=colors[k], alpha = 0.1, linestyle = '-') 
    
    # plot event point
    if evt_what == '轉重症':
        ax.scatter(scale*(evt_x-x_min),scale*(evt_y-y_min), marker='P', s =300, c='black')
    else:
        ax.scatter(scale*(evt_x-x_min),scale*(evt_y-y_min), marker='P', s =300, c='black')
        ax.scatter(scale*(evt_x-x_min),scale*(evt_y-y_min), marker='P', s =88, c='lightyellow')

    # Set plot limits
    major_ticks = np.arange(0, 1125, grid_size)
    ax.grid(which='major', alpha=0.5, linestyle='--')
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    plt.xticks([])
    plt.yticks([])
    
    ax.set_xlim(0,scale*(x_max-x_min))
    ax.set_ylim(0,scale*(y_max-y_min))
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Position Trajectory_{pic_name}')

    # plt.axis('off')
    plt.grid(True)
    plt.savefig(fname=f'./output/pic/{pic_name}.png')
    print(f' === complete {pic_name} image === ')

# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]

# Load the beacon positionTime
with open("./guider20240808/databank/pkl/origin.pkl", 'rb') as f:
    txyzPds_origin = pickle.load(f)
with open("./guider20240808/databank/pkl/filter01.pkl", 'rb') as f:
    txyzPds = pickle.load(f)

# Draw Trajectory 
def Trajectory_plot(drawPds,flag='origin'):
    for i, evt in events.iterrows():
        print(f' == work on {i} event == ')
        hours = 8
        positionTime = evt['positionTime']
        evt_x = evt['X']
        evt_y = evt['Y']
        evt_what = evt['事件分類']
        發生地點 = evt['發生地點']
        startTime = positionTime-datetime.timedelta(hours=hours)
        
        # if i != 30:
        #     continue
        dfs = {}
        for beacon in select_beacons:
            df = drawPds[beacon].loc[(drawPds[beacon]['positionTime'] >= startTime) & (drawPds[beacon]['positionTime'] <= positionTime)]    
            if len(df) > 0:
                dfs[beacon]=df
        
        plot_trajectory(dfs, evt_x-x_min, evt_y-y_min, evt_what, pic_name=f'{i+1}_{發生地點}_{positionTime.hour}_{hours}hour_{flag}')
        
Trajectory_plot(txyzPds,'filter_1')       
Trajectory_plot(txyzPds_origin,'')      
