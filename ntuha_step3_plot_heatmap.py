import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json, pickle, glob
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import to_rgb, to_rgba
import seaborn as sns; sns.set()

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
# txyzPds = {}

# for beacon in beacon_ids:
#     recordName= f'{beacon}.pkl'
#     pickle_filepath = os.path.join(databank_filepath,recordName)
    
#     if os.path.isfile(pickle_filepath):
#         print(f'=== {beacon}.pkl exist, loading ===')

#         txyzPd_origin = pd.read_pickle(pickle_filepath)
#         pd_pz = pd.json_normalize(txyzPd_origin['position'])
#         pd_time = pd.to_datetime(txyzPd_origin['positionTime'],format='mixed').dt.tz_convert(local_timezone)
#         df = pd.concat([pd_time,pd_pz],axis=1)
        
#         df['x'] = df['x']-x_min
#         df['y'] = df['y']-y_min
        
#         txyzPds[beacon]=df


def create_gif(pic_dir, output_gif_path="output.gif", duration=300):
    image_paths = glob.glob('')
    images = [Image.open(image_path) for image_path in image_paths]
    # Save as GIF
    images[0].save(
    output_gif_path,
    save_all=True,
    append_images=images[1:],
    duration=duration,
    loop=0 # 0 means infinite loop
    )

def plot_heatmap(dfs, evt_x, evt_y, evt_what, pic_name='evtTimePoint',grid=False):
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
    
    # Load the image
    img = Image.open('./guider20240808/databank/ED_Area.png')
    img_array = np.array(img)
    img_array = np.flipud(img_array)     
    
    heatmap_rows = img_array.shape[0] // grid_size
    heatmap_cols = img_array.shape[1] // grid_size
    
    df = dfs.copy()
    # Extract x and y coordinates; handle potential errors.
    x = scale*(df['x']-x_min)
    y = scale*(df['y']-y_min)
    
    df['min_diff'] = df['beforeEvtMin'].diff().fillna(0)
    consecutive_indices = np.where(df['min_diff'].values < 0)
    heatmap = np.zeros((heatmap_rows, heatmap_cols), dtype=float)
    gifs = []
    if consecutive_indices[0].size > 1:
        cons_i = 0
        max_heatmap_value = 100
        min_heatmap_value = 0
        for ixxx, idx in enumerate(consecutive_indices[0]):
            min_diff,beforeEvtMin =df.iloc[idx][['min_diff','beforeEvtMin']]
            heatmap += 3*min_diff
            heatmap = np.maximum(heatmap, min_heatmap_value)
            if ixxx == consecutive_indices[0].size-1:
                x_consecutive = x[cons_i:]
                y_consecutive = y[cons_i:]
            else:
                x_consecutive = x[cons_i:idx]
                y_consecutive = y[cons_i:idx]
                cons_i=idx

            fig, ax = plt.subplots(figsize=(10, 10))  # adjust figsize for better view
            ax.imshow(img_array)            

            # Calculate grid cell indices
            grid_x = np.floor(x_consecutive / grid_size).astype(int)
            grid_y = np.floor(y_consecutive / grid_size).astype(int)
            # Check for valid grid indices; very important
            valid_indices = (grid_x >= 0) & (grid_x < heatmap_rows) & (grid_y >= 0) & (grid_y < heatmap_cols)
        
            grid_x = grid_x[valid_indices]
            grid_y = grid_y[valid_indices]
        
            for i, j in zip(grid_x, grid_y):
                # Use safe indexing to avoid out-of-bounds errors
                x_min_index = max(0, i-2)
                x_max_index = min(heatmap_cols - 1, i + 2)
                y_min_index = max(0, j-2)
                y_max_index = min(heatmap_rows - 1, j + 2)
        
                for x_index in range(x_min_index, x_max_index + 1):
                    for y_index in range(y_min_index, y_max_index + 1):
                        if (x_index == i-2 & y_index == j-2) or \
                            (x_index == i+2 & y_index == j-2) or \
                            (x_index == i-2 & y_index == j+2) or \
                            (x_index == i+2 & y_index == j+2):
                                continue
                        heatmap[y_index, x_index] += 1
            heatmap = np.minimum(heatmap, max_heatmap_value)

            for ri in range(heatmap_rows):
                for rj in range(heatmap_cols):
                    alpha = -1*(heatmap[ri,rj]/max_heatmap_value) + 1

                    rect = mpatches.Rectangle((rj*grid_size,ri*grid_size), grid_size, grid_size,
                                              alpha=alpha,
                                              facecolor='navy',
                                              color='navy')
                    ax.add_patch(rect)

            ax.scatter(x_consecutive, y_consecutive, c = 'violet', alpha=0.3, s = 15)

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
            if(not grid):
                plt.xticks([])
                plt.yticks([])
            
            ax.set_xlim(0,scale*(x_max-x_min))
            ax.set_ylim(0,scale*(y_max-y_min))
            
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(f'Position heatmap_{beforeEvtMin}')
        
            # plt.axis('off')
            plt.grid(True)
            pic_filepath = f'./output/heatmap/{pic_name}/{str(int(beforeEvtMin)).rjust(6,"0")}.png'
            os.makedirs(os.path.dirname(pic_filepath),exist_ok=True)
            
            plt.savefig(fname=pic_filepath)
            print(f' === complete {pic_name}{beforeEvtMin} image === ')

# Draw Trajectory 
def heatmap_plot(events, drawPds,hours=1,flag='origin',grid=False):
    for i, evt in events.iterrows():
        print(f' == work on {i} event == ')
        positionTime = evt['positionTime']
        e_x = evt['X']
        e_y = evt['Y']
        evt_what = evt['事件分類']
        發生地點 = evt['發生地點']
        endtime = positionTime
        startTime = endtime-datetime.timedelta(hours=hours)
        
        # if i != 30:
        #     continue
        dfs_ = []
        for beacon in select_beacons:
            df = drawPds[beacon].loc[(drawPds[beacon]['positionTime'] >= startTime) & (drawPds[beacon]['positionTime'] <= endtime)]    
            if len(df) > 0:
                df.loc[:,['positionTime']] = df['positionTime'].dt.round('s')
                df.loc[:,['beforeEvtMin']] = (positionTime-df['positionTime']).dt.round('min')/datetime.timedelta(minutes=1)
                dfs_.append(df)
        dfs = pd.concat(dfs_,axis=0, ignore_index=True)
        dfs = dfs.sort_values('beforeEvtMin', ascending=False)
        plot_heatmap(dfs, 
                     evt_x=e_x-x_min, 
                     evt_y=e_y-y_min, 
                     evt_what=evt_what,
                     pic_name=f'{i+1}_{發生地點}_{positionTime.hour}_{hours}hour_{flag}',
                     grid=grid)
        break
 
# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]

# Load the beacon positionTime
with open("./guider20240808/databank/pkl/filter01.pkl", 'rb') as f:
    txyzPds = pickle.load(f)   

heatmap_plot(events, txyzPds,3,'heatmap_0',grid=False)       
