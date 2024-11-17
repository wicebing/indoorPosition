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
    loadings.append(load.reset_index())

load_all = reduce(lambda x, y: x.merge(y, how='outer', on='id_hours'), loadings)



def export_lossTick_to_excel(lossTick, output_filename="./output/lossTick_report.xlsx"):
    """Exports lossTick data to an Excel file with multiple sheets.

    Args:
        lossTick (dict): A dictionary where keys are names of sheets (e.g., device IDs or 'all')
                         and values are dictionaries containing 'lossTick', 'byWDH', and 'lossTickPercent' DataFrames.
        output_filename (str): The name of the output Excel file.
    """

    with pd.ExcelWriter(output_filename) as writer:
        for sheet_name, data_dict in lossTick.items():
            for data_label, df in data_dict.items():

                #To handle multiindex dataframe to be exportable
                df = df.copy()  # Create a copy to avoid modifying the original
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(map(str, col)).strip('_') for col in df.columns.values]
                if isinstance(df.index, pd.MultiIndex):
                    df.index = ['_'.join(map(str, col)).strip('_') for col in df.index.values]

                df.to_excel(writer, sheet_name=f"{sheet_name}_{data_label}")  # Include the label in the sheet name


# Example usage:
# export_lossTick_to_excel(lossTick)
