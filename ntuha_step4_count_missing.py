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
    
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'skip'] +=1
    dfc.loc[(dfc['time_diff']>10) & (dfc['position_diff']>2), 'group'] = True
    dfc['skip'] = dfc['skip'].cumsum()
    dfc['group'] = dfc['group'].cumsum()
    
    dfc['weekday'] = dfc[time_col].dt.weekday
    dfc['hour'] = dfc[time_col].dt.hour
    
    dfc['loss_tick'] = np.maximum(np.floor(dfc['time_diff'] - 1),0).fillna(0)
    dfc['id_hours'] = dfc['positionTime'].dt.round('h')

    return dfc

# Load the event timePoint
events = pd.read_excel("./guider20240808/databank/events.xlsx")
events['日期'] = events['日期'].astype(str)
events['時間'] = events['時間'].astype(str)
events['positionTime'] = pd.to_datetime(events['日期'] + ' ' + events['時間'], format='%Y-%m-%d %H%M', errors='coerce').dt.tz_localize(local_timezone)
events = events[['positionTime','發生地點','事件分類', 'X', 'Y']]

# Load the beacon positionTime
with open("./guider20240808/databank/pkl/origin.pkl", 'rb') as f:
    txyzPds_origin = pickle.load(f)

aao = txyzPds_origin.copy()
aa={}
lossTick = {}
for k,v in aao.items():
    aa[k] = filter_single(v) 

temp_all = []
temp_h_all = []
for k,v in aa.items():
    lossTick[k] = {}
    temp = v.groupby(['weekday','hour'])['time_diff'].sum()
    temp_h = v.groupby(['weekday','hour'])['id_hours'].nunique()
    
    temp = temp.reset_index()
    temp_h = temp_h.reset_index()
    temp = temp.pivot(columns='weekday', index='hour')
    temp_h = temp_h.pivot(columns='weekday', index='hour')
    temp_h = temp_h*60*60
    
    lossTick[k]['lossTick'] = temp
    lossTick[k]['byWDH'] = temp_h
    lossTick[k]['lossTickPercent'] = pd.DataFrame(temp.values/temp_h.values)
    
    temp_all.append(temp)
    temp_h_all.append(temp_h)

lossTick_all = reduce(lambda x, y: x.add(y, fill_value=0), temp_all)
hours_all = reduce(lambda x, y: x.add(y, fill_value=0), temp_h_all)
lossTickPercent_all = lossTick_all.values/hours_all.values

lossTick['all'] = {}
lossTick['all']['lossTick'] =lossTick_all
lossTick['all']['byWDH'] = hours_all
lossTick['all']['lossTickPercent'] = pd.DataFrame(lossTickPercent_all)



def export_lossTick_to_excel(lossTick, output_filename="lossTick_report.xlsx"):
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
export_lossTick_to_excel(lossTick)