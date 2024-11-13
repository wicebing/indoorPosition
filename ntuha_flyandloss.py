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

beacon_ids = utils.get_beacons()
print('=== load beacons ids ===')

x_min=302491
x_max=302516
y_min=2770397
y_max=2770422
scale = 45
grid_size = 45
txyzPds = {}
txyzOutlier = {}

def detect_and_label_outliers(df, columns_to_check=['x', 'y'], window='3s'):
    dfc = df.set_index('positionTime').copy()  # Avoid modifying the original DataFrame

    dfc['x_avg'] = dfc['x'].rolling(window).mean()
    dfc['y_avg'] = dfc['y'].rolling(window).mean()

    # Identify outliers
    dfc['fly'] = None
    dfc.loc[((dfc['x']-dfc['x_avg']).abs()>5) | ((dfc['y']-dfc['y_avg']).abs()>3), 'fly'] = True

    dfc['fly'] = dfc['fly'].fillna(False) # handle cases where no outlier was detected.
    return dfc.reset_index()

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
        
        txyzOutlier[beacon] = {'origin':len(df),'outlier':0}
        for i in range(5):
            aa = detect_and_label_outliers(df, columns_to_check=['x', 'y'], window='3s')
            df = df.loc[~aa.fly]
            print(len(aa)-len(df),len(aa),len(df))
            txyzOutlier[beacon]['outlier'] += len(aa)-len(df)
        
        txyzPds[beacon]=df

