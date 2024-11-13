import pandas as pd
import numpy as np
from PIL import Image
import datetime,os,math, pytz, json. pickle
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

def detect_and_label_outliers(df, columns_to_check=['x', 'y'], window='3s'):
    x_min=302491
    y_min=2770397

    dfc = df.set_index('positionTime').copy()  # Avoid modifying the original DataFrame
    dfc['x'] -= x_min
    dfc['y'] -= y_min

    dfc['x_avg'] = dfc['x'].rolling(window).mean()
    dfc['y_avg'] = dfc['y'].rolling(window).mean()

    # Identify outliers
    dfc['fly'] = None
    dfc.loc[((dfc['x']-dfc['x_avg']).abs()>5) | ((dfc['y']-dfc['y_avg']).abs()>3), 'outlier'] = True

    # dfc['fly'] = dfc['fly'].fillna(False) # handle cases where no outlier was detected.
    return dfc.reset_index()

aao = txyzPds["N029"].copy()

for i in range(1):
    aa = detect_and_label_outliers(aao, columns_to_check=['x', 'y'], window='3s')
    aao = aa.loc[~aa.fly]
    print(len(aa)-len(aao),len(aa),len(aao))
