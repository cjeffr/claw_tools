import pandas as pd
import os
import glob
import numpy as np


def load_waveforms_pandas(SOURCE_DIR, loc_idx):
    """
    This function will load all simulations gauge data into a dictionary with 
    key names for the simulation name.
    All simulations must be in the same parent directory
    """
    from pathlib import Path
    
    #Create an empty dictionary to hold all the data 
    # get folder names inside SOURCEDIR
    df_dict = {}
    sim_folders = [f.name for f in os.scandir(SOURCE_DIR) if f.is_dir()]
    df_dict = {key:[] for key in sim_folders}
    df_dict
    data = {}
    # Loop over simulation folders to get all gauge data loaded into the dictionary
    for fldr in df_dict:
        for path in Path(os.path.join(SOURCE_DIR, fldr)).rglob(f'gauge{loc_idx}*.txt'):
            file = str(path)
            
            # Gets header information from the gauge file includes lat/lon not being used currently
            with open(file) as f:
                header = f.readline().split()
                gauge_id = int(header[2][-3:])
                cols = ['Time', f'{gauge_id}']
                lat = float(header[5])
                lon = float(header[4])

            data[gauge_id] = pd.read_csv(file, skiprows=3, header=None, 
                                       delim_whitespace=True, usecols=[1,5], index_col='Time', 
                                         names=cols)
        df_dict[fldr] = data

    return df_dict


def merge_clean_data(data_dict, storm_id):
    """
    Takes a dictionary of dataframes, concatenates them together and
    Changes time from seconds to hours from landfall
    """
    df = pd.concat(data_dict, axis=1)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    landfall_time = get_landfall_time(storm_id)
    df.index -= landfall_time
    df.index /= 3600.0
    return df


def get_landfall_time(storm_id):
    import xarray as xr
    storm_data = xr.open_dataset(os.path.join('/home/catherinej/storm_files', 
                                              f'NACCS_TP_{storm_id:04}_SYN_L2.nc'))
    landfall_loc =  [-73.31, 40.68]
    eye = storm_data.eye_loc.data
    lat_row = np.where(eye[:,1] >= 40.68)
    lon_row = np.where(eye[:,0] <= -73.31)
    idx = max(min(lat_row[0]), min(lon_row[0]))
    landfall_time = storm_data.time[idx].values
    return landfall_time


def multiple_gauges_subplots(data, num_plots, outfile):
    import yaml
    # Open gmt config file for plot settings
    with open('gmt_config.yml', 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        
    # Assign config parameters to this plot only
    with pygmt.config(**cfg['xy_plot']):
        with fig.subplot(
            nrows = 2,
            ncols = 3, 
            figsize=('24c', '12c'),
            frame='lrtb',
            margins='1c/1c',
            sharex = 'b',
            sharey = 'l'):
            
            for idx, gauge in enumerate(data):
                with fig.set_panel(panel=idx):
                    pygmt.makecpt(cmap='categorical',
                                  series = [0, len(data[gauge]), 1],
                                  color_model = f'+c0-{len(data[gauge])}')
                    fig.basemap(region=region, 
                                projection = 'X?', 
                                frame = ['ya.35f.125+l"Sea Surface (m)"',
                                         'xa2f1+l"Hours from Landfall"',
                                         f'+t"Gauge {gauge}"'])
                    with pygmt.config(
                        MAP_FRAME_TYPE='inside'):
                        fig.basemap(region=region,
                                    projection = 'X?',
                                    frame=['yf.125', 'xf1'])
                    df_list = df_dict[gauge]
                    sorted_dflist = sorted(df_list, 
                                           key = lambda x: x.columns[0])
                    for index, df in enumerate(sorted_dflist):
                        ycol = df.columns[0]
                        fig.plot(x = df.index,
                                 y = df[ycol].values,
                                 zvalue = index,
                                 cmap = True,
                                 pen = 'thick,+z,-',
                                 label = ycol)
                    fig.legend(position = 'jTR+jTR+o-1.85c/0.02c',
                               box = True,
                               region = region)
    fig.savefig(f'{outfile}.png')
    

                                
def calc_distance(lat1, lat2, lon1, lon2):
    from math import sin, cos, sqrt, atan2, radians
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = sin(dlat / 2)**2 + cos(lat1)*cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = R * c
    return distance


def gauges_in_bay(x,y):
    bay_coords = [-72.875655, -72.6617104, 40.730813, 40.791133]
    v1 = (bay_coords[1] - bay_coords[0], bay_coords[3] - bay_coords[2])
    v2 = (x - bay_coords[0],  y - bay_coords[2])
    xp = v1[0]*v2[1] - v1[1]*v2[0]
    if xp > 0:
        return 'bay'
    else:
        return 'ocean'


def find_gauge_locations(PATH):
    gauge_loc_df = pd.read_csv(PATH)
    landfall_loc = [-73.31, 40.68]
    for idx in gauge_loc_df.index:
        lat = gauge_loc_df.iloc[idx]['lat']
        lon = gauge_loc_df.iloc[idx]['lon']
        dist = calc_distance(lat, landfall_loc[1],
                             lon, landfall_loc[0])
        gauge_loc_df.loc[idx, 'Distance'] = dist
        location = gauges_in_bay(lon, lat)
        gauge_loc_df.loc[idx, 'Location'] = location
    bay_gauges = gauge_loc_df.loc[gauge_loc_df['Location'] == 'bay']
    bay_gauges = bay_gauges.rename(columns={'Unnamed: 0': 'GaugeNames'})
    return bay_gauges


def load_gauge_data(allfiles):
    df_list = []
    for file in allfiles:
        sim_name = file.split('/')[4]
        cols = ['Time', f'{sim_name}']
        df = pd.read_csv(file, skiprows=3, header=None,
                         delim_whitespace=True, usecols=[1,5],
                         names = cols, index_col='Time')
        if sim_name == 'no_breach':
            no_breach = df
        df_list.append(df)
    df_list = reindex_df(df_list, no_breach)
    data_df = merge_clean_data(df_list, 486)
    return data_df
    
        
    
def reindex_df(df_list, res_15):
    df_l = []
    for df in df_list:
        df_re = df.reindex(df.index.union(res_15.index)).interpolate('index').reindex(res_15.index)
        df_l.append(df_re)
    return df_l