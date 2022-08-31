import pandas as pd
import os
import glob

def load_waveforms_pandas(SOURCE_DIR):
    """
    This function will load all simulations gauge data into a dictionary with 
    key names for the simulation name.
    All simulations must be in the same parent directory
    """
    from pathlib import Path
    
    #Create an empty dictionary to hold all the data 
    # get folder names inside SOURCEDIR
    df_dict = {}
    sim_folders = [f.name for f in os.scandir(PATH) if f.is_dir()]
    df_dict = {key:[] for key in sim_folders}
    df_dict
    data = {}
    # Loop over simulation folders to get all gauge data loaded into the dictionary
    for fldr in df_dict:
        for path in Path(os.path.join(PATH, fldr)).rglob('gauge*.txt'):
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
    df.columns = df.columns.droplevel(0)
    landfall_time = get_landfall_time(storm_id)
    df.index -= landfall_time
    df.index /= 3600.0
    return df


def get_landfall_time(storm_id):
    import xarray as xr
    storm_data = xr.open_dataset(os.path.join('/home/catherinej/storm_files,
                                              f'NACCS_TP_{storm_id:04}_SYN_L2.nc'))
    landfall_loc =  [-73.31, 40.68]
    eye = ds.eye_loc.data
    lat_row = np.where(eye[:,1] >= 40.68)
    lon_row = np.where(eye[:,0] <= -73.31)
    idx = max(min(lat_row[0]), min(lon_row[0]))
    landfall_time = ds.time[idx].values
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
    

                                
        