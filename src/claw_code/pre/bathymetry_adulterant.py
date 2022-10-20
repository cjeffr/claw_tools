import numpy as np
import os
from clawpack.geoclaw import topotools
from clawpack.amrclaw import region_tools

def find_island_points(topo_data):
    island_x = []
    points = []
    inds = []
    for i, lon in enumerate(topo_data.x):
        first_point = None
        row = np.array(topo_data.Z[:, i])
        not_land = np.where(row <= 0)
        land = np.where(row > 0)

        for point in land[0]:
            if topo_data.Z[point - 1, i] <= 0 and topo_data.Z[point, i] > 0:
                first_point = (point, topo_data.y[point], i, lon)
                break
        if first_point:
            for point in not_land[0]:
                if point > first_point[0] and topo_data.Z[point + 1, i] <= 0:
                    last_point = (point, topo_data.y[point], i, lon)
                    island_x.append((first_point, last_point))
                    break
    for p in island_x:
        inds.append([p[0][2], p[0][0], p[1][0]])
        for i in p:
            points.append([i[3], i[1]])

    return {
        "island_x": island_x,
        "inds": inds,
        "points": points
    }


def create_island_mask(topo_data, points_indices):
    indices = []
    for data in points_indices:
        for i in range(data[1], data[2] + 1):
            indices.append((data[0], i))
            
    mask_array = np.zeros(shape=topo_data.Z.shape)
    
    for locs in indices:
        col, row = locs
        mask_array[row, col] = 1
        
    masked_data = np.ma.masked_array(topo_data.Z, np.logical_not(mask_array))
    np.savez('/home/catherinej/BarrierBreach/data/masked_island.npz', data=masked_data, mask=masked_data.mask)
    return masked_data


def calc_no_island_values(topo_data, island_idxs):
    replace_mask = []
    for idx in island_idxs:
        first = idx[0]
        last = idx[1]
        avg = (topo_data.Z[first[0] - 1,first[2]] + topo_data.Z[last[0] + 1,first[2]])/2
        replace_mask.append([first[0], last[0], first[2], avg])
        
    return replace_mask


def remove_island(topo_data, island_idxs):
    replace_data = calc_no_island_values(topo_data, island_idxs)
    for zdata in replace_data:
        topo_data.Z[zdata[0]:zdata[1], zdata[2]] = zdata[3]
    return topo_data

    
def find_inlet(topo_data):
    inlet_y = []
    points = []
    inds = []
    for i, lat in enumerate(topo_data.y):
        first_point = None
        row = np.array(topo_data.Z[i,:])
        not_land = np.where(row <= 0)
        land = np.where(row > 0)
        for point in land[0]:
            if topo_data.Z[i, point - 1] > 0 and topo_data.Z[i, point] <= 0:
                first_point = (point, topo_data.x[point], i, lat)
                # print(first_point)
                break
        if first_point:
            # print('first point')
            for point in not_land[0]:
                if point > first_point[0] and topo_data.Z[i, point + 1] <=0:
                    last_point = (point, topo_data.x[point], i, lat)
                    inlet_y.append((first_point, last_point))
                    break
    for p in inlet_y:
            inds.append([p[0][2], p[0][0], p[1][0]])
            for i in p:
                points.append([i[3], i[1]])
    return {
        'inlet_y': inlet_y,
        'inds': inds,
        'points':points}

def search_bathymetry(topo_data, lat_search=False, lon_search=False):
    assert lat_search != lon_search, "pick one of lat or lon"
    if lat_search:
        location_array = topo_data.y
    else:
        location_array = topo_data.x
        
    search_points = []
    points = []
    inds = []
    for i, lat in enumerate(topo_data.y) if lat_search else i, lon in enumerate(topo_data.x):
        first_point = None
        row = np.array(topo_data.Z[i,:]) if lat_search else np.array(topo_data.Z[:, i]) #or vise versa, dunno which is which
        not_land = np.where(row <= 0)
        land = np.where(row > 0)
        for point in land[0]:
            if (lat_search and topo_data.Z[i, point - 1] > 0 and topo_data.Z[i, point] <= 0) or (lon_search and topo_data.Z[point - 1, i] <=0 and topo_data.Z[point, i] > 0):
                first_point = (point, location_array[point], i, location_array[i])

                break
        if first_point:
            for point in not_land[0]:
                if (lat_search and point > first_point[0] and topo_data.Z[i, point + 1] <=0) or (lon_search and point > first_point[0] and topo_data.Z[point +1, i] <= 0):
                    last_point = (point, location_array[point], i, location_array[i])
                    search_points.append((first_point, last_point))
                    break
    for p in inlet_y:
            inds.append([p[0][2], p[0][0], p[1][0]])
            for i in p:
                points.append([i[3], i[1]])
    return {
        'inlet_y': inlet_y,
        'inds': inds,
        'points':points}

def find_points(topo_data):
    return search_bathymetry(topo_data, lat_search=True) # or false, or whatever it is for this case

def find_inlet(topo_data):
    return search_bathymetry(topo_data, lon_search=True) # or false, or whatever it is for this case


def create_static_breach(topo_data, breach_loc):
    for idx in range(len(breach_loc)):
        w = breach_loc.loc[idx]['West']
        e = breach_loc.loc[idx]['East']
        n = breach_loc.loc[idx]['North']
        s = breach_loc.loc[idx]['South']
        depth = breach_loc.loc[idx]['Depth']
        # w = breach_loc[0]
        # e = breach_loc[1]
        # s = breach_loc[2]
        # n = breach_loc[3]
        # depth = breach_loc[4]
        for i, lon in enumerate(topo_data.x):
            if (lon > w) and (lon < e):
                for j, lat in enumerate(topo_data.y):
                    if (lat > s) and (lat < n):
                        if topo_data.Z[j, i] >= 0:
                            topo_data.Z[j,i] = depth
    return topo_data


def close_inlet(topo_data, inlet_loc, name, save_path):
    w = inlet_loc['west']
    e = inlet_loc['east']
    n = inlet_loc['north']
    s = inlet_loc['south']
    for i, lon in enumerate(topo_data.x):
        if (lon > w) and (lon < e):
            for j, lat in enumerate(topo_data.y):
                if (lat > s) and (lat < n):
                    if topo_data.Z[j, i] < 0:
                        topo_data.Z[j, i] = 2.0
    topo_data.save(os.path.join(save_path, f'{name}.nc'))
    return topo_data.Z
    
                                                                                                 
if __name__ == '__main__':
    topo = topotools.Topography('/home/catherinej/bathymetry/moriches.nc', 4)
    filter_region = [-72.885652,-72.634247,40.718299,40.828344] # Moriches Bay, NY
    topo_data = topo.crop(filter_region)
    island_coords = find_island(topo_data)
    masked_data = create_island_mask(topo_data)