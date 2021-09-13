"""
Plot Mobile Mesonet Data

Plot mobile mesonet data on top of radar reflectivity (from either a mobile radar or NEXRAD). 
Before plotting, make sure to smooth the mobile mesonet data using Paul Markowski's mmsmooth 
program (See Markowski et al. 2012a section 2b for more information about mobile mesonets).

Shawn Murdzek
smurdzek@psu.edu
Date Created: March 20, 2018
"""


#---------------------------------------------------------------------------------------------------
# Import Modules
#---------------------------------------------------------------------------------------------------

import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import metpy
from metpy.plots import StationPlot
from metpy.units import units
from metpy.calc import reduce_point_density
import metpy.calc.thermo as mct
import numpy.ma as ma
import pandas as pd
import pyart
import cartopy.crs as ccrs

v_metpy = int(metpy.__version__[0])


#---------------------------------------------------------------------------------------------------
# Define Input Parameters
#---------------------------------------------------------------------------------------------------

# Mobile mesonet file directory

mm_dir = './20100514/'

# Mobile mesonet file names

mm_files = ['p1_100514.qcd',
            'p2_100514.qcd',
            'p3_100514.qcd',
            'p4_100514.qcd',
            'p5_100514.qcd',
            'p7_100514.qcd']

# Initial day mobile mesonet is collecting data

init_day = dt.datetime(2010, 5, 14)

# Analysis Times. Should correspond to the time that the radar scan is valid

anal_t = [dt.datetime(2010, 5, 14, 18, 28, 0)]

# For 'cart', the (lat, lon) coordinate of the origin must be specified (in deg N and deg E). The 
# conversion from (lat, lon) to (x, y) coordinates uses a flat-Earth approximation

coords = 'latlon'
origin_lon = -104.35
origin_lat = 40.1

# Lower left corner and upper right corner of grid
# For coords == 'latlon', units are deg N and deg E
# For coords = 'cart', units are km

x_min = -102.8
x_max = -102.45
y_min = 31.75
y_max = 32.05

# Option to show grid

show_grid = True

# Define a time period to use to define a background thermodynamic field (in decimal hours). Add 24
# to the decimal time if it is on the next day (e.g., 2.5 would be 26.5). Set use_base to False to 
# plot raw thermodynamic quantities instead of perturbation quantities

use_base = False
avg_start_t = 21 + (10.0 / 60.0)
avg_end_t = 21.5

# Maximum time from anal_time to plot mobile mesonet probes (seconds)

dtmax = 300.0

# Storm motion vector (m/s) for time-to-space conversion and plotting storm-relative winds in
# station models

cx = 5.0
cy = 4.0

# Directory containing radar data

rad_dir = './20100514/'

# Radar data file names (must have an equal number of radar files and analysis times)

rad_files = ['KMAF20100514_182800_V03.gz']

# Radar data type (options: 'nexrad' or 'dorade'). ref_field is only needed for the 'dorade' option

rad_type = 'nexrad'
ref_field = 'REF'

# Colormap, plotting range (dBZ), and opacity for radar reflectivity. I personally like gist_gray or 
# pyart.graph.cm_colorblind.HomeyerRainbow for colormaps

ref_cmap = pyart.graph.cm_colorblind.HomeyerRainbow
ref_min = 0.0 
ref_max = 70.0
ref_opacity = 1.0

# Option to color-code mobile mesonet station models and variable used for color-coding

color_code = False
color_var = 'THETAE'

# Minimum and maximum values for color_var

var_min = 332.0
var_max = 342.0

# Colorbar label for color_var

cbar_label = r"$\theta_{e}$ (K)"

# Option to plot the numeric values of the mobile mesonet observations directly next to the station
# plots as well as which variables to plot northwest, northeast, southeast, and southwest of the
# station model. To not plot a variable in a position, put None. fontsize controls the size of the
# numeric values that are plotted.

numeric_val = True
nw_var = 'THETAV'
ne_var = None
se_var = None
sw_var = None
fontsize = 8

# Option to plot station models with wind barbs (wind keyword). The two options for frame are 
# 'storm-relative' (subtracts off cx and cy) or 'fixed' (plots raw mobile mesonet wind data as is).
# Spacing refers to the minimum distance between plotted station models 
# (in deg for coords == 'latlon', in km for coords = 'cart')

wind = True
frame = 'storm-relative'
spacing = 0.02

# Switch to filter out stationary mobile mesonet data. In 2009, data collected from stationary mobile 
# mesonets was questionable, but this problem was fixed in 2010 (see README_NSSL-PSU_mobilemesonet)

filter_stat = True

# Directory to save plots in and suffix for output file (e.g., '.png', '.pdf')

save_dir = './'
f_suffix = '.png'


#---------------------------------------------------------------------------------------------------
# Define Functions
#---------------------------------------------------------------------------------------------------

def ll_to_xy(lat, lon, lat0, lon0):
    """
    Function to convert (lat, lon) coordinates to (x, y) coordinates using (lat0, lon0) as the
    origin. This function is based on the ll_to_xy subroutine in util.f.Barnes.backinterp, which is
    part of David Dowell's OBAN/dual-Doppler software package. The conversion uses a flat Earth
    approximation.
    Inputs:
        lat = Latitudes (deg N)
        lon = Longitudes (deg E)
        lat0 = Latitude of origin (deg N)
        lon0 = Longitude of origin (deg E)
    Outputs:
        x = X-coordinates (km)
        y = Y-coordinates (km)
    """

    # Define radius of the Earth in km

    R = 6367.0

    # Convert latitudes and longitudes to radians

    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    lat0 = np.deg2rad(lat0)
    lon0 = np.deg2rad(lon0)

    # Compute x and y

    x = R * np.cos(0.5 * (lat0 + lat)) * (lon - lon0)
    y = R * (lat - lat0)

    return x, y


#---------------------------------------------------------------------------------------------------
# Read in Mobile Mesonet Data
#---------------------------------------------------------------------------------------------------

# Create dictionary to store mobile mesonet DataFrames in

mm_data = {}

# Define column headers in mobile mesonet CSV files. See README_NSSL-PSU_mobilemesonet for 
# descriptions

cols = ['id', 'time', 'lat', 'lon', 'Tfast', 'Tslow', 'RH', 'p', 'dir', 'spd', 'qc1', 'qc2', 'qc3',
        'qc4']

# Read in data from each mobile mesonet CSV file

for f in mm_files:
    
    dataframe = pd.read_csv(mm_dir + f, header=None, names=cols, delim_whitespace=True)
    
    # Mask data using the quality control (QC) flags. Mask all variables if qc1 (data logger issue)
    # or qc3 (sanity check) are True.
    
    wind_vars = ['dir', 'spd']
    thermo_vars = ['Tfast', 'Tslow', 'RH', 'p']
    all_vars = thermo_vars + wind_vars
    
    dataframe.loc[:, all_vars] = dataframe.loc[:, all_vars].mask(dataframe['qc1'] == 1)
    dataframe.loc[:, all_vars] = dataframe.loc[:, all_vars].mask(dataframe['qc3'] == 1)

    # Drop rows with NaN values (so rows with either qc1 or qc3 equal to 1)

    dataframe.dropna(inplace=True)
    
    # Mask missing (-999.0) values
    
    dataframe.mask(dataframe == -999.0, inplace=True)

    # Convert Cartesian coordinates for mobile mesonet probes

    dataframe['lon'] = -1 * dataframe['lon']
    dataframe['x'], dataframe['y'] = ll_to_xy(dataframe['lat'], dataframe['lon'], origin_lat, 
                                              origin_lon)
    
    # Compute the x and y components of the wind (denoted u and v)
    
    dataframe['u'] = -(dataframe['spd'] * np.sin(np.deg2rad(dataframe['dir'])))
    dataframe['v'] = -(dataframe['spd'] * np.cos(np.deg2rad(dataframe['dir'])))
    if (frame == 'storm-relative'):
        dataframe['u'] = dataframe['u'] - cx
        dataframe['v'] = dataframe['v'] - cy
        
    wind_vars = wind_vars + ['u', 'v']

    # Mask wind variables if qc2 (vehicle acceleration) is True

    dataframe.loc[:, wind_vars] = dataframe.loc[:, wind_vars].mask(dataframe['qc2'] == 1)
    
    # Mask thermodynamic variables if qc4 (stationary vehicle) is True owing to insufficient T and 
    # RH sensor aspiration. Note that this was not an issue during the 2010 field phase of VORTEX2 
    # owing to the new "U tube" housing for the T and RH sensors.
    
    if filter_stat:
        dataframe.loc[:, thermo_vars] = dataframe.loc[:, thermo_vars].mask(dataframe['qc4'] == 1)
    
    # Determine when the decimal times stop increasing (this corresponds to the day change) and add
    # 24 hours to all times after the observations cross into the next day
    
    day_change_ind = np.where((dataframe.iloc[1:, 1].values - dataframe.iloc[:-1, 1].values) < 0)[0]
    
    for i in day_change_ind:
        
        dataframe.iloc[i+1:, 1] += 24
   
    # Compute derived thermodynamic quantities using MetPy. To this, units must be added using Pint
    # to make MetPy happy

    T = dataframe['Tfast'].values * units.degC
    RH = dataframe['RH'].values * units.percent
    pres = dataframe['p'].values * units.millibars
    if v_metpy < 1:
        mix = mct.mixing_ratio_from_relative_humidity(RH, T, pres)
        Td = mct.dewpoint_rh(T, RH)
    else:
        mix = mct.mixing_ratio_from_relative_humidity(pres, T, RH)
        Td = mct.dewpoint_from_relative_humidity(T, RH)

    dataframe['THETAV'] = mct.virtual_potential_temperature(pres, T, mix).magnitude
    dataframe['THETAE'] = mct.equivalent_potential_temperature(pres, T, Td).magnitude
    dataframe['THETA'] = mct.potential_temperature(pres, T).magnitude
    dataframe['mixing'] = mix.magnitude
    dataframe['Td'] = Td.magnitude
 
    # Save DataFrame to dictionary
    
    mm_data[dataframe['id'].iloc[0]] = dataframe

# Define dictionary for units

mm_units = {'id': None, 'time':'decimal hours', 'lat':'deg N', 'lon':'deg E', 'Tfast':'deg C', 
            'Tslow':'deg C', 'RH':'%', 'p':'mb', 'dir':'deg', 'spd':'m/s', 'qc1':None, 'qc2':None,
            'qc3':None, 'qc4':None, 'u':'m/s', 'v':'m/s', 'THETAV':'K', 'THETAE':'K', 'THETA':'K',
            'mixing':'kg/kg', 'Td':'deg C'}


#---------------------------------------------------------------------------------------------------
# Determine Base State Thermodynamic Values
#---------------------------------------------------------------------------------------------------

if use_base:

    # Compute average thermodynamic quantities

    thermo_base = pd.DataFrame()
    thermo_vars = thermo_vars + ['THETAV', 'THETAE', 'THETA', 'mixing', 'Td']    

    for var in thermo_vars:
        
        var_sum = 0
        n_pts = 0
        
        for k in mm_data.keys():
            
            series = mm_data[k][var].loc[(mm_data[k].time > avg_start_t) & 
                                         (mm_data[k].time < avg_end_t)]
            var_sum += series.sum()
            n_pts += np.sum(~np.isnan(series))
        
        thermo_base[var] = np.array([var_sum / n_pts])

        # Subtract base state to yield perturbation thermodynamic quantities
        
        for k in mm_data.keys():
            
            mm_data[k][var] = mm_data[k][var] - thermo_base[var].values


#---------------------------------------------------------------------------------------------------
# Create Mobile Mesonet Plots
#---------------------------------------------------------------------------------------------------

col_var_max = []
col_var_min = [] 

for t, rad_f in zip(anal_t, rad_files):
    
    t_dec =  (t - init_day).total_seconds() / 3600.0
    dt_dec = dtmax / 3600.0
    
    if rad_type == 'nexrad':
        
        # Read in NEXRAD Level 2 radar file
        
        radar = pyart.io.read_nexrad_archive(rad_dir + rad_f)
        scan = radar.extract_sweeps([0])
        ref = scan.fields['reflectivity']['data']

    elif rad_type == 'dorade':

        # Read in DORADE radar file

        radar = pyart.aux_io.read_radx(rad_dir + rad_f)
        scan = radar
        ref = scan.fields[ref_field]['data']

    else:

        print("Radar option not valid. Try 'nexrad' or 'dorade'")
        break

    # Extract gate latitude and longitudes, reflectivities, and mean elevation angle
        
    gate_lat = scan.gate_latitude['data']
    gate_lon = scan.gate_longitude['data']
    gate_x = scan.gate_x['data'] * 0.001
    gate_y = scan.gate_y['data'] * 0.001
    avg_el = np.mean(scan.elevation['data'])
        
    # Mask reflectivity values below ref_min for clarity
        
    ref = ma.masked_array(ref, mask=(ref < ref_min))
    
    # Create figure and plotting axis and plot radar reflectivity
    
    fig = plt.figure(figsize=(10, 10))
    ref_levels = np.arange(ref_min, ref_max, 5)

    if coords == 'latlon':
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
        ref_cax = ax.contourf(gate_lon, gate_lat, ref, ref_levels, cmap=ref_cmap, vmin=ref_min,
                              vmax=ref_max, transform=ccrs.PlateCarree(), alpha=ref_opacity)
    elif coords == 'cart':
        ax = fig.add_subplot(1, 1, 1)
        rad_x, rad_y = ll_to_xy(scan.latitude['data'][0], scan.longitude['data'][0], origin_lat, 
                                origin_lon)
        ref_cax = ax.contourf(gate_x + rad_x, gate_y + rad_y, ref, ref_levels, cmap=ref_cmap, 
                              vmin=ref_min, vmax=ref_max, alpha=ref_opacity)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.0, top=0.95)  
 
    # Eliminate white lines between contours

    for c in ref_cax.collections:
        c.set_edgecolor('face')

    # Concatenate all mobile mesonet obs for this time period into one DataFrame
    
    all_mm_df = pd.DataFrame(columns=cols)
    
    for k in mm_data.keys():
        all_mm_df = pd.concat([all_mm_df, mm_data[k].loc[(mm_data[k].time > (t_dec - dt_dec)) &
                                                         (mm_data[k].time < (t_dec + dt_dec))]],
                              ignore_index=True, sort=False)
    
    # Perform time-to-space conversion (see Markowski et al. 2002). To do the conversion when 
    # coords == 'latlon', cx and cy will be converted from m/s to deg/s. R is the radius of the 
    # Earth in meters

    delta_t = (all_mm_df.time.values - t_dec) * 3600.0
    R = 6.371e6

    if coords == 'latlon':
        mm_y = all_mm_df.lat.values
        mm_x = all_mm_df.lon.values
        mm_y_adjust = mm_y - (cy * (360.0 / (2.0 * np.pi * R)) * delta_t)
        mm_x_adjust = mm_x - (cx * (360.0 / (2.0 * np.pi * R * np.cos(np.deg2rad(mm_y)))) * delta_t)
        proj = ccrs.PlateCarree()
    elif coords == 'cart':
        mm_x = all_mm_df.x.values
        mm_y = all_mm_df.y.values
        mm_x_adjust = mm_x - (cx * delta_t / 1000.0)
        mm_y_adjust = mm_y - (cy * delta_t / 1000.0)
        proj = None

    # Thin observations, giving priority to those observations closer to the analysis time

    pts = np.transpose(np.array([mm_x_adjust, mm_y_adjust]))
    priority = 100.0 - np.abs(all_mm_df.time.values - t_dec)
    ind = reduce_point_density(pts, spacing, priority=priority)

    # The plot_barb method cannot plot masked arrays, so missing wind measurements must be manually
    # removed by creating a new array, wind_ind.
    # Note: Missing values of the color variable, as well as points that failed QC, are correctly
    # removed by the scatterplot function
    
    wind_ind = np.logical_and(ind, ~np.isnan(all_mm_df.u.values))

    # Plot mobile mesonet winds, if desired. Winds are multipled by two so that way a full (half) 
    # barb is 5 (2.5) m/s. The default is for one barb to be 10 m/s

    if wind:
        statplt_wind = StationPlot(ax, mm_x_adjust[wind_ind], mm_y_adjust[wind_ind],
                                   transform=proj)
        statplt_wind.plot_barb(all_mm_df.u.values[wind_ind] * 2, all_mm_df.v.values[wind_ind] * 2)

    # Plot station model center

    if color_code:

        if coords == 'latlon':
            cax = ax.scatter(mm_x_adjust[ind], mm_y_adjust[ind], c=all_mm_df[color_var].values[ind],
                             vmin=var_min, vmax=var_max, edgecolor='k', s=50, cmap='plasma',
                             transform=proj)
        elif coords == 'cart':
            cax = ax.scatter(mm_x_adjust[ind], mm_y_adjust[ind], c=all_mm_df[color_var].values[ind], 
                             vmin=var_min, vmax=var_max, edgecolor='k', s=50, cmap='plasma')

        # Save maximum values for the color_var

        col_var_max.append(np.amax(all_mm_df[color_var].values))
        col_var_min.append(np.amin(all_mm_df[color_var].values))

        # Create colorbar

        cbar = fig.colorbar(cax, ax=ax, orientation='horizontal', pad=0.00, aspect=30)
        cbar.set_label(cbar_label, size=16)

    else:

        if coords == 'latlon':
            ax.scatter(mm_x_adjust[ind], mm_y_adjust[ind], c='k', s=20, transform=proj)
        elif coords == 'cart':
            ax.scatter(mm_x_adjust[ind], mm_y_adjust[ind], c='k', s=20)

    # Plot numeric values of mobile mesonet observations, if desired

    if numeric_val:

        # Create dictionary to save variable names and locations for annotation later

        num_val_names = {}

        statplt = StationPlot(ax, mm_x_adjust[ind], mm_y_adjust[ind], transform=proj, fontsize=10)

        if nw_var != None:

            statplt.plot_parameter('NW', all_mm_df[nw_var].values[ind], formatter='.1f', 
                                   fontsize=fontsize)
            num_val_names['NW'] = nw_var
        
        if ne_var != None:

            statplt.plot_parameter('NE', all_mm_df[ne_var].values[ind], formatter='.1f', 
                                   fontsize=fontsize)
            num_val_names['NE'] = ne_var

        if se_var != None:

            statplt.plot_parameter('SE', all_mm_df[se_var].values[ind], formatter='.1f', 
                                   fontsize=fontsize)
            num_val_names['SE'] = se_var

        if sw_var != None:

            statplt.plot_parameter('SW', all_mm_df[sw_var].values[ind], formatter='.1f', 
                                   fontsize=fontsize)
            num_val_names['SW'] = sw_var

    # Set plotting domain limits and add x and y tick

    if coords =='latlon':
        ax.set_extent([x_min, x_max, y_min, y_max])

        # Set aspect ratio such that there is no deformation of the plotting window owing to the
        # fact that 1 deg latitude != 1 deg longitude

        width = (R * np.cos(0.5 * (np.deg2rad(y_max) + np.deg2rad(y_min))) * 
                 (np.deg2rad(x_max) - np.deg2rad(x_min)))
        height = R * (np.deg2rad(y_max) - np.deg2rad(y_min))
        ax.set_aspect(height / width)

        if show_grid:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', alpha=0.3)
        else:
            gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, color='black', alpha=0.0)
        gl.xlabels_top = False
        gl.ylabels_right = False
    elif coords == 'cart':
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        if show_grid:
            ax.grid(color='k')

    # Configure reflectivity colorbar

    ref_cbar = fig.colorbar(ref_cax, ax=ax, orientation='horizontal', pad=0.05, aspect=30)
    ref_cbar.set_label(r'%.2f$^{\circ}$ Reflectivity (dBZ)' % avg_el, size=16)

    # Add title and annotation

    plt.title('%s UTC, %.1f sec Radius' % (t.strftime('%Y%m%d %H%M:%S'), dtmax), size=20)
    
    if use_base:
        fig.text(0.025, 0.015, ('Thermodynamic Base State Time: %.5f to %.5f' % 
                               (avg_start_t, avg_end_t)), size=12)

    if numeric_val:

        model_key = 'Station Plot Key:'

        for k in num_val_names.keys():

            model_key = model_key + (' %s = %s (%s);' % (k, num_val_names[k], 
                                                         mm_units[num_val_names[k]]))

        fig.text(0.025, 0.035, model_key, size=12)

    # Save and close figure. Figures are saved as PNGs because it takes a lot of time to save
    # radar images as PDFs.

    plt.savefig('%s%s%s' % (save_dir, t.strftime('%Y%m%d_%H%M%S'), f_suffix))
    plt.close(fig)

# Print min and max values of color variable

if color_code:

    print()
    print('Maximum Color Variable Value =', max(col_var_max))
    print('Minimum Color Variable Value =', min(col_var_min))
    print()


"""
End plot_mobile_mesonets.py
"""
