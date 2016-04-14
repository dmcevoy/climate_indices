from __future__ import division
import indices
import logging
import netCDF4
import netcdf_utils
import numpy as np
import sys

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    '''
    This program will compute SPEI corresponding to a precipitation dataset at a specified month scale. Index values 
    are fitted to the Gamma distribution.  
    
    example invocation:
    
    python spei_gamma.py C:/tmp/nclimgrid/lowres/lowres_nclimgrid_189501_201412_prcp.nc \
                         prcp \
                         C:/tmp/nclimgrid/lowres/lowres_nclimgrid_189501_201412_tavg.nc \
                         tavg \
                         C:/tmp/nclimgrid/lowres/python_lowres_nclimgrid_189501_201412_ \ 
                         3
                          
    @param precip_file: precipitation dataset file, in NetCDF format
    @param precip_var_name: name of the precipitation variable within the input NetCDF
    @param temp_file: temperature dataset file, in NetCDF format
    @param temp_var_name: name of the temperature variable within the input NetCDF
    @param output_file_base: base file name for the output file which will result
    @param month_scale: month scale over which the SPI will be computed   
    '''

    try:
        
        # get the command line arguments
        precip_file = sys.argv[1]
        precip_var_name = sys.argv[2]
        temp_file = sys.argv[3]
        temp_var_name = sys.argv[4]
        output_file_base = sys.argv[5]
        month_scale = int(sys.argv[6])

        # the valid min and max values (the range within which the resulting SPEI values will be clipped)
        valid_min = -3.09
        valid_max = 3.09
        
        # offset we add to the (P - PE values) in order to bring all values into the positive range     
        p_minus_pe_offset = 1000.0

        # open the NetCDF files
        with netCDF4.Dataset(precip_file) as precip_dataset, netCDF4.Dataset(temp_file) as temp_dataset:
            
            # use names for the horizontal and vertical dimensions, this allows for more than just lon/lat 3-D variables
            # TODO write a function that pulls these out from the template NetCDF, recognizing the three current
            # dimensionalities: lon/lat for grids, state/division for climate divisions, and station/None for stations 
            x_dim_name = 'lon'
            y_dim_name = 'lat'
            
            # find the start and end date
            start_date = netCDF4.num2date(precip_dataset.variables['time'][0], precip_dataset.variables['time'].units)
            end_date = netCDF4.num2date(precip_dataset.variables['time'][-1], precip_dataset.variables['time'].units)
            total_data_years = end_date.year - start_date.year + 1
            
            # create the variable name from the indicator, distribution, and month scale
            variable_name = 'spei_gamma_{}'.format(str(month_scale).zfill(2))

            # make a basic set of variable attributes
            variable_attributes = {'valid_min' : valid_min,
                                   'valid_max' : valid_max,
                                   'long_name' : 'SPEI (Gamma), {}-month scale'.format(str(month_scale))}              

            # initialize the output NetCDF with the same dimensionality, coordinates, etc. as the input precipitation NetCDF
            output_dataset = netcdf_utils.initialize_dataset(output_file_base + variable_name + '.nc',
                                                             precip_dataset,
                                                             x_dim_name,
                                                             y_dim_name,
                                                             variable_name,
                                                             variable_attributes,
                                                             np.nan)
            
            # loop over the grid cells
            for x in range(precip_dataset.variables[x_dim_name].size):
                for y in range(precip_dataset.variables[y_dim_name].size):
                    
                    logger.info('Processing x/y {}/{}'.format(x, y))
                    
                    # slice out the period of record for the x/y point
                    precip_data = precip_dataset.variables[precip_var_name][:, x, y]
                    temp_data = temp_dataset.variables[temp_var_name][:, x, y]
                                           
                    # only process non-empty grid cells, i.e. the data array contains at least some non-NaN values
                    if (isinstance(precip_data, np.ma.MaskedArray)) and precip_data.mask.all():
                        
                            continue
                    
                    else:  # we have some valid values to work with
                        
                        # perform the SPEI computation (fit to the Pearson distribution) and assign the values into the dataset
                        output_dataset.variables[variable_name][:, x, y] = indices.spei_gamma(precip_data,
                                                                                              temp_data,
                                                                                              start_date.year,
                                                                                              precip_dataset.variables[y_dim_name][y],
                                                                                              month_scale, 
                                                                                              valid_min, 
                                                                                              valid_max)
                        
            
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise
