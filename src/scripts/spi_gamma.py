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

#-----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    '''
    This program will compute SPI corresponding to a precipitation dataset at a specified month scale. Index values 
    are fitted to the Gamma distribution.  
    
    example invocation:
    
    python spi_gamma.py C:/tmp/nclimgrid/lowres/lowres_nclimgrid_189501_201412_prcp.nc \
                        prcp \
                        C:/tmp/nclimgrid/lowres/python_lowres_nclimgrid_189501_201412_ \ 
                        3
                        
    @param precip_file: precipitation dataset file, in NetCDF format
    @param precip_var_name: name of the precipitation variable within the input NetCDF
    @param output_file_base: base file name for the output file which will result
    @param month_scale: month scale over which the SPI will be computed   
    '''

    try:
        
        # get the command line arguments
        precip_file = sys.argv[1]
        precip_var_name = sys.argv[2]
        output_file_base = sys.argv[3]
        month_scale = int(sys.argv[4])

        # the valid min and max values (the range within which the resulting SPI values will be clipped)
        valid_min = -3.09
        valid_max = 3.09
             
        # open the NetCDF files
        with netCDF4.Dataset(precip_file) as precip_dataset:
            
            # use names for the horizontal and vertical dimensions, this allows for more than just lon/lat 3-D variables
            # TODO write a function that pulls these out from the template NetCDF, recognizing the three current
            # dimensionalities: lon/lat for grids, state/division for climate divisions, and station/None for stations 
            x_dim_name = 'lon'
            y_dim_name = 'lat'
            
            # create the variable name from the indicator, distribution, and month scale
            variable_name = 'spi_gamma_{}'.format(str(month_scale).zfill(2))

            # make a basic set of variable attributes
            variable_attributes = {'valid_min' : valid_min,
                                   'valid_max' : valid_max,
                                   'long_name' : 'SPI (Gamma), {}-month scale'.format(str(month_scale))}              

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
                                           
                    # only process non-empty grid cells, i.e. data array contains at least some non-NaN values
                    if (isinstance(precip_data, np.ma.MaskedArray)) and precip_data.mask.all():
                        
                            continue
                    
                    else:  # we have some valid values to work with
                        
                        # perform the SPI computation (fit to the Gamma distribution) and assign the values into the dataset
                        data = indices.spi_gamma(precip_data, 
                                                 month_scale, 
                                                 valid_min, 
                                                 valid_max)
                        output_dataset.variables[variable_name][:, x, y] = data
#                         output_dataset.variables[variable_name][:, x, y] = indices.spi_gamma(precip_data, 
#                                                                                              month_scale, 
#                                                                                              valid_min, 
#                                                                                              valid_max)
            
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise
