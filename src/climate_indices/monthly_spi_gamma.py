import indices
import logging
import netCDF4
import numpy as np
import sys

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------------------------------------------------
def find_netcdf_datatype(object):
    
    if isinstance(object, netCDF4.Variable):

        if object.dtype == 'float16':
            netcdf_datatype = 'f2'
        elif object.dtype == 'float32':
            netcdf_datatype = 'f4'
        elif object.dtype == 'float64':
            netcdf_datatype = 'f8'
        elif object.dtype == 'int16':
            netcdf_datatype = 'i2'
        elif object.dtype == 'int32':
            netcdf_datatype = 'i4'
        else:
            raise ValueError('Unsupported data type: {}'.format(object.dtype))
    
    elif isinstance(object, float):

        netcdf_datatype = 'f4'
        
    else:
        raise ValueError('Unsupported argument type: {}'.format(type(object)))
    
    return netcdf_datatype
    
#-----------------------------------------------------------------------------------------------------------------------
def initialize_dataset(file_path,
                       template_dataset,
                       x_dim_name,
                       y_dim_name,
                       data_variable_name,
                       data_variable_long_name,
                       data_variable_attributes,
                       data_fill_value):
    
    # make sure the data matches the dimensions
    time_size = template_dataset.variables['time'].size
    y_size = template_dataset.variables[y_dim_name].size
    x_size = template_dataset.variables[x_dim_name].size
    
    # open the output file for writing, set its dimensions and variables
    netcdf = netCDF4.Dataset(file_path, 'w')

    # copy the global attributes from the template
    netcdf.setncatts(template_dataset.__dict__)
        
    # create the time, x, and y dimensions
    netcdf.createDimension('time', None)
    netcdf.createDimension(x_dim_name, x_size)
    netcdf.createDimension(y_dim_name, y_size)
    
    # get the appropriate data types to use for the variables based on the values arrays
    time_dtype = find_netcdf_datatype(template_dataset.variables['time'])
    x_dtype = find_netcdf_datatype(template_dataset.variables[x_dim_name])
    y_dtype = find_netcdf_datatype(template_dataset.variables[y_dim_name])
    data_dtype = find_netcdf_datatype(data_fill_value)
    
    # create the variables
    time_variable = netcdf.createVariable('time', time_dtype, ('time',))
    x_variable = netcdf.createVariable(x_dim_name, x_dtype, (x_dim_name,))
    y_variable = netcdf.createVariable(y_dim_name, y_dtype, (y_dim_name,))
    data_variable = netcdf.createVariable(data_variable_name, 
                                          data_dtype, 
                                          ('time', x_dim_name, y_dim_name,), 
                                          fill_value=data_fill_value)
    
    # set the variables' attributes
    time_variable.setncatts(template_dataset.variables['time'].__dict__)
    x_variable.setncatts(template_dataset.variables[x_dim_name].__dict__)
    y_variable.setncatts(template_dataset.variables[y_dim_name].__dict__)
    data_variable.setncatts(data_variable_attributes)
    data_variable.setncattr('long_name', data_variable_long_name),

    
    # set the coordinate variables' values
    time_variable[:] = template_dataset.variables['time'][:]
    x_variable[:] = template_dataset.variables[x_dim_name]
    y_variable[:] = template_dataset.variables[y_dim_name]

    return netcdf
    
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    '''
    This program will compute SPI corresponding to a precipitation dataset at a specified month scale. Index values 
    are fitted to the Pearson type III distribution.  
    
    example invocation:
    
    python monthly_spi_gamma.py C:/tmp/nclimgrid/lowres/lowres_nclimgrid_189501_201412_prcp.nc \
                                prcp \
                                C:/tmp/nclimgrid/lowres/python_lowres_nclimgrid_189501_201412_ 
                          
    @param precip_file: precipitation dataset file, in NetCDF format
    @param precip_var_name: name of the precipitation variable within the input NetCDF
    @param output_file_base: base file name for the output file which will result
    '''

    try:
        
        # get the command line arguments
        precip_file = sys.argv[1]
        precip_var_name = sys.argv[2]
        output_file_base = sys.argv[3]

        # the valid min and max values (the range within which the resulting SPEI values will be clipped)
        valid_min = -3.09
        valid_max = 3.09
        
        # open the NetCDF files
        with netCDF4.Dataset(precip_file) as precip_dataset:
            
            # use names for the horizontal and vertical dimensions, this allows for more than just lon/lat 3-D variables
            # TODO write a function that pulls these out from the template NetCDF, recognizing the three current
            # dimensionalities: lon/lat for grids, state/division for climate divisions, and station/None for stations 
            x_dim_name = 'lon'
            y_dim_name = 'lat'
            
            # make a basic set of variable attributes
            variable_attributes = {'valid_min' : valid_min,
                                   'valid_max' : valid_max}              

            # initialize the output NetCDFs for each month scale
            month_scales = [1, 2, 3, 6, 9, 12, 24, 36, 48, 60, 72]
            spi_gamma_datasets = {}
            for month_scale_index, month_scale_var_name in enumerate(month_scales):

                # create the variable name from the indicator, distribution, and month scale
                variable_name_spi_gamma = 'spi_gamma_{}'.format(str(month_scale_var_name).zfill(2))

                # get the output files for each variable
                # initialize the output NetCDFs with the same dimensionality, coordinates, etc. as the input precipitation NetCDF
                spi_gamma_dataset = initialize_dataset(output_file_base + variable_name_spi_gamma + '.nc',
                                                       precip_dataset,
                                                       x_dim_name,
                                                       y_dim_name,
                                                       variable_name_spi_gamma,
                                                       'SPI (Gamma), {}-month scale'.format(str(month_scale_var_name)),
                                                       variable_attributes,
                                                       np.nan)
            
                # add to the list of month scale datasets
                spi_gamma_datasets[variable_name_spi_gamma] = spi_gamma_dataset
                
            # loop over the grid cells
            for x in range(precip_dataset.variables[x_dim_name].size):
                for y in range(precip_dataset.variables[y_dim_name].size):

                    logger.info('Processing x/y {}/{}'.format(x, y))
                    
                    # slice out the period of record for the x/y point
                    precip_data = precip_dataset.variables[precip_var_name][:, x, y]
                                           
                    # only process non-empty grid cells, i.e. the data array contains at least some non-NaN values
                    if (isinstance(precip_data, np.ma.MaskedArray)) and precip_data.mask.all():
                        
                            continue
                    
                    else:  # we have some valid values to work with
                        
                        for month_scale_index, month_scale_var_name in enumerate(sorted(spi_gamma_datasets.keys())):

                            # perform the SPI computation (fit to the Gamma distribution) and assign the values into the dataset
                            spi_gamma_datasets[month_scale_var_name].variables[month_scale_var_name][:, x, y] = \
                                indices.spi_gamma(precip_data,
                                                  month_scales[month_scale_index], 
                                                  valid_min, 
                                                  valid_max)
            
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise
