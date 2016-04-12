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
    
    # set the coordinate variables' values
    time_variable[:] = template_dataset.variables['time'][:]
    x_variable[:] = template_dataset.variables[x_dim_name]
    y_variable[:] = template_dataset.variables[y_dim_name]

    return netcdf
    
#----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    
    '''
    This program will compute SPI corresponding to a precipitation dataset at a specified month scale. Index values 
    are fitted to the Gamma distribution.  
    
    example invocation:
    
    python spei_pearson.py C:/tmp/nclimgrid/lowres/lowres_nclimgrid_189501_201412_prcp.nc \
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
            output_dataset = initialize_dataset(output_file_base + variable_name + '.nc',
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
