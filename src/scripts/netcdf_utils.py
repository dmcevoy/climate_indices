from __future__ import division
import logging
import netCDF4

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
def find_netcdf_datatype(data_object):
    
    if isinstance(data_object, netCDF4.Variable):

        if data_object.dtype == 'float16':
            netcdf_datatype = 'f2'
        elif data_object.dtype == 'float32':
            netcdf_datatype = 'f4'
        elif data_object.dtype == 'float64':
            netcdf_datatype = 'f8'
        elif data_object.dtype == 'int16':
            netcdf_datatype = 'i2'
        elif data_object.dtype == 'int32':
            netcdf_datatype = 'i4'
        else:
            raise ValueError('Unsupported data type: {}'.format(data_object.dtype))
    
    elif isinstance(data_object, float):

        netcdf_datatype = 'f4'
        
    else:
        raise ValueError('Unsupported argument type: {}'.format(type(data_object)))
    
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
    
