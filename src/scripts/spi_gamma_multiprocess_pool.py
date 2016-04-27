from __future__ import division
from datetime import datetime
import indices
import logging
from multiprocessing import Array, Pool, cpu_count
import netCDF4
import netcdf_utils
import numpy as np
import sys
import ctypes
# import scripts_namespace

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------------------------------------------------
def compute_indicator(args):
         
    # extract the arguments
    index = args[0]
    
    # turn the shared array into a numpy array
    data = np.ctypeslib.as_array(shared_array)
    data = data.reshape(data_shape)
                 
    # only process non-empty grid cells, i.e. data array contains at least some non-NaN values
    if (isinstance(data[:, index], np.ma.MaskedArray) and data[:, index].mask.all()) \
        or np.isnan(data[:, index]).all() or (data[:, index] <= 0).all():
              
        pass         
                   
    else:  # we have some valid values to work with
              
        logger.info('Processing latitude: {}'.format(index))
              
        # perform a fitting to gamma     
        fitted_values = indices.spi_gamma(data[:, index],
                                          month_scale, 
                                          valid_min, 
                                          valid_max)
  
        # update the shared array
        data[:, index] = fitted_values

#-----------------------------------------------------------------------------------------------------------------------
def init_process(array, 
                 worker_data_shape, 
                 worker_month_scale, 
                 worker_valid_min, 
                 worker_valid_max):
  
    # put the arguments to the global namespace  
    global shared_array, data_shape, month_scale, valid_min, valid_max
    shared_array = array
    data_shape = worker_data_shape
    month_scale = worker_month_scale
    valid_min = worker_valid_min
    valid_max = worker_valid_max
        
    
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
        
        # log some timing info, used later for elapsed time 
        start_datetime = datetime.now()
        logger.info("Start time:    {}".format(start_datetime, '%x'))
        
        # get the command line arguments
        input_file = sys.argv[1]
        input_var_name = sys.argv[2]
        output_file_base = sys.argv[3]
        month_scale = int(sys.argv[4])

        # create the variable name from the indicator, distribution, and month scale
        variable_name = 'spi_gamma_{}'.format(str(month_scale).zfill(2))

        # open the NetCDF files
        with netCDF4.Dataset(input_file) as input_dataset, \
            netCDF4.Dataset(output_file_base + '_' + variable_name + '.nc', 'w') as output_dataset:
             
            # the valid min and max values (the range within which the resulting SPI values will be clipped)
            valid_min = -3.09
            valid_max = 3.09
                    
            # use NaN as our fill/missing value
            fill_value = np.NaN
        
            # use names for the horizontal and vertical dimensions, this allows for more than just lon/lat 3-D variables
            # TODO write a function that pulls these out from the template NetCDF, recognizing the three current
            # dimensionalities: lon/lat for grids, state/division for climate divisions, and station/None for stations 
            lon_dim_name = 'lon'
            lat_dim_name = 'lat'
             
            # make sure the data matches the dimensions
            time_size = input_dataset.variables['time'].size
            lat_size = input_dataset.variables[lat_dim_name].size
            lon_size = input_dataset.variables[lon_dim_name].size
     
            # create the variable name from the indicator, distribution, and month scale
            variable_name = variable_name + '_{}'.format(str(month_scale).zfill(2))
 
            # make a basic set of variable attributes
            variable_attributes = {'valid_min' : valid_min,
                                   'valid_max' : valid_max,
                                   'long_name' : 'SPI (Gamma), {}-month scale'.format(str(month_scale))}              
 
            # copy the global attributes from the input
            output_dataset.setncatts(input_dataset.__dict__)
                 
            # create the time, x, and y dimensions
            output_dataset.createDimension('time', None)
            output_dataset.createDimension(lon_dim_name, lon_size)
            output_dataset.createDimension(lat_dim_name, lat_size)
             
            # get the appropriate data types to use for the variables based on the values arrays
            time_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables['time'])
            lon_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables[lon_dim_name])
            lat_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables[lat_dim_name])
            data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
             
            # create the variables
            time_variable = output_dataset.createVariable('time', time_dtype, ('time',))
            x_variable = output_dataset.createVariable(lon_dim_name, lon_dtype, (lon_dim_name,))
            y_variable = output_dataset.createVariable(lat_dim_name, lat_dtype, (lat_dim_name,))
            data_variable = output_dataset.createVariable(variable_name, 
                                                          data_dtype, 
                                                          ('time', lon_dim_name, lat_dim_name,), 
                                                          fill_value=fill_value)
             
            # set the variables' attributes
            time_variable.setncatts(input_dataset.variables['time'].__dict__)
            x_variable.setncatts(input_dataset.variables[lon_dim_name].__dict__)
            y_variable.setncatts(input_dataset.variables[lat_dim_name].__dict__)
            data_variable.setncatts(variable_attributes)
             
            # set the coordinate variables' values
            time_variable[:] = input_dataset.variables['time'][:]
            x_variable[:] = input_dataset.variables[lon_dim_name]
            y_variable[:] = input_dataset.variables[lat_dim_name]
          
            # create a shared memory array which can be accessed from within another process
            shared_array = Array(ctypes.c_double, time_size * lat_size, lock=False)
            data_shape = (time_size, lat_size)
                
            # create a processor with a number of worker processes
            number_of_workers = 1 #cpu_count()

            # create a Pool, essentially forking with copies of the shared array going to each pooled/forked process
            pool = Pool(processes=number_of_workers, 
                        initializer=init_process, 
                        initargs=(shared_array, data_shape, month_scale, valid_min, valid_max))
            
            # for each longitude slice
            for lon_index in range(lon_size):
    
                logger.info('\n\nProcessing longitude: {}\n'.format(lon_index))

                # get the shared memory array and convert into a numpy array with proper dimensions
                longitude_array = np.ctypeslib.as_array(shared_array)
                longitude_array = np.reshape(longitude_array, data_shape)
 
                # read the longitude slice into the shared memory array     
                longitude_array[:] = input_dataset.variables[input_var_name][:, lon_index, :]
                
                # set all negative values to np.NaN (our fill value is -999.9 but 
                # is read as a the value rather than being replaced by np.NaN as we'd prefer)
                longitude_array[longitude_array < 0] = np.NaN
                
                # a list of arguments we'll map to the processes of the pool
                arguments_iterable = []

                # loop over each latitude point in the longitude slice
                for lat_index in range(lat_size):
                    
                    # have the processor process the shared array at this index
                    arguments = [lat_index]
#                     arguments = [lat_index, data_shape, month_scale, valid_min, valid_max]
                    arguments_iterable.append(arguments)
                        
                # map the arguments iterable to the compute function, allow the processes to run asynchronously
                pool.map(compute_indicator, arguments_iterable)
    
                # get the longitude slice of fitted values from the shared memory array and convert  
                # into a numpy array with proper dimensions which we can then use to write to NetCDF
                fitted_array = np.ctypeslib.as_array(shared_array)
                fitted_array = np.reshape(fitted_array, (time_size, 1, lat_size))
                                                  
                # write the longitude slice of computed values into the output NetCDF
                output_dataset.variables[variable_name][:, lon_index, :] = fitted_array
    
            # all processes have completed, close the pool
            pool.close()

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))
        
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise

