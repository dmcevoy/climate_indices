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

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------------------------------------------------
def compute_worker(args):
         
    # extract the arguments
    lat_index = args[0]
    
    # turn the shared array into a numpy array
    data = np.ctypeslib.as_array(shared_array)
    data = data.reshape(data_shape)
                 
    # data now expected to be in shape: (indicators, distributions, month_scales, times, lats)
    #
    # with indicator (spi: 0,  spei: 1)
    #      distribution (gamma: 0,  pearson: 1)
    #      month_scales (0, month_scales)
    #
    # with data[0, 0, 0] indicating the longitude slice with shape: (times, lats) with values for precipitation 
    # with data[1, 0, 0] indicating the longitude slice with shape: (times, lats) with values for temperature 
    
    # only process non-empty grid cells, i.e. data array contains at least some non-NaN values
    if (isinstance(data[0, 0, 0, :, lat_index], np.ma.MaskedArray) and data[0, 0, 0, :, lat_index].mask.all()) \
        or np.isnan(data[0, 0, 0, :, lat_index]).all() or (data[0, 0, 0, :, lat_index] <= 0).all():
              
        pass         
                   
    else:  # we have some valid values to work with
              
        logger.info('Processing latitude: {}'.format(lat_index))
              
        for month_index, month_scale in enumerate(month_scales):
            
            # only process month scales after 0 since month_scale = 0 is reserved for the input data 
            if month_index > 0:
                
                # loop over all specified indicators
                for indicator in indicators:

                    # loop over all specified distributions
                    for distribution in distributions:
                        
                        if indicator == 'spi':
                            
                            if distribution == 'gamma':
                                    # perform a fitting to gamma     
                                    data[0, 0, month_index, :, lat_index] = indices.spi_gamma(data[0, 0, 0, :, lat_index],
                                                                                              month_scale, 
                                                                                              valid_min, 
                                                                                              valid_max)
                            elif distribution == 'pearson':
                                    # perform a fitting to Pearson type III     
                                    data[0, 1, month_index, :, lat_index] = indices.spi_pearson(data[0, 0, 0, :, lat_index], 
                                                                                                month_scale, 
                                                                                                valid_min, 
                                                                                                valid_max, 
                                                                                                data_start_year, 
                                                                                                data_end_year, 
                                                                                                calibration_start_year, 
                                                                                                calibration_end_year)
    
                        elif indicator == 'spei':
                            
                            if distribution == 'gamma':
                                    # perform a fitting to gamma     
                                    data[1, 0, month_index, :, lat_index] = indices.spei_gamma(data[0, 0, 0, :, lat_index],
                                                                                               data[0, 0, 1, :, lat_index],
                                                                                               data_start_year,
                                                                                               lats_array[lat_index],
                                                                                               month_scale, 
                                                                                               valid_min, 
                                                                                               valid_max)
                            elif distribution == 'pearson':
                                    # perform a fitting to Pearson type III     
                                    data[1, 1, month_index, :, lat_index] = indices.spei_pearson(data[0, 0, 0, :, lat_index],
                                                                                                 data[0, 0, 1, :, lat_index],
                                                                                                 month_scale, 
                                                                                                 lats_array[lat_index],
                                                                                                 valid_min, 
                                                                                                 valid_max,
                                                                                                 data_start_year,
                                                                                                 data_end_year,
                                                                                                 calibration_start_year, 
                                                                                                 calibration_end_year)
                                    
                            else:
                                raise ValueError('Invalid distribution specified: {}'.format(distribution))
                        else:
                            raise ValueError('Invalid indicator specified: {}'.format(indicator))
                        
#-----------------------------------------------------------------------------------------------------------------------
def init_process(array, 
                 shared_lats_array,
                 worker_data_shape,
                 worker_indicators, 
                 worker_distributions,
                 worker_month_scales, 
                 worker_valid_min, 
                 worker_valid_max,
                 worker_data_start_year, 
                 worker_data_end_year, 
                 worker_calibration_start_year, 
                 worker_calibration_end_year):
  
    # put the arguments to the global namespace  
    global shared_array, lats_array, data_shape, month_scales, valid_min, valid_max, distributions, indicators, data_start_year, data_end_year, calibration_start_year, calibration_end_year
    
    shared_array = array
    lats_array = shared_lats_array
    data_shape = worker_data_shape
    month_scales = worker_month_scales
    valid_min = worker_valid_min
    valid_max = worker_valid_max
    distributions = worker_distributions
    indicators = worker_indicators
    data_start_year = worker_data_start_year 
    data_end_year = worker_data_end_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year

#-----------------------------------------------------------------------------------------------------------------------
def initialize_dataset(output_file,
                       template_dataset,
                       lon_dim_name,
                       lat_dim_name,
                       fill_value,
                       variable_name,
                       valid_min,
                       valid_max):

    output_dataset = netCDF4.Dataset(output_file_base + '_' + variable_name + '.nc', 'w')

    # make sure the data matches the dimensions
    time_size = precip_dataset.variables['time'].size
    lat_size = precip_dataset.variables[lat_dim_name].size
    lon_size = precip_dataset.variables[lon_dim_name].size

    # make a basic set of variable attributes
    variable_attributes = {'valid_min' : valid_min,
                           'valid_max' : valid_max}              
  
    # copy the global attributes from the input
    output_dataset.setncatts(template_dataset.__dict__)
          
    # create the time, x, and y dimensions
    output_dataset.createDimension('time', None)
    output_dataset.createDimension(lon_dim_name, lon_size)
    output_dataset.createDimension(lat_dim_name, lat_size)
      
    # get the appropriate data types to use for the variables based on the values arrays
    time_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables['time'])
    lon_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables[lon_dim_name])
    lat_dtype = netcdf_utils.find_netcdf_datatype(template_dataset.variables[lat_dim_name])
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
    time_variable.setncatts(template_dataset.variables['time'].__dict__)
    x_variable.setncatts(template_dataset.variables[lon_dim_name].__dict__)
    y_variable.setncatts(template_dataset.variables[lat_dim_name].__dict__)
    data_variable.setncatts(variable_attributes)
      
    # set the coordinate variables' values
    time_variable[:] = template_dataset.variables['time'][:]
    x_variable[:] = template_dataset.variables[lon_dim_name]
    y_variable[:] = template_dataset.variables[lat_dim_name]

    return output_dataset

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
        precip_file = sys.argv[1]
        precip_var_name = sys.argv[2]
        temp_file = sys.argv[3]
        temp_var_name = sys.argv[4]
        output_file_base = sys.argv[5]
        calibration_start_year = int(sys.argv[6])
        calibration_end_year = int(sys.argv[7])

        # use names for the horizontal and vertical dimensions, this allows for more than just lon/lat 3-D variables
        # TODO write a function that pulls these out from the template NetCDF, recognizing the three current
        # dimensionalities: lon/lat for grids, state/division for climate divisions, and station/None for stations 
        lon_dim_name = 'lon'
        lat_dim_name = 'lat'
             
        # use NaN as our fill/missing value
        fill_value = np.NaN
    
        # the valid min and max values (the range within which the resulting SPI values will be clipped)
        valid_min = -3.09
        valid_max = 3.09
                    
        # open the NetCDF files
        with netCDF4.Dataset(precip_file) as precip_dataset, \
            netCDF4.Dataset(temp_file) as temp_dataset:
             
            month_scales = [0, 1, 2, 3, 6, 9, 12, 24, 36, 48, 60, 72]
            main_indicators = ['spi', 'spei']
            main_distributions = ['gamma', 'pearson']
#             month_scales = [0, 3]
#             main_indicators = ['spi']
#             main_distributions = ['gamma']

            # create output datasets for each output variable
            output_datasets = {}
            for i, indicator in enumerate(main_indicators):
                for j, distribution in enumerate(main_distributions):
                    for k, month_scale in enumerate(month_scales):

                        if k > 0:  # skip the first month scale since it's reserved for input data
                            
                            # create the variable name from the indicator, distribution, and month scale
                            variable_name = "{0}_{1}_{2}".format(indicator, distribution, str(month_scale).zfill(2))
    
                            # open an output dataset for this variable, map it to the variable name    
                            output_file = output_file_base + '_' + variable_name + '.nc'
                            output_datasets[variable_name] = initialize_dataset(output_file,
                                                                                precip_dataset,
                                                                                lon_dim_name,
                                                                                lat_dim_name,
                                                                                fill_value,
                                                                                variable_name,
                                                                                valid_min,
                                                                                valid_max)

            # find the start and end date
            data_start_date = netCDF4.num2date(precip_dataset.variables['time'][0], precip_dataset.variables['time'].units)
            data_end_date = netCDF4.num2date(precip_dataset.variables['time'][-1], precip_dataset.variables['time'].units)
            
            # get the data dimensions
            time_size = precip_dataset.variables['time'].size
            lat_size = precip_dataset.variables[lat_dim_name].size
            lon_size = precip_dataset.variables[lon_dim_name].size
          
            # create a shared memory array to contain the input/output data which can be accessed from within another process
            shared_array = Array(ctypes.c_double, len(main_indicators) * len(main_distributions) * len(month_scales) * time_size * lat_size, lock=False)
#             data_shape = (time_size, lat_size)
            full_data_shape = (len(main_indicators), len(main_distributions), len(month_scales), time_size, lat_size)
                
            # create a shared memory array for the latitudes which can be accessed from within another process
            shared_lats_array = Array(ctypes.c_double, len(precip_dataset.variables[lat_dim_name][:]), lock=False)  
            lats_array = np.ctypeslib.as_array(shared_lats_array)
            lats_array = np.reshape(lats_array, precip_dataset.variables[lat_dim_name].shape)
 
            # read the longitude slice into the shared memory array     
            lats_array[:] = precip_dataset.variables[lat_dim_name][:]
            
            # create a processor with a number of worker processes
            number_of_workers = cpu_count()

            # create a Pool, essentially forking with copies of the shared array going to each pooled/forked process
            pool = Pool(processes=number_of_workers, 
                        initializer=init_process, 
                        initargs=(shared_array, shared_lats_array, full_data_shape, main_indicators, main_distributions, month_scales, valid_min, valid_max, data_start_date.year, data_end_date.year, calibration_start_year, calibration_end_year))
            
            # for each longitude slice
            for lon_index in range(lon_size):
    
                logger.info('\n\nProcessing longitude: {}\n'.format(lon_index))

                # get the shared memory array and convert into a numpy array with proper dimensions
                longitude_array = np.ctypeslib.as_array(shared_array)
                longitude_array = np.reshape(longitude_array, full_data_shape)
 
                # read the longitude slice into the shared memory array     
                longitude_array[0, 0, 0, :, :] = precip_dataset.variables[precip_var_name][:, lon_index, :]
                longitude_array[0, 0, 1, :, :] = temp_dataset.variables[temp_var_name][:, lon_index, :]
                
                # set all negative values to np.NaN (our fill value is -999.9 but 
                # is read as a the value rather than being replaced by np.NaN as we'd prefer)
                longitude_array[longitude_array < 0] = np.NaN
                
                # a list of arguments we'll map to the processes of the pool
                arguments_iterable = []

                # loop over each latitude point in the longitude slice
                for lat_index in range(lat_size):
                    
                    # have the processor process the shared array at this lat_index
                    arguments = [lat_index]
                    arguments_iterable.append(arguments)
                        
                # map the arguments iterable to the compute function
                pool.map(compute_worker, arguments_iterable)
    
                # get the longitude slice of fitted values from the shared memory array and convert  
                # into a numpy array with proper dimensions which we can then use to write to NetCDF
                fitted_array = np.ctypeslib.as_array(shared_array)
                fitted_array = np.reshape(fitted_array, full_data_shape)
                     
                for i, indicator in enumerate(main_indicators):
                    for j, distribution in enumerate(main_distributions):
                        for k, month_scale in enumerate(month_scales):
                            if k > 0:  # skip the first month scale since it's used for input
                                
                                # create the variable name so we'll know how to access the corresponding dataset
                                variable_name = "{0}_{1}_{2}".format(indicator, distribution, str(month_scale).zfill(2))
                                
                                # copy the longitude slice values into the NetCDF
                                dataset = output_datasets[variable_name]
                                dataset[variable_name][:, lon_index, :] = fitted_array[i, j, k, :, :]

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
