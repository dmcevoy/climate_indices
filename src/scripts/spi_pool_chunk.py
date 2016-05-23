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
def compute_indicator(args):

    # extract the arguments
    lon_index_start = args[0]
    lat_index = args[1]

    # turn the shared arrays into numpy arrays
    input_data = np.ctypeslib.as_array(input_shared_array)
    input_data = input_data.reshape(input_data_shape)
    gamma_data = np.ctypeslib.as_array(output_gamma_array)
    gamma_data = gamma_data.reshape(output_data_shape)
    pearson_data = np.ctypeslib.as_array(output_pearson_array)
    pearson_data = pearson_data.reshape(output_data_shape)

    for lon_index in range(lons_per_chunk):

        # only process non-empty grid cells, i.e. input_data array contains at least some non-NaN values
        if (isinstance(input_data[:, lon_index, lat_index], np.ma.MaskedArray) and input_data[:, lon_index, lat_index].mask.all()) \
            or np.isnan(input_data[:, lon_index, lat_index]).all() or (input_data[:, lon_index, lat_index] <= 0).all():

#             logger.info('No input_data at lon/lat: {0}/{1}'.format(lon_index_start + lon_index, lat_index))
            pass

        else:  # we have some valid values to work with

            logger.info('Processing longitude/latitude: {}/{}'.format(lon_index_start + lon_index, lat_index))
    
            for scale_index, month_scale in enumerate(month_scales):
            
                # perform a fitting to gamma
                gamma_data[scale_index, :, lon_index, lat_index] = indices.spi_gamma(input_data[:, lon_index, lat_index],
                                                                                      month_scale,
                                                                                      valid_min,
                                                                                      valid_max)

                # perform a fitting to gamma
                pearson_data[scale_index, :, lon_index, lat_index] = indices.spi_pearson(input_data[:, lon_index, lat_index],
                                                                                         month_scale,
                                                                                         valid_min,
                                                                                         valid_max, 
                                                                                         data_start_year, 
                                                                                         data_end_year, 
                                                                                         calibration_start_year, 
                                                                                         calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
def init_process(worker_input_array,
                 worker_input_shape,
                 worker_gamma_array,
                 worker_pearson_array,
                 worker_output_shape,
                 worker_month_scales,
                 worker_valid_min,
                 worker_valid_max,
                 worker_data_start_year, 
                 worker_data_end_year, 
                 worker_calibration_start_year,
                 worker_calibration_end_year,
                 worker_lons_per_chunk):

    # put the arguments to the global namespace
    global input_shared_array, input_data_shape, output_gamma_array, output_pearson_array, output_data_shape, \
           month_scales, valid_min, valid_max, data_start_year, data_end_year, \
           calibration_start_year, calibration_end_year, lons_per_chunk
    input_shared_array = worker_input_array
    input_data_shape = worker_input_shape
    output_gamma_array = worker_gamma_array
    output_pearson_array = worker_pearson_array
    output_data_shape = worker_output_shape
    month_scales = worker_month_scales
    valid_min = worker_valid_min
    valid_max = worker_valid_max
    data_start_year = worker_data_start_year
    data_end_year = worker_data_end_year
    calibration_start_year = worker_calibration_start_year
    calibration_end_year = worker_calibration_end_year
    lons_per_chunk = worker_lons_per_chunk


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
        output_file_base = sys.argv[3]
        calibration_start_year = int(sys.argv[4])
        calibration_end_year = int(sys.argv[5])

        # the month scales for which we'll compute indices
        month_scales = [1, 2, 3, 6, 9, 12, 24, 36, 48, 60, 72]
            
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

        # open the NetCDF files
        with netCDF4.Dataset(precip_file) as input_dataset:

            # get the dimension sizes
            time_size = input_dataset.variables['time'].size
            lat_size = input_dataset.variables[lat_dim_name].size
            lon_size = input_dataset.variables[lon_dim_name].size

            # find the year range of the data
            initial_time = netCDF4.num2date(input_dataset.variables['time'][0], input_dataset.variables['time'].units)
            total_years = int(time_size / 12)
            initial_data_year = initial_time.year
            final_data_year = initial_data_year + total_years
            
            # create and initialize a NetCDF dataset object that'll be written for each month scale
            datasets = {}
            for distribution_fitting in ['gamma', 'pearson']:
                    
                for month_scale in month_scales:
                
                    # create the variable name from the indicator, distribution, and month scale
                    variable_name = 'spi_{0}_{1}'.format(distribution_fitting, str(month_scale).zfill(2))
    
                    # open the NetCDF for writing
                    output_dataset = netCDF4.Dataset(output_file_base + '_' + variable_name + '.nc', 'w')
    
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
    
                    datasets[variable_name] = output_dataset
                    
            # the number of longitudes we'll have per longitude slice
            lon_stride = 20

            # create a shared memory array which can be accessed from within another process
            input_shared_array = Array(ctypes.c_double, time_size * lon_stride * lat_size, lock=False)
            input_data_shape = (time_size, lon_stride, lat_size)
            output_gamma_array = Array(ctypes.c_double, len(month_scales) * time_size * lon_stride * lat_size, lock=False)
            output_person_array = Array(ctypes.c_double, len(month_scales) * time_size * lon_stride * lat_size, lock=False)
            output_data_shape = (len(month_scales), time_size, lon_stride, lat_size)

            # create a processor with a number of worker processes
            number_of_workers = cpu_count()

            # create a Pool, essentially forking with copies of the shared array going to each pooled/forked process
            pool = Pool(processes=number_of_workers,
                        initializer=init_process,
                        initargs=(input_shared_array, input_data_shape, output_gamma_array, output_person_array, output_data_shape, month_scales, valid_min, valid_max, initial_data_year, final_data_year, calibration_start_year, calibration_end_year, lon_stride))

            # loop over each chunk of longitudes
            for lon_index in range(0, lon_size, lon_stride):

                logger.info('\n\nProcessing chunk starting at longitude: {}\n'.format(lon_index))

                # get the shared memory array and convert into a numpy array with proper dimensions
                longitude_array = np.ctypeslib.as_array(input_shared_array)
                longitude_array = np.reshape(longitude_array, input_data_shape)

                # read the longitude slice into the shared memory array
                if (lon_index + lon_stride > lon_size):
                    incomplete_chunk = input_dataset.variables[precip_var_name][:, lon_index:, :]
                    longitude_array[:, 0:incomplete_chunk.shape[1], :] = incomplete_chunk
                else:
                    longitude_array[:] = input_dataset.variables[precip_var_name][:, lon_index:lon_index + lon_stride, :]

                # set all negative values to np.NaN (our fill value is -999.9 but
                # is read as a the value rather than being replaced by np.NaN as we'd prefer)
                longitude_array[longitude_array < 0] = np.NaN

                # a list of arguments we'll map to the processes of the pool
                arguments_iterable = []

                # loop over each latitude point in the longitude slice
                for lat_index in range(lat_size):

                    # have the processor process the shared array at this index
                    arguments = [lon_index, lat_index]
                    arguments_iterable.append(arguments)

                # map the arguments iterable to the compute function
#                 pool.apply_async(compute_indicator, arguments_iterable)
                pool.map(compute_indicator, arguments_iterable)
#                pool.map_async(compute_indicator, arguments_iterable)

                # write out the SPI values for both distributions
                for distribution_fitting in ['gamma', 'pearson']:

                    if distribution_fitting =='gamma':
                                       
                        # get the shared memory array and convert into a numpy array with proper dimensions
                        results_array = np.ctypeslib.as_array(output_gamma_array)

                    elif distribution_fitting =='pearson':
                                       
                        # get the shared memory array and convert into a numpy array with proper dimensions
                        results_array = np.ctypeslib.as_array(output_person_array)

                    # reshape the shared memory array
                    results_array = np.reshape(results_array, output_data_shape)
                    
                    # loop over each month scale and write the longitude chunk to file for each
                    for scale_index, month_scale in enumerate(month_scales):
    
                        # create the variable name from the indicator, distribution, and month scale
                        variable_name = 'spi_{0}_{1}'.format(distribution_fitting, str(month_scale).zfill(2))
        
                        # write the longitude chunk of computed values into the output NetCDF
                        if (lon_index + lon_stride > lon_size):
                            datasets[variable_name].variables[variable_name][:, lon_index:lon_size, :] = results_array[scale_index, :, 0:lon_size - lon_index, :]
                        else:
                            datasets[variable_name].variables[variable_name][:, lon_index:lon_index + lon_stride, :] = results_array[scale_index, :, :, :]

            # all processes have completed, close the pool
            pool.close()

            # join the processes, wait on completion
            pool.join()

        # close the open output datasets
        for dataset in datasets:
            dataset.close()
            
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))

    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise
