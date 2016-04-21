from __future__ import division
from datetime import datetime
import logging
import netCDF4
import netcdf_utils
import numpy as np
import sys
import threading
import distribution_fitter
from Queue import Queue

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

# lock for threading
lock = threading.Lock()

# create a queue to hold worker thread parameter items (assuming a single parameter is required)
worker_queue = Queue()

#-----------------------------------------------------------------------------------------------------------------------
def compute_indicator_by_lons_pooled(input_dataset,
                                     output_dataset,
                                     input_var_name,
                                     output_var_name,
                                     month_scale,
                                     valid_min, 
                                     valid_max):
    
    while True:
    
        # get the longitude index we'll use from the worker queue
        dim1_index = worker_queue.get()
        
        # lock the thread when doing I/O
        lock.acquire()
        
        # slice out the period of record for the x/y point
        data = input_dataset.variables[input_var_name][:, dim1_index, :]
        
        # release the lock since we'll not share anything else until doing the I/O to the output dataset                        
        lock.release()
        
        # keep the original data shape, we'll use this to reshape later
        original_shape = input_dataset.variables[input_var_name].shape
        
        for dim2_index in range(input_dataset.variables[input_var_name].shape[2]):
            
            # only process non-empty grid cells, i.e. data array contains at least some non-NaN values
            if isinstance(data[:, dim2_index], np.ma.MaskedArray) and data[:, dim2_index].mask.all():
        
                pass         
             
            else:  # we have some valid values to work with
        
                logger.info('Processing x/y {}/{}'.format(dim1_index, dim2_index))
        
                # perform a fitting to gamma     
                data[:, dim2_index] = distribution_fitter.fit_to_gamma(data[:, dim2_index],
                                                                       month_scale, 
                                                                       valid_min, 
                                                                       valid_max)
            
        # reacquire the thread lock for doing NetCDF I/O
        lock.acquire()
        
        # slice out the period of record for the x/y point
        output_dataset.variables[output_var_name][:, dim1_index, :] = np.reshape(data, (original_shape[0], 1, original_shape[2]))
        
        # release the lock since we'll not share anything else until doing the I/O to the output dataset                        
        lock.release()
        
        # communicate to the queue that this task has been completed and the item can be removed
        worker_queue.task_done()
    
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
        logger.info("Start time: {}".format(start_datetime, '%x'))
        
        # get the command line arguments
        precip_file = sys.argv[1]
        precip_var_name = sys.argv[2]
        output_file_base = sys.argv[3]
        month_scale = int(sys.argv[4])

        # replace with command line argument, etc.
        number_of_cpus = 4

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
            

            # start enough worker threads to match the number of CPUs
            for i in range(number_of_cpus):
                
                thread = threading.Thread(target=compute_indicator_by_lons_pooled,
                                          args=(precip_dataset, 
                                                output_dataset, 
                                                precip_var_name,
                                                variable_name, 
                                                month_scale, 
                                                valid_min, 
                                                valid_max))
                thread.daemon = True
                thread.start()
                        
            # loop over the grid cells by full longitude (X-dimension) slice
            for x in range(precip_dataset.variables[x_dim_name].size):
                
                # add the index to the list of items to perform work on
                worker_queue.put(x)
                
            # block until all tasks are done
            worker_queue.join()
                
        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time: {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time: {}".format(elapsed, '%x'))
        
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise
