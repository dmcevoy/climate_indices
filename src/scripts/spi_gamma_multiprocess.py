from __future__ import division
from datetime import datetime
import indices
import logging
from multiprocessing import Process, Array
#from multiprocessing import Pool#, JoinableQueue, Process
import netCDF4
import netcdf_utils
import numpy as np
import sys
from multiprocessing import JoinableQueue
import ctypes

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
class Processor:
     
    queue = None
                 
    def __init__(self, 
                 month_scale, 
                 valid_min, 
                 valid_max,
                 nb_workers=1):
              
        # create a number of worker processes
        self.queue = JoinableQueue()
        self.processes = [Process(target=self.compute_indicator_longitude) for _ in range(nb_workers)]
        for p in self.processes:
            p.start()
         
        # set some values that will be used across all processes
        self.month_scale = month_scale 
        self.valid_min = valid_min
        self.valid_max = valid_max
                 
    def add_work_item(self, item):
         
        # add the parameters list to the parameters queue
        self.queue.put(item)
 
    def compute_indicator(self):
         
        while True:
              
            # get a list of arguments from the queue
            arguments = self.queue.get()
              
            # if we didn't get one we keep looping
            if arguments is None:
                break
  
            # process the arguments here
            data = arguments[0]
            index = arguments[1] 
            
            # only process non-empty grid cells, i.e. data array contains at least some non-NaN values
            if isinstance(data[:, index], np.ma.MaskedArray) and data[:, index].mask.all():
             
                pass         
                  
            else:  # we have some valid values to work with
             
                logger.info('Processing latitude: {}'.format(index))
             
                # perform a fitting to gamma     
                data[:, index] = indices.spi_gamma(data[:, index],
                                                   self.month_scale, 
                                                   self.valid_min, 
                                                   self.valid_max)
 
            # indicate that the task has completed
            self.queue.task_done()
 
    def terminate(self):
 
        """ wait until queue is empty and terminate processes """
        self.queue.join()
        for p in self.processes:
            p.terminate()

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
            x_dim_name = 'lon'
            y_dim_name = 'lat'
             
            # make sure the data matches the dimensions
            time_size = input_dataset.variables['time'].size
            y_size = input_dataset.variables[y_dim_name].size
            x_size = input_dataset.variables[x_dim_name].size
     
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
            output_dataset.createDimension(x_dim_name, x_size)
            output_dataset.createDimension(y_dim_name, y_size)
             
            # get the appropriate data types to use for the variables based on the values arrays
            time_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables['time'])
            x_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables[x_dim_name])
            y_dtype = netcdf_utils.find_netcdf_datatype(input_dataset.variables[y_dim_name])
            data_dtype = netcdf_utils.find_netcdf_datatype(fill_value)
             
            # create the variables
            time_variable = output_dataset.createVariable('time', time_dtype, ('time',))
            x_variable = output_dataset.createVariable(x_dim_name, x_dtype, (x_dim_name,))
            y_variable = output_dataset.createVariable(y_dim_name, y_dtype, (y_dim_name,))
            data_variable = output_dataset.createVariable(variable_name, 
                                                          data_dtype, 
                                                          ('time', x_dim_name, y_dim_name,), 
                                                          fill_value=fill_value)
             
            # set the variables' attributes
            time_variable.setncatts(input_dataset.variables['time'].__dict__)
            x_variable.setncatts(input_dataset.variables[x_dim_name].__dict__)
            y_variable.setncatts(input_dataset.variables[y_dim_name].__dict__)
            data_variable.setncatts(variable_attributes)
             
            # set the coordinate variables' values
            time_variable[:] = input_dataset.variables['time'][:]
            x_variable[:] = input_dataset.variables[x_dim_name]
            y_variable[:] = input_dataset.variables[y_dim_name]
          
        # create a shared memory array we'll use for each longitude slice we'll read/write   
        shared_array_base = Array(ctypes.c_double, )
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        shared_array = shared_array.reshape(time_size * y_size)

        # create a processor with a number of worker processes
        number_of_workers = 1
        processor = Processor(month_scale, valid_min, valid_max, number_of_workers)
        
        # for each longitude slice
        for dim1_index in range(input_dataset.variables['prcp'].shape[1]):

            # read the longitude slice into a data array     
            longitude_slice = input_dataset.variables[input_var_name][:, dim1_index, :]
            
            # convert the array onto a shared memory array which can be accessed from within another process
            shared_array_base = Array(ctypes.c_double, longitude_slice)
            shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
            shared_array = shared_array.reshape(time_size * y_size)
            
            # loop over each latitude point in the longitude slice
            for dim2_index in range(input_dataset.variables['prcp'].shape[2]):
                
                # have the processor process the shared array at this index
                arguments = [shared_array, dim2_index]
                processor.add_work_item(arguments)
                
            # join to the processor and don't continue until all processes have completed
            processor.terminate()

            # write the fitted longitude slice values into the output NetCDF
            output_dataset.variables[variable_name][:, dim1_index, :] = np.reshape(shared_array, (time_size, 1, y_size))

        # report on the elapsed time
        end_datetime = datetime.now()
        logger.info("End time:      {}".format(end_datetime, '%x'))
        elapsed = end_datetime - start_datetime
        logger.info("Elapsed time:  {}".format(elapsed, '%x'))
        
    except Exception, e:
        logger.error('Failed to complete', exc_info=True)
        raise

