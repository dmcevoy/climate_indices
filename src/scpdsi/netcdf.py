from datetime import datetime
import logging
import netCDF4
import numpy as np

# set up a global logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------
def open_datasets(files):

    datasets = []
    for netcdf_file in files:
        logger.info('Opening and appending Dataset for file: {file}'.format(file=netcdf_file))
        datasets.append(netCDF4.Dataset(netcdf_file))
        
    return datasets

#--------------------------------------------------------------------------------------
def close_datasets(datasets):

    for dataset in datasets:
        dataset.close()
        
#--------------------------------------------------------------------------------------
def load_data(input_dataset,
              variable_name,
              time_index):

    # make sure the data set has the dimensions and variables we expect
    if (not input_dataset.dimensions.has_key('lat')):
        logger.critical("Input data set file is missing the 'lat' dimension")
        return 1
    if (not input_dataset.dimensions.has_key('lon')):
        logger.critical("Input data set file is missing the 'lon' dimension")
        return 1
    if (not input_dataset.dimensions.has_key('time')):
        logger.critical("Input data set file is missing the 'time' dimension")
        return 1
    if (not input_dataset.variables.has_key('lat')):
        logger.critical("Input data set file is missing the 'lat' variable")
        return 1
    if (not input_dataset.variables.has_key('lon')):
        logger.critical("Input data set file is missing the 'lon' variable")
        return 1
    if (not input_dataset.variables.has_key('time')):
        logger.critical("Input data set file is missing the 'time' variable")
        return 1
    if (not input_dataset.variables.has_key(variable_name)):
        logger.critical('Input data set file is missing the ' + variable_name + ' variable')
        return 1

    #TODO verify that the dimensions are in [time, lon, lat] order

    try:
        # pull the data from the variable
        data = input_dataset.variables[variable_name][time_index:time_index + 1:1, :, :]
    except Exception as ex:
        message = 'Unable to read data for the variable named \'{var}\':  Cause: {cause}'.format(var=variable_name, cause=ex.__cause__)
        logger.critical(message)
        raise RuntimeError(message)

    return data

#--------------------------------------------------------------------------------------
def extract_coords(datasets):

    for dataset in datasets:

        if not dataset.dimensions.has_key('lat'):
            message = "Input data set file is missing the 'lat' dimension"
            logger.critical(message)
            raise ValueError(message)
        if not dataset.dimensions.has_key('lon'):
            message = "Input data set file is missing the 'lon' dimension"
            logger.critical(message)
            raise ValueError(message)
        if not dataset.dimensions.has_key('time'):
            message = "Input data set file is missing the 'time' dimension"
            logger.critical(message)
            raise ValueError(message)
        if not dataset.variables.has_key('lat'):
            message = "Input data set file is missing the 'lat' variable"
            logger.critical(message)
            raise ValueError(message)
        if not dataset.variables.has_key('lon'):
            message = "Input data set file is missing the 'lon' variable"
            logger.critical(message)
            raise ValueError(message)
        if not dataset.variables.has_key('time'):
            message = "Input data set file is missing the 'time' variable"
            logger.critical(message)
            raise ValueError(message)

        #TODO make sure the data variable(s) have dimensions [time, lon, lat]

        # on the first pass we get our "reference" set of lons, lats, and times, and all subsequent files should match
        if 'lons' not in locals():  # this is true only on the first pass 
            lons = dataset.variables['lon'][:]
            lats = dataset.variables['lat'][:]
            times = dataset.variables['time'][:]

        else:
            # get the lons and lats from the current file
            comparison_lons = dataset.variables['lon'][:]
            comparison_lats = dataset.variables['lat'][:]

            # make sure that this file's lons and lats match up with the initial file
            if not np.allclose(lons, comparison_lons, atol=0.00001):
                message = 'Longitude values not matching between data sets'
                logger.critical(message)
                raise ValueError(message)
            if not np.allclose(lats, comparison_lats, atol=0.00001):
                message = 'Latitude values not matching between data sets'
                logger.critical(message)
                raise ValueError(message)

    return times, lons, lats

#--------------------------------------------------------------------------------------
def write_dataset(output_file,
                  template_dataset,
                  variable_data,
                  variable_name,
                  attributes):

    # get the coordinates from the template file
    times = template_dataset.variables['time'][:]
    lons = template_dataset.variables['lon'][:]
    lats = template_dataset.variables['lat'][:]

    #TODO verify the shape of the PDSI data values array against the template dimensions

    # open the output file for writing, set its dimensions and variables
    dataset = netCDF4.Dataset(output_file, 'w')
    dataset.createDimension('time', None)
    dataset.createDimension('lon', len(lons))
    dataset.createDimension('lat', len(lats))

    # set some global group attributes
    dataset.title = 'Self-calibrated Palmer Drought Severity Index (scPDSI)'
    dataset.source = 'calculation using a new NCEI Python version of code originally developed in Matlab by J. Wolf, University of Idaho'
    dataset.institution = 'National Centers for Environmental Information, NESDIS, NOAA, U.S. Department of Commerce'
    dataset.standard_name_vocabulary = 'CF Standard Name Table (v26, 08 November 2013)'
    dataset.date_created = str(datetime.now())
    dataset.date_modified = str(datetime.now())
    dataset.Conventions = 'CF-1.6'

    # create a time coordinate variable with an increment per month of the period of record
    time_variable = dataset.createVariable('time', 'i4', ('time',))
    time_variable.long_name = 'time'
    time_variable.standard_name = 'time'
#    time_variable.calendar = template_dataset.variables['time'].calendar
#    time_variable.units = template_dataset.variables['time'].units
    time_variable[:] = times

    # create the lon coordinate variable
    lon_variable = dataset.createVariable('lon', 'f4', ('lon',))
    lon_variable.long_name = 'longitude'
    lon_variable.standard_name = 'longitude'
    lon_variable[:] = lons
    try:
        units = template_dataset.variables['lon'].units
        if units is not None:
            lon_variable.units = units
    except AttributeError:
        logger.info('No units found for longitude coordinate variable in the template data set NetCDF')

    # create the lat coordinate variable
    lat_variable = dataset.createVariable('lat', 'f4', ('lat',))
    lat_variable.long_name = 'latitude'
    lat_variable.standard_name = 'latitude'
    lat_variable[:] = lats
    try:
        units = template_dataset.variables['lat'].units
        if units is not None:
            lat_variable.units = units
    except AttributeError:
        logger.info('No units found for latitude coordinate variable in the template data set NetCDF')

    # create the variable
    variable = dataset.createVariable(variable_name,
                                      'f4',
                                      ('time', 'lon', 'lat'),
                                      fill_value=np.NaN,
                                      zlib=True,
                                      least_significant_digit=3)

    # add the attributes to the variable
    variable.setncatts(attributes)
    
    # load the data values array into the variable
    variable[:] = variable_data

    # close the output NetCDF file
    dataset.close()