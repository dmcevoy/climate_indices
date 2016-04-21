from __future__ import division
import distribution_fitter
import logging
from numba import float64, int32, jit
import numpy as np
from thornthwaite import thornthwaite

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


#-----------------------------------------------------------------------------------------------------------------------
@jit(float64[:](float64[:], int32))
def percentage_of_average(precip_monthly_values, 
                          month_scale):
    
    sums = distribution_fitter.get_sliding_sums(precip_monthly_values, month_scale)
    monthly_means = np.full((12,), np.nan)
    for i in range(12):
        monthly_means[i] = np.nanmean(sums[i::12])
    
    percent_of_averages = np.full(sums.shape, np.nan)
    for i in range(sums.size):
        percent_of_averages[i] = sums[i] / monthly_means[i % 12]
    
    return percent_of_averages
    
#-----------------------------------------------------------------------------------------------------------------------
def spi_gamma(precip_monthly_values, 
              month_scale, 
              valid_min, 
              valid_max):

    
    return distribution_fitter.fit_to_gamma(precip_monthly_values, 
                                            month_scale, 
                                            valid_min, 
                                            valid_max)

#-----------------------------------------------------------------------------------------------------------------------
def spi_pearson(precip_monthly_values,
                month_scale, 
                valid_min, 
                valid_max,
                data_start_year,
                data_end_year,
                calibration_start_year, 
                calibration_end_year):

    # perform the SPI computation (fit to the Pearson distribution) and assign the values into the dataset
    return distribution_fitter.fit_to_pearson(precip_monthly_values, 
                                              month_scale, 
                                              valid_min, 
                                              valid_max, 
                                              data_start_year, 
                                              data_end_year, 
                                              calibration_start_year, 
                                              calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
def spei_gamma(precip_monthly_values,
               temp_monthly_values,
               data_start_year,
               latitude,
               month_scale, 
               valid_min, 
               valid_max):

    # compute the PET values using Thornthwaite's equation
    pet_monthly_values = thornthwaite(temp_monthly_values, latitude, data_start_year, np.nan)
        
    # offset we add to the (P - PE values) in order to bring all values into the positive range     
    p_minus_pe_offset = 1000.0

    # get the difference of P - PE, assign the values into the temperature array (in order to reuse the memory)
    p_minus_pe = (precip_monthly_values - pet_monthly_values) + p_minus_pe_offset
    
    # perform the SPEI computation (fit to the Gamma distribution) and assign the values into the dataset
    return distribution_fitter.fit_to_gamma(p_minus_pe, 
                                            month_scale, 
                                            valid_min, 
                                            valid_max)
    
#-----------------------------------------------------------------------------------------------------------------------
def spei_pearson(precip_monthly_values,
                 temp_monthly_values,
                 month_scale, 
                 latitude,
                 valid_min, 
                 valid_max,
                 data_start_year,
                 data_end_year,
                 calibration_start_year, 
                 calibration_end_year):

    # compute the PET values using Thornthwaite's equation
    pet_monthly_values = thornthwaite(temp_monthly_values, latitude, data_start_year, np.nan)
        
    # offset we add to the (P - PE values) in order to bring all values into the positive range     
    p_minus_pe_offset = 1000.0

    # get the difference of P - PE, assign the values into the temperature array (in order to reuse the memory)
    p_minus_pe = (precip_monthly_values - pet_monthly_values) + p_minus_pe_offset
    
    # perform the SPEI computation (fit to the Pearson distribution) and assign the values into the dataset
    return distribution_fitter.fit_to_pearson(p_minus_pe, 
                                              month_scale, 
                                              valid_min, 
                                              valid_max, 
                                              data_start_year, 
                                              data_end_year, 
                                              calibration_start_year, 
                                              calibration_end_year)

#-----------------------------------------------------------------------------------------------------------------------
def spei_spi_pearson_pet(precip_monthly_values,
                         temp_monthly_values,
                         month_scale, 
                         latitude,
                         valid_min, 
                         valid_max,
                         data_start_year,
                         data_end_year,
                         calibration_start_year, 
                         calibration_end_year):

    '''
    This function computes PET, SPI and SPEI fitted to the Pearson Type III distribution.
    
    :param precip_monthly_values: an array of monthly total precipitation values, of the same size 
                                  and shape as the input temperature array
    :param temp_monthly_values: an array of monthly average temperature values, of the same size 
                                and shape as the input precipitation array
    :param month_scale: the number of months over which the values should be scaled before computing the index
    :param latitude: the latitude, in degrees, of the location
    :param valid_min: valid minimum of the resulting SPI and SPEI values
    :param valid_max: valid maximum of the resulting SPI and SPEI values
    :param data_start_year: the initial year of the input datasets (assumes that the two inputs cover the same period)
    :param data_end_year: the final year of the input datasets (assumes that the two inputs cover the same period)
    :param calibration_start_year: the initial year of the calibration period
    :param calibration_end_year: the final year of the calibration period
    :return: an array of SPEI values, of the same size and shape as the input temperature and precipitation arrays
    :return: an array of SPI values, of the same size and shape as the input temperature and precipitation arrays
    :return: an array of PET values, of the same size and shape as the input temperature and precipitation arrays
    '''

    # compute the SPI fitted to the Pearson Type III distribution
    spi = distribution_fitter.fit_to_pearson(precip_monthly_values, 
                                                     month_scale, 
                                                     valid_min, 
                                                     valid_max, 
                                                     data_start_year, 
                                                     data_end_year, 
                                                     calibration_start_year, 
                                                     calibration_end_year)
    
    # compute the PET values using Thornthwaite's equation
    pet = thornthwaite(temp_monthly_values, latitude, data_start_year, np.nan)
        
    # offset we add to the (P - PE values) in order to bring all values into the positive range     
    p_minus_pe_offset = 1000.0

    # get the difference of P - PE, assign the values into the temperature array (in order to reuse the memory)
    p_minus_pe = (precip_monthly_values - pet) + p_minus_pe_offset
    
    # perform the SPEI computation (fit to the Pearson distribution) and assign the values into the dataset
    spei = distribution_fitter.fit_to_pearson(p_minus_pe, 
                                                      month_scale, 
                                                      valid_min, 
                                                      valid_max, 
                                                      data_start_year, 
                                                      data_end_year, 
                                                      calibration_start_year, 
                                                      calibration_end_year)

    return spei, spi, pet
