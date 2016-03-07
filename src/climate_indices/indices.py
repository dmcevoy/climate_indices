import distribution_fitter
import logging
from numpy import NaN
from thornthwaite import thornthwaite

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)


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

    # perform the SPEI computation (fit to the Pearson distribution) and assign the values into the dataset
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
    pet_monthly_values = thornthwaite(temp_monthly_values, latitude, data_start_year, NaN)
        
    # get the difference of P - PE, assign the values into the temperature array (in order to reuse the memory)
    p_minus_pe_offset = 1000.0
    p_minus_pe = (precip_monthly_values - pet_monthly_values) + p_minus_pe_offset
    
    # perform the SPEI computation (fit to the Pearson distribution) and assign the values into the dataset
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
    pet_monthly_values = thornthwaite(temp_monthly_values, latitude, data_start_year, NaN)
        
    # get the difference of P - PE, assign the values into the temperature array (in order to reuse the memory)
    p_minus_pe_offset = 1000.0
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
