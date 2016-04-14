"""
Calculate potential evapotranspiration using the Thornthwaite (1948 method)

Modified from original code found here: https://github.com/woodcrafty/PyETo
 
:copyright: (c) 2015 by Mark Richards.
:license: BSD 3-Clause, see LICENSE.txt for more details.

References
----------
Thornthwaite CW (1948) An approach toward a rational classification of climate. Geographical Review, 38, 55-94.
"""

from __future__ import division
import calendar
import math
from numba import float64, int32, jit
import numpy as np
from _check import (
    check_doy as _check_doy,
    check_latitude_rad as _check_latitude_rad,
    check_sol_dec_rad as _check_sol_dec_rad,
    check_sunset_hour_angle_rad as _check_sunset_hour_angle_rad,
)

_MONTH_DAYS_NONLEAP = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
_MONTH_DAYS_LEAP = np.array([31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64[:](float64, int32))
def monthly_mean_daylight_hours(latitude, year=None):
    """
    Calculate mean daylight hours for each month of the year for a given latitude.

    :param latitude: Latitude [radians]
    :param year: Year for the daylight hours are required. The only effect of
        *year* is to change the number of days in Feb to 29 if it is a leap year.
        If left as the default, None, then a normal (non-leap) year is assumed.
    :return: Mean daily daylight hours of each month of a year [hours]
    :rtype: List of floats.
    """

    # validate the latitude argument
    _check_latitude_rad(latitude)

    # get the array of days for each month based on whether or not we're in a leap year
    if year is None or not calendar.isleap(year):
        month_days = _MONTH_DAYS_NONLEAP
    else:
        month_days = _MONTH_DAYS_LEAP
        
    # allocate an array of daylight hours for each of the 12 months of the year
    monthly_mean_dlh = np.zeros((12,))
    
    # keep count of the day of the year
    day_of_year = 1
    
    # loop over each calendar month to calculate the daylight hours for the month
    for i, days_in_month in enumerate(month_days):
        cumulative_daylight_hours = 0.0   # cumulative daylight hours for the month
        for _ in range(1, days_in_month + 1):
            daily_solar_declination = solar_declination(day_of_year)
            daily_sunset_hour_angle = sunset_hour_angle(latitude, daily_solar_declination)
            cumulative_daylight_hours += daylight_hours(daily_sunset_hour_angle)
            day_of_year += 1
        
        # calculate the mean daylight hours of the month
        monthly_mean_dlh[i] = cumulative_daylight_hours / days_in_month
        
    return monthly_mean_dlh

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(int32))
def solar_declination(day_of_year):
    """
    Calculate solar declination from day of the year.

    Based on FAO equation 24 in Allen et al (1998).

    :param day_of_year: Day of year integer between 1 and 365 or 366).
    :return: solar declination [radians]
    :rtype: float
    """
    _check_doy(day_of_year)
    return 0.409 * math.sin(((2.0 * math.pi / 365.0) * day_of_year - 1.39))

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(float64, float64))
def sunset_hour_angle(latitude, sol_dec):
    """
    Calculate sunset hour angle (*Ws*) from latitude and solar
    declination.

    Based on FAO equation 25 in Allen et al (1998).

    :param latitude: Latitude [radians]. Note: *latitude* should be negative
        if in the southern hemisphere, positive if in the northern hemisphere.
    :param sol_dec: Solar declination [radians]. Can be calculated using ``sol_dec()``.
    :return: Sunset hour angle [radians].
    :rtype: float
    """
    # validate the latitude and solar declination angle arguments
    _check_latitude_rad(latitude)
    _check_sol_dec_rad(sol_dec)

    cos_sha = -math.tan(latitude) * math.tan(sol_dec)
    
    # If tmp is >= 1 there is no sunset, i.e. 24 hours of daylight
    # If tmp is <= 1 there is no sunrise, i.e. 24 hours of darkness
    # See http://www.itacanet.org/the-sun-as-a-source-of-energy/part-3-calculating-solar-angles/
    # Domain of acos is -1 <= x <= 1 radians (this is not mentioned in FAO-56!)
    return math.acos(min(max(cos_sha, -1.0), 1.0))

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64(float64))
def daylight_hours(sunset_hour_angle):
    """
    Calculate daylight hours from sunset hour angle.

    Based on FAO equation 34 in Allen et al (1998).

    :param sha: Sunset hour angle [rad]. Can be calculated using ``sunset_hour_angle()``.
    :return: Daylight hours.
    :rtype: float
    """
    
    # validate the sunset hour angle argument
    _check_sunset_hour_angle_rad(sunset_hour_angle)
    
    return (24.0 / math.pi) * sunset_hour_angle

#-----------------------------------------------------------------------------------------------------------------------
@jit(float64[:](float64[:], float64, int32, float64))
def thornthwaite(monthly_temps, 
                 latitude, 
                 start_year,
                 fill_value):
    """
    Estimate monthly potential evapotranspiration (PET) using the
    Thornthwaite (1948) method.

    Thornthwaite equation:

        *PET* = 1.6 (*L*/12) (*N*/30) (10*Ta* / *I*)***a*

    where:

    * *Ta* is the mean daily air temperature [deg C, if negative use 0] of the month being calculated
    * *N* is the number of days in the month being calculated
    * *L* is the mean day length [hours] of the month being calculated
    * *a* = (6.75 x 10-7)*I***3 - (7.71 x 10-5)*I***2 + (1.792 x 10-2)*I* + 0.49239
    * *I* is a heat index which depends on the 12 monthly mean temperatures and
      is calculated as the sum of (*Tai* / 5)**1.514 for each month, where
      Tai is the air temperature for each month in the year

    :param monthly_temps: Iterable containing mean daily air temperature for each month [deg C].
    :param latitude: latitude of the location, in degrees north (-90..90)
    :param start_year: year corresponding to the start of the data set (data set is 
        assumed to start on January of the initial year) 
    :return: Estimated monthly potential evapotranspiration of each month of the year [mm/month]
    :rtype: array of floats with shape: (total # of months) (equal to the shape of the input dataset
    """

    # validate the input data array
    data_shape = monthly_temps.shape
    final_year_empty_months = 0
    if len(data_shape) == 1:
        
        # dataset is assumed to represent one long row of months starting with January of the initial year, and we'll 
        # reshape into (years, 12) where each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)
        
        # get the number of months left off of the final year
        final_year_empty_months = 12 - (data_shape[0] % 12)
        
        # if any months were left off then we'll pad the final months of the year with NaNs
        if final_year_empty_months > 0:
            
            # make an array of NaNs for each of the remaining months of the final year of the dataset
            final_year_pad = np.full((final_year_empty_months,), np.nan)
            
            # append the pad months to the dataset to complete the final year
            monthly_temps = np.concatenate((monthly_temps, final_year_pad))
        
        # reshape the dataset from (months) to (years, 12)
        total_years = monthly_temps.size // 12
        monthly_temps = np.reshape(monthly_temps, (total_years, 12))
        
    elif (len(data_shape) > 2) or ((len(data_shape) == 2) and (data_shape[1] != 12)):
        raise ValueError('Input monthly mean temperatures data array has an invalid shape: {0}.'.format(data_shape))
    
    # at this point we assume that our dataset array has shape (years, 12) where 
    # each row is a year with 12 columns of monthly values (Jan, Feb, ..., Dec)
    
    # convert the latitude from degrees to radians
    latitude = math.radians(latitude)
    
    # adjust negative temperature values to zero, since negative values aren't allowed (no evaporation below freezing)
    monthly_temps[monthly_temps < 0] = 0.0
    
    # mean the monthly temperature values over the month axis, giving us 12 monthly means for the period of record
    mean_monthly_temps = np.nanmean(monthly_temps, axis=0)    
    
    # calculate the heat index (I)
    I = np.sum(np.power(mean_monthly_temps / 5.0, 1.514))

    # calculate the a coefficient
    a = (6.75e-07 * I ** 3) - (7.71e-05 * I ** 2) + (1.792e-02 * I) + 0.49239

    # get mean daylight hours for both normal and leap years 
    mean_daylight_hours_nonleap = np.array(monthly_mean_daylight_hours(latitude))
    mean_daylight_hours_leap = np.array(monthly_mean_daylight_hours(latitude, 1984))
    
    # allocate the PET array we'll fill
    pet = np.full(monthly_temps.shape, fill_value)
    for year in range(monthly_temps.shape[0]):
        
        if calendar.isleap(start_year + year):
            month_days = _MONTH_DAYS_LEAP
            mean_daylight_hours = mean_daylight_hours_leap
        else:
            month_days = _MONTH_DAYS_NONLEAP
            mean_daylight_hours = mean_daylight_hours_nonleap

        # calculate the Thornthwaite equation
        pet[year, :] = 16 * (mean_daylight_hours / 12.0) * (month_days / 30.0) * ((10.0 * monthly_temps[year, :] / I) ** a)
    
    # reshape the dataset from (years, 12) into (months) 
    pet = pet.reshape(-1)
    
    # pull off the empty months in order to match the size of the original input dataset array
    if final_year_empty_months > 0:
        pet = np.delete(pet, np.s_[-final_year_empty_months:], 0)
    
    return pet
