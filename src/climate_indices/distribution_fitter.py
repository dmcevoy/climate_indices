from __future__ import division
import logging
from math import exp, lgamma, log, pi, sqrt
from numba import float64, int32, jit
import numpy as np
from scipy.special import gammainc, gammaincc

# set up a basic, global logger
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d  %H:%M:%S')
logger = logging.getLogger(__name__)

#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:,:](float64[:], int32))
def gamma_parameters(monthly_values,
                     scale_months):
    '''
    TODO
    
    :param monthly_values: 
    :param scale_months: 
    :return:  
    '''
    
    # allocate the array of gamma parameters we'll return
    gamma_parameters = np.full((12, 3), np.nan, dtype=np.float64)
    
    # get sliding sums at the specified scale 
    summed_monthly_values = get_sliding_sums(monthly_values, scale_months)

    # process each calendar month's values separately
    for i in range(12):
        
        # get the values for the calendar month
        calendar_month_sums = summed_monthly_values[i::12]
        
        # strip out all the NaN values
        calendar_month_sums = calendar_month_sums[np.logical_not(np.isnan(calendar_month_sums))]
        
        # get the non-zero values only (resulting array will still contain NaNs if present)
        nonzero_calendar_month_values = calendar_month_sums[np.nonzero(calendar_month_sums)]

        # determine how many zeros there were
        number_of_sums = calendar_month_sums.shape[0]
        number_of_nonzeros = nonzero_calendar_month_values.shape[0]
        number_of_zeros = number_of_sums - number_of_nonzeros
        
        # calculate the probability of zero, the first gamma parameter
        probability_of_zero = number_of_zeros / number_of_sums
        gamma_parameters[i, 0] = probability_of_zero
        
        mean = np.mean(nonzero_calendar_month_values)
        
        if (number_of_nonzeros == 1) or (probability_of_zero == 1.0):
            gamma_parameters[i, 1] = mean  # beta
            gamma_parameters[i, 2] = 1.0   # gamma
        
        else:
            # use MLE 
            log_sum = np.nansum(np.log(nonzero_calendar_month_values))
            alpha = log(mean) - (log_sum / number_of_nonzeros)
            gamma_parameters[i, 2] = (1.0 + sqrt(1.0 + ((4.0 * alpha) / 3.0))) / (4.0 * alpha)  # gamma
            gamma_parameters[i, 1] = mean / gamma_parameters[i, 2]  # beta
    
    return gamma_parameters        

#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:](float64[:], int32, float64, float64))
def fit_to_gamma(values,
                 scale_months,
                 lower_limit,
                 upper_limit):
    '''
    TODO
    
    :param values: 
    :param scale_months: 
    :param lower_limit: 
    :param upper_limit:
    :return:  
    '''
    
    # compute gamma parameters using the specified month scale
    gamma_values = gamma_parameters(values, scale_months)
    
    # replace the values array with sliding sums at the specified month scale
    values = get_sliding_sums(values, scale_months)
    
    # replace the sums stored in the values array with fitted values
    probability = 0.0
    for month_index in range(scale_months - 1, len(values)):
    
        calendarMonth = month_index % 12
        if values[month_index] != np.nan:
        
            # compute the probability
            probability = gamma_cdf(gamma_values[calendarMonth, 1],
                                    gamma_values[calendarMonth, 2],
                                    gamma_values[calendarMonth, 0],
                                    values[month_index])

            # convert the probability to a fitted value
            values[month_index] = inv_normal(probability)
        
    # return the fitted values clipped to the specified upper and lower limits
    return np.clip(values, lower_limit, upper_limit)
        
#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64, float64, float64, float64))
def gamma_cdf(beta,
              gamma,
              pzero,
              x):
    '''
    Compute the probability of alpha being less than or equal to a value using incomplete gamma parameters.
    
    :param beta:
    :param gamma:
    :param pzero:
    :param x:    
    :return: probability of alpha <= x
    '''

    if x <= 0.0:
        return (pzero)
    else:
        return (pzero + ((1.0 - pzero) * gammap(gamma, x / beta)))

#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64, float64))
def gammap(gamma,
           x):
    '''
    TODO
    
    :param gamma: 
    :param x:
    :return:  
    '''

    if x < (gamma + 1.0):
    
        return gammainc(gamma, x)
    
    else:
    
        return 1.0 - gammaincc(gamma, x)
       
#-----------------------------------------------------------------------------------------------------------------------
# @profile
# #@jit(float64(float64, float64))
# def gammser(a, 
#             x):
#     '''
#     TODO
#     
#     :param a: 
#     :param x:
#     :return:  
#     '''
# 
#     epsilon = 3.0e-7
#     gln = lgamma(a)
# 
#     if x == 0.0:
#         return 0.0
# 
#     ap = a
#     delta_sum = 1.0 / a
#     delta = delta_sum
# 
#     for _ in range(100):
# 
#         ap += 1
#         delta *= x / ap
#         delta_sum += delta
#         if abs(delta) < (epsilon * abs(delta_sum)):
#         
#             # TODO can we instead break the loop here and fall down to the final return statement, since it's equivalent?
#             return delta_sum * exp((-x + (a * log(x))) - gln)
# 
#     return delta_sum * exp((-x + (a * log(x))) - gln)

#-----------------------------------------------------------------------------------------------------------------------
# @profile
# #@jit(float64(float64, float64))
# def gammcf(a,
#            x):
#     '''
#     TODO
#     
#     :param a: 
#     :param x:
#     :return:  
#     '''
# 
#     g = 0.0
#     n = 0
#     epsilon = 3.0e-7
#     gln = lgamma(a)
#     gold = 0.0
#     a0 = 1.0
#     a1 = x
#     b0 = 0.0
#     b1 = 1.0
#     fac = 1.0
#     for n in range(1, 100):
#     
#         an = n
#         ana = an - a
#         a0 = (a1 + (a0 * ana)) * fac
#         b0 = (b1 + (b0 * ana)) * fac
#         anf = an * fac
#         a1 = (x * a0) + (anf * a1)
#         b1 = (x * b0) + (anf * b1)
#         if a1 != 0.0:
#         
#             fac = 1.0 / a1
#             g = b1 * fac
#             if abs((g - gold) / g) < epsilon:
#                 
#                 # TODO can we instead just break the loop here and fall down into the final return statement, which is equivalent?
#                 return g * exp((-x + (a * log(x))) - gln)
#             
#             gold = g
#         
#     return g * exp((-x + (a * log(x))) - gln)

#-----------------------------------------------------------------------------------------------------------------------
# @jit(float64(float64))
# def lgamma(x):
#     
#     '''
#     TODO
#     
#     :param x: 
#     :return:  
#     '''
#     
#     if x < 0.0:
#         result = lgamma(1.0)
#     else:
#         tmp = ((x - 0.5) * log(x + 4.5)) - (x + 4.5)
#         ser = (((((1.0 + (76.18009173 / (x + 0))) - (86.50532033 / (x + 1))) + (24.01409822 / (x + 2))) - (1.231739516 / (x + 3))) + (0.00120858003 / (x + 4))) - \
#                      (0.00000536382 / (x + 5))
#         result = tmp + log(ser * sqrt(2 * pi))
#     return result

#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64))
def inv_normal(prob):
    '''
    See Abromowitz and Stegun _Handbook of Mathematical Functions_, p. 933
    
    :param prob: 
    :return: 
    '''

    t = 0.0
    minus = 0.0
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    if prob > 0.5:
        minus = 1.0
        prob = 1.0 - prob
    else:
        minus = -1.0

    if prob < 0.0:
        return 0.0
    elif prob == 0.0:
        return np.nan
    else:
        t = sqrt(log(1.0 / (prob * prob)))
        return (minus * (t - (((((c2 * t) + c1) * t) + c0) / ((((((d3 * t) + d2) * t) + d1) * t) + 1.0))))

#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:](float64[:], int32, float64, float64, int32, int32, int32, int32))
def fit_to_pearson(data,                    
                   scale_months,
                   lower_limit,
                   upper_limit,
                   data_start_year,
                   data_end_year,
                   calibration_start_year,
                   calibration_end_year):
    '''
    TODO
    
    :param values: 
    :param scale_months: 
    :param lower_limit: 
    :param upper_limit:
    :param data_start_year: 
    :param data_end_year: 
    :param calibration_start_year: 
    :param calibration_end_year:
    :return:  
    '''

    # make sure data is in complete year multiples, otherwise pad the end of the monthly array with enough months to fill out the final year
    original_length = data.size
    remaining_months = 12 - (original_length % 12)
    if remaining_months > 0:
        data = np.append(data, np.full((remaining_months,), np.nan))
        
    # calculate the "sliding sums" for the entire period of record
    summed_values = get_sliding_sums(data, scale_months)

    # compute the values we'll use to fit to the Pearson Type III distribution
    monthly_pearson_values = compute_pearson_params(data, scale_months, data_start_year, data_end_year, 
                                                    calibration_start_year, calibration_end_year)
    
    # allocate the array of values we'll eventually return, all values initialized to the fill value
    fitted_values = np.full(data.shape, np.nan)

    # compute Pearson CDF -> probability values -> fitted values for the entire period of record
    probability_of_zero = 0.0
    probability_value = 0.0
    pearson_values = np.zeros((4,))
    for i in range(scale_months - 1, len(data)):

        month_index = i % 12
        pearson_values[0] = monthly_pearson_values[1, month_index]
        pearson_values[1] = monthly_pearson_values[2, month_index]
        pearson_values[2] = monthly_pearson_values[3, month_index]
        pearson_values[3] = monthly_pearson_values[0, month_index]
        probability_of_zero = pearson_values[3]

        # only fit to the distribution if we don't have a fill value for the current month's scale sum
        if summed_values[i] != np.nan:

            # get the Pearson Type III cumulative density function value
            p3Cdf = 0.0
            
            # handle trace amounts as a special case
            if summed_values[i] < 0.0005:
            
                if probability_of_zero > 0.0:
                
                    p3Cdf = 0.0
                
                else:
                
                    p3Cdf = 0.0005  # minimum probability
                
            else:
            
                # calculate the CDF value for the current month's summed value
                p3Cdf = pearson3cdf(summed_values[i], pearson_values)
            

            if p3Cdf != np.nan:
            
                # calculate the probability value, clipped between 0 and 1
                probability_value = np.clip((probability_of_zero + ((1.0 - probability_of_zero) * p3Cdf)), 0.0, 1.0)

                # the fitted value is the quantile of the probability value
                fitted_values[i] = quantile(probability_value)
                
    # clip off the remaining months as well as limit the upper and lower range of the results
    return np.clip(fitted_values[:original_length], lower_limit, upper_limit)

#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:](float64[:], int32))
def get_sliding_sums(values,
                     number_of_values_to_sum):
    '''
    Get the valid sliding summations using 1-D convolution. The initial (number_of_values_to_sum - 1) elements 
    of the result array will be padded with np.NaN values.
    
    :param values: the array of values over which we'll compute sliding sums
    :param number_of_values_to_sum: the number of values for which each sliding summation will encompass, for example if
            this value is 3 then the first two elements of the output array will contain the pad value and the third 
            element of the output array will contain the sum of the first three elements, and so on 
    :return: an array of sliding sums, equal in length to the input values array, left padded with NaN values  
    '''
    # get the valid sliding summations with 1D convolution
    sliding_sums = np.convolve(values, np.ones(number_of_values_to_sum), mode='valid')
    
    # pad the first (n - 1) elements of the array with NaN values
    return np.hstack(([np.nan]*(number_of_values_to_sum - 1), sliding_sums))

#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64, float64[:]))
def pearson3cdf(value,
                pearson_params):
    '''
    TODO
    
    :param value:
    :param pearson_params:
    :return 
    '''

    # it's only possible to make the calculation if the second Pearson parameter is above zero
    if pearson_params[1] <= 0.0:
    
        logger.info("The second Pearson parameter is less than or equal to zero, invalid for the CDF calculation, returning the missing/fill value")
        return np.nan
    
    result = 0
    skew = pearson_params[2]
    if abs(skew) <= 1e-6:
    
        z = (value - pearson_params[0]) / pearson_params[1]
        return 0.5 + (0.5 * error_function(z * sqrt(0.5)))
    
    alpha = 4.0 / (skew * skew)
    x = ((2.0 * (value - pearson_params[0])) / (pearson_params[1] * skew)) + alpha
    if x > 0:
    
        result = gammainc(alpha, x) # was GammaFunction.incompleteGammaP(alpha, x)
        if skew < 0.0:
        
            result = 1.0 - result
        
    else:
    
        # calculate the lowest possible value that will fit the distribution precip (i.e. Z = 0)
        minimumPossibleValue = pearson_params[0] - ((alpha * pearson_params[1] * skew) / 2.0)
        if value <= minimumPossibleValue:
        
            result = 0.0005  # minimum probability
        
        else:
        
            result = np.nan

    return result

#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64))
def error_function(value):
    '''
    TODO
    
    :param value:
    :return:  
    '''
    
    result = 0.0
    if value != 0.0:

        absValue = abs(value)

        if absValue > 6.25:
            if value < 0:
                result = -1.0
            else:
                result = 1.0
        else:
            exponential = exp(value * value * (-1))
            sqrtOfTwo = sqrt(2.0)
            zz = abs(value * sqrtOfTwo)
            if absValue > 5.0:
                # alternative error function calculation for when the input value is in the critical range
                result = exponential * (sqrtOfTwo / pi) / \
                         (absValue + 1 / (zz + 2 / (zz + 3 / (zz + 4 / (zz + 0.65)))))

            else:
                # coefficients of rational-function approximation
                P0 = 220.2068679123761
                P1 = 221.2135961699311
                P2 = 112.0792914978709
                P3 = 33.91286607838300
                P4 = 6.373962203531650
                P5 = 0.7003830644436881
                P6 = 0.03526249659989109
                Q0 = 440.4137358247522
                Q1 = 793.8265125199484
                Q2 = 637.3336333788311
                Q3 = 296.5642487796737
                Q4 = 86.78073220294608
                Q5 = 16.06417757920695
                Q6 = 1.755667163182642
                Q7 = 0.08838834764831844

                # calculate the error function from the input value and constant values
                result = exponential * ((((((P6 * zz + P5) * zz + P4) * zz + P3) * zz + P2) * zz + P1) * zz + P0) /  \
                         (((((((Q7 * zz + Q6) * zz + Q5) * zz + Q4) * zz + Q3) * zz + Q2) * zz + Q1) * zz + Q0)

            if value > 0.0:
                result = 1 - result
            elif value < 0:
                result = result - 1.0

    return result

#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64(float64))
def quantile(probability_value):
    '''
    TODO
    :param probability_value:
    :return
    '''
    
    Q = probability_value - 0.5
    A0 = 3.38713287279636661
    A1 = 133.141667891784377
    A2 = 1971.59095030655144
    A3 = 13731.6937655094611
    A4 = 45921.9539315498715
    A5 = 67265.7709270087009
    A6 = 33430.5755835881281
    A7 = 2509.08092873012267
    B1 = 42.3133307016009113
    B2 = 687.187007492057908
    B3 = 5394.19602142475111
    B4 = 21213.7943015865959
    B5 = 39307.8958000927106
    B6 = 28729.0857357219427
    B7 = 5226.49527885285456
    C0 = 1.42343711074968358
    C1 = 4.63033784615654530
    C2 = 5.76949722146069141
    C3 = 3.64784832476320461
    C4 = 1.27045825245236838
    C5 = 0.241780725177450612
    C6 = 0.0227238449892691846
    C7 = 0.000774545014278341408
    D1 = 2.05319162663775882
    D2 = 1.67638483018380385
    D3 = 0.689767334985100005
    D4 = 0.148103976427480075
    D5 = 0.0151986665636164572
    D6 = 0.000547593808499534495
    D7 = 0.00000000105075007164441684
    E0 = 6.65790464350110378
    E1 = 5.46378491116411437
    E2 = 1.78482653991729133
    E3 = 0.296560571828504891
    E4 = 0.0265321895265761230
    E5 = 0.00124266094738807844
    E6 = 0.0000271155556874348758
    E7 = 0.000000201033439929228813
    F1 = 0.599832206555887938
    F2 = 0.136929880922735805
    F3 = 0.0148753612908506149
    F4 = 0.000786869131145613259
    F5 = 0.0000184631831751005468
    F6 = 0.000000142151175831644589
    F7 = 0.00000000000000204426310338993979

    if abs(Q) <= 0.425:
    
        # the case where 0.075 <= probability_value <= 0.925
        R = 0.180625 - (Q * Q)
        return (Q * ((((((((((((((A7 * R) + A6) * R) + A5) * R) + A4) * R) + A3) * R) + A2) * R) + A1) * R) + A0)) / \
               ((((((((((((((B7 * R) + B6) * R) + B5) * R) + B4) * R) + B3) * R) + B2) * R) + B1) * R) + 1.0)
    
    R = probability_value

    if Q >= 0:
    
        # the case where 0.075 > probability_value > 0.95 (since we won't get here unless the conditional above is bypassed)
        R = 1.0 - probability_value

    if R <= 0:
    
        # the case where 0 >= probability_value >= 1.0 HOWEVER 
        logger.debug('Invalid probability value: {}, R value turned out to be <= 0, using the missing/fill value'.format(probability_value))
        return np.nan

    result = 0.0
    R = sqrt(log(R) * -1.0)

    if R > 5:
        R = R - 5.0
        result = ((((((((((((((E7 * R) + E6) * R) + E5) * R) + E4) * R) + E3) * R) + E2) * R) + E1) * R) + E0) / \
                 ((((((((((((((F7 * R) + F6) * R) + F5) * R) + F4) * R) + F3) * R) + F2) * R) + F1) * R) + 1.0)
    
    else:
        R = R - 1.6
        result = ((((((((((((((C7 * R) + C6) * R) + C5) * R) + C4) * R) + C3) * R) + C2) * R) + C1) * R) + C0) / \
                 ((((((((((((((D7 * R) + D6) * R) + D5) * R) + D4) * R) + D3) * R) + D2) * R) + D1) * R) + 1.0)
    
    if Q < 0:
        result = -result

    return result
    
#----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:](float64[:], int32, int32, int32, int32, int32))
def compute_pearson_params(values, 
                           month_scale,
                           data_start_year, 
                           data_end_year,
                           calibration_start_year,
                           calibration_end_year):
    '''
    This function computes the distribution fitting parameters for the Pearson Type III distribution.
    
    :param month_scale: 
    :param data_start_year: 
    :param data_end_year: 
    :param calibration_start_year: 
    :param calibration_end_year: 
    :return: 
    '''
    # make sure that we've been passed in a flat (1-D) array of values    
    if len(values.shape) != 1:
        message = 'Values array has an invalid shape: {}'.format(values.shape)
        logger.error(message)
        raise ValueError(message)
    # make sure that the number of data values passed in is evenly divisible by 12, i.e. a multiple of full years
    if values.shape[0] % 12 != 0:
        message = 'Monthly values array has an invalid number of elements -- ' + \
                  'must be a multiple of 12 (complete years): {}'.format(values.shape[0])
        logger.error(message)
        raise ValueError(message)

    # calculate a list of sums for the entire period of record
    summed_values = get_sliding_sums(values, month_scale)
        
    # make sure that we have data within the full calibration period, otherwise use the full period of record
    if (calibration_start_year < data_start_year) or (calibration_end_year > data_end_year):
        logger.warn('Insufficient data for the specified calibration period ({1}-{2}), instead using the full period ' + 
                    'of record ({3}-{4})'.format(calibration_start_year, 
                                                 calibration_end_year, 
                                                 data_start_year, 
                                                 data_end_year))
        calibration_start_year = data_start_year
        calibration_end_year = data_end_year

    # get the year index for the calibration start and end years
    calibration_begin_index = (calibration_start_year - data_start_year)
    calibration_end_index = (calibration_end_year - data_start_year) + 1
    
    # reshape the array of sums from (months) to (years, calendar months)
    number_of_data_years = values.shape[0] / 12
    summed_values = np.reshape(summed_values, (number_of_data_years, 12))
    
    # now we'll use these sums to come up with the probability of zero and Pearson parameters for each calendar month
    monthly_pearson_values = np.zeros((4, 12,))
    for month_index in range(12):
    
        # get the sums for the month within the calibration period
        sums = summed_values[calibration_begin_index:calibration_end_index, month_index]

        # make sure we have at least four sum values that are both non-missing 
        # and non-zero, otherwise use the entire period of record
        number_of_zeros = sums.size - np.count_nonzero(sums)
        number_of_non_nans = np.count_nonzero(~np.isnan(sums))
        if (number_of_non_nans - number_of_zeros) < 4:
            sums = summed_values[:, month_index]
            number_of_zeros = sums.size - np.count_nonzero(sums)
            number_of_non_nans = np.count_nonzero(~np.isnan(sums))
            
        # calculate the probability of zero for the month
        probability_of_zero = 0.0
        if number_of_zeros > 0:

            probability_of_zero = number_of_zeros / number_of_non_nans
            
        # get the estimated L-moments, if we have more than 3 non-NaN/non-zero sum values
        if (number_of_non_nans - number_of_zeros) > 3:

            lmoments = estimate_lmoments(sums)

            # get the Pearson parameters for the month, based on the L-moments
            if (lmoments[1] > 0.0) and (abs(lmoments[2]) < 1.0):
                
                pearson_parameters = estimate_pearson_parameters(lmoments);
                monthly_pearson_values[0, month_index] = probability_of_zero
                monthly_pearson_values[1, month_index] = pearson_parameters[0]
                monthly_pearson_values[2, month_index] = pearson_parameters[1]
                monthly_pearson_values[3, month_index] = pearson_parameters[2]

    return monthly_pearson_values;

#-----------------------------------------------------------------------------------------------------------------------    
#@profile
@jit(float64[:](float64[:]))
def estimate_lmoments(values):

    '''
    Estimate sample L-moments, based on Fortran code written for inclusion in IBM Research Report RC20525,
    'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3' by J. R. M. Hosking, IBM Research Division,
    T. J. Watson Research Center, Yorktown Heights, NY 10598, Version 3 August 1996. This is a Python translation of
    the original Fortran subroutine named SAMLMR and it has been optimized for calculating the first three L-moments only. 
    
    :param values: 1-D (flattened) array of float values
    :return: estimate of the first three L-moments
    '''
    
    # we need to have at least 4 sum values in order to make a sample L-moments estimation
    number_of_values = np.count_nonzero(~np.isnan(values))
    if (number_of_values < 4):
        raise Exception("Insufficient number of values to perform sample L-moments estimation")
        
    # sort the values into ascending order
    values = np.sort(values)
    
    sums = np.zeros((3,))

    for i in range(1, number_of_values + 1):
        z = i
        term = values[i - 1]
        sums[0] = sums[0] + term
        for j in range(1, 3):
            z = z - 1
            term = term * z
            sums[j] = sums[j] + term
        
    y = float(number_of_values)
    z = float(number_of_values)
    sums[0] = sums[0] / z
    for j in range(1, 3):
        y = y - 1.0
        z = z * y
        sums[j] = sums[j] / z
    
    k = 3
    p0 = -1.0
    for _ in range(2):
        ak = float(k)
        p0 = -p0
        p = p0
        temp = p * sums[0]
        for i in range(1, k):
            ai = i
            p = -p * (ak + ai - 1.0) * (ak - ai) / (ai * ai)
            temp = temp + (p * sums[i])
        sums[k - 1] = temp
        k = k - 1
      
    lmoments = np.zeros((3,))  
    if (sums[1] != 0):
        lmoments[0] = sums[0]
        lmoments[1] = sums[1]
        lmoments[2] = sums[2] / sums[1]
        
    return lmoments
    
#-----------------------------------------------------------------------------------------------------------------------
#@profile
@jit(float64[:](float64[:]))
def estimate_pearson_parameters(lmoments):

    '''
    Estimate parameters via L-moments for the Pearson Type III distribution, based on Fortran code written 
    for inclusion in IBM Research Report RC20525, 'FORTRAN ROUTINES FOR USE WITH THE METHOD OF L-MOMENTS, VERSION 3' 
    by J. R. M. Hosking, IBM Research Division, T. J. Watson Research Center, Yorktown Heights, NY 10598
    This is a Python translation of the original Fortran subroutine named 'pearson3'.
    
    :param lmoments: 3-element, 1-D (flat) array containing the first three L-moments (lambda-1, lambda-2, and tau-3)
    :return 3-element, 1-D (flat) array containing the Pearson Type III parameters corresponding to the input L-moments 
    '''
    
    C1 = 0.2906
    C2 = 0.1882
    C3 = 0.0442
    D1 = 0.36067
    D2 = -0.59567
    D3 = 0.25361
    D4 = -2.78861
    D5 = 2.56096
    D6 = -0.77045
    T3 = abs(lmoments[2])
    
    # ensure the validity of the L-moments
    if ((lmoments[1] <= 0) or (T3 >= 1)):
        raise Exception("Unable to calculate Pearson parameters due to invalid L-moments")

    # initialize the output array    
    parameters = np.zeros((3,))
    parameters[0] = lmoments[0]
    
    if (T3 <= 1e-6):
        # skewness is effectively zero
        parameters[1] = lmoments[1] * sqrt(pi)

    else:
        if (T3 < 0.333333333):
            T = pi * 3 * T3 * T3
            alpha = (1.0 + (C1 * T)) / (T * (1.0 + (T * (C2 + (T * C3))))) 
        else:
            T = 1.0 - T3
            alpha = T * (D1 + (T * (D2 + (T * D3)))) / (1.0 + (T * (D4 + (T * (D5 + (T * D6))))))
            
        alpha_root = sqrt(alpha)
        beta = sqrt(pi) * lmoments[1] * exp(lgamma(alpha) - lgamma(alpha + 0.5))
        parameters[1] = beta * alpha_root
        if (lmoments[2] < 0):
            parameters[2] = -2.0 / alpha_root
        else:
            parameters[2] = 2.0 / alpha_root

    return parameters
