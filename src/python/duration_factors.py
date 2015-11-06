from stack import Stack
import math
import numpy as np
import logging
import numba

# set up a global logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

MISSING_VALUE = -999.9

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def least_squares(x,  # array of integers
                  y,  # array of doubles
                  n,  # int, length of array x
                  wet_or_dry):  # boolean

    correlation = 0.0
    c_tol = 0.85
    max_value = 0.0
    max_diff = 0.0
    max_i = 0
    sumX = 0.0
    sumY = 0.0
    sumX2 = 0.0
    sumY2 = 0.0
    sumXY = 0.0
    for i in range(n):

        # make sure we don't use NaN values
        if np.isnan(y[i]):
            continue

        this_x = float(x[i])
        this_y = y[i]
        sumX += this_x
        sumY += this_y
        sumX2 += this_x * this_x
        sumY2 += this_y * this_y
        sumXY += this_x * this_y

    SSX = sumX2 - (sumX * sumX) / n
    SSY = sumY2 - (sumY * sumY) / n
    SSXY = sumXY - (sumX * sumY) / n
    if (SSX > 0.0) and (SSY > 0.0):
        correlation = SSXY / (math.sqrt(SSX) * math.sqrt(SSY))
    else:
        logger.warn('SSX: {ssx}  SSY: {ssy}'.format(ssx=SSX, ssy=SSY))
    i = n - 1

    # if we're dealing with wet conditions then we want to be using positive numbers, and for dry conditions
    # then we want to be using negative numbers, so we introduce a sign variable to facilitate this
    if wet_or_dry == 'dry':
        sign = -1
    else:
        sign = 1

    while ((sign * correlation) < c_tol) and (i > 3):
        # when the correlation is off, it appears better to
        # take the earlier sums rather than the later ones.
        this_x = float(x[i])
        this_y = y[i]
        sumX -= this_x
        sumY -= this_y
        sumX2 -= this_x * this_x
        sumY2 -= this_y * this_y
        sumXY -= this_x * this_y
        SSX = sumX2 - (sumX * sumX) / i
        SSY = sumY2 - (sumY * sumY) / i
        SSXY = sumXY - (sumX * sumY) / i
        if (SSX > 0.0) and (SSY > 0.0):
            correlation = SSXY / (math.sqrt(SSX) * math.sqrt(SSY))
        else:
            logger.warn('SSX: {ssx}  SSY: {ssy}'.format(ssx=SSX, ssy=SSY))
        i -= 1

    least_squares_slope = SSXY / SSX
    for j in range(i + 1):

        if sign * (y[j] - (least_squares_slope * x[j])) > (sign * max_diff):

            max_diff = y[j] - least_squares_slope * x[j];
            max_i = j;
            max_value = y[j];

    least_squares_intercept = max_value - least_squares_slope * x[max_i]

    return least_squares_slope, least_squares_intercept

#-----------------------------------------------------------------------------------------------------------------------
'''
    /*
     * TODO document with reference to the relevant sections of the Palmer and/or Wells papers
     *
     */
    //calculates m and b, which are used to calculated X(i)
    //based on the Z index.  These constants will determine the
    //weight that the previous PDSI value and the current Z index
    //will have on the current PDSI value.  This is done by finding
    //several of the driest periods at this station and assuming that
    //those periods represents an extreme drought.  Then a linear
    //regression is done to determine the relationship between length
    //of a dry (or wet) spell and the accumulated Z index during that
    //same period.
    //
    //it appears that there needs to be a different weight given to
    //negative and positive Z values, so the variable 'sign' will
    //determine whether the driest or wettest periods are looked at.
'''
#@numba.jit
def compute_duration_factors(zindex_values,
                             calibration_start_year,
                             calibration_end_year,
                             input_start_year,
                             periods_per_year=12,
                             month_scale=1):

    '''
    :param zindex_values: array with dimensions (lons, lats, times), times are months, in full year increments 
                          (i.e. len(times) should be evenly divisible by 12)
    :param calibration_start_year: 
    :param calibration_end_year:
    :param input_start_year:
    :param periods_per_year: the number of periods we want to compute, 12 for monthly, 4 for seasonal, etc.
    :param month_scale:   
    '''
    month_scales = [3, 6, 9, 12, 18, 24, 30, 36, 42, 48]

    duration_factors = np.full((zindex_values.shape[0], zindex_values.shape[1], 2, 2), -999.9)
    for lon in range(zindex_values.shape[0]):
        for lat in range(zindex_values.shape[1]):
            for wet_or_dry in ['wet', 'dry']:

                logger.info('Computing {wet_or_dry} factors for lon/lat: {lon}/{lat}'.format(wet_or_dry=wet_or_dry, lon=lon, lat=lat))

                z_sums = np.zeros((len(month_scales), ))
                for i, month_scale in enumerate(month_scales):

                    try:
                        z_sums[i] = get_z_sum(month_scale,
                                              wet_or_dry,
                                              zindex_values[lon][lat],
                                              periods_per_year,
                                              calibration_start_year,
                                              calibration_end_year,
                                              input_start_year)
                    except Exception:
                        logger.warn('Failed to get the {wet_or_dry} Z sums for lon/lat: {lon}/{lat}'.format(wet_or_dry=wet_or_dry, lon=lon, lat=lat))
                        raise

                # get the slope and intercept for the duration factor for this lon/lat using least squares regression
                try:
                    slope, intercept = least_squares(month_scales, z_sums, len(month_scales), wet_or_dry)
                except Exception:
                    logger.warn('Failed to get the least squares {wet_or_dry} slope and intercept for lon/lat: {lon}/{lat}'.format(wet_or_dry=wet_or_dry, lon=lon, lat=lat))
                    raise

                # if we're dealing with wet conditions then we want to be using positive numbers, and if dry conditions then
                # we need to be using negative numbers, so we use a PDSI limit of 4 on the wet side and -4 on the dry side
                pdsiLimit = 4.0  # WET
                wet_or_dry_index = 0
                if (wet_or_dry == 'dry'):
                    pdsiLimit = -4.0
                    wet_or_dry_index = 1

                # now divide slope and intercept by 4 or -4 because that line represents a PDSI of either 4.0 or -4.0
                slope = slope / pdsiLimit
                intercept = intercept / pdsiLimit

                duration_factors[lon, lat, wet_or_dry_index, 0] = slope
                duration_factors[lon, lat, wet_or_dry_index, 1] = intercept

    # we now have a duration factors array with a 2 x 2 array for each lon/lat point, where
    #     duration_factors[lon][lat][0][0] = wet slope for lon/lat
    #     duration_factors[lon][lat][0][1] = wet intercept for lon/lat
    #     duration_factors[lon][lat][1][0] = dry slope for lon/lat
    #     duration_factors[lon][lat][1][1] = dry intercept for lon/lat
    return duration_factors

#-----------------------------------------------------------------------------------------------------------------------
@numba.jit
def get_z_sum(interval_length,
              wet_or_dry,
              zindex_values,
              periods_per_year,
              calibration_start_year,
              calibration_end_year,
              input_start_year):

        tempZ = Stack()
        values_to_sum = Stack()
        summed_values = Stack()

        for zindex in zindex_values:

            # we need to skip Z-index values from the list if they don't exist, this can result from empty months in the final year of the data set
            if not np.isnan(zindex):
                tempZ.push(zindex)

        # remove periods before the start of the calibration interval
        initial_calibration_period_index = (calibration_start_year - input_start_year) * periods_per_year
        for i in range(initial_calibration_period_index):

            if tempZ.size > 0:
                tempZ.pop()
            else:
                break

        calibration_periods_left = (calibration_end_year - calibration_start_year + 1) * periods_per_year;

        # get the first interval length of values from the end of the calibration period working backwards, creating the first sum_value of interval periods
        sum_value = 0
        for i in range(interval_length):
            if tempZ.isEmpty():
                i = interval_length
            else:
                # pull a value off the end of the list
                z = tempZ.pop()

                # reduce the remaining number of calibration interval periods we have left to process
                calibration_periods_left -= 1

                '''
                /* assumes that nCalibrationPeriods is >= length, reasonable
                 ** This is a reasonable assumption and does not hurt if
                 ** anything if false--just calibrate over a slightly longer
                 ** interval, which is already way too short in that case */
                '''
                if (not np.isnan(z) and (MISSING_VALUE != z)):

                    # add to the sum_value
                    sum_value += z

                    # add to the array of values we've used for the initial sum_value
                    values_to_sum.add_first(z)

                else:
                    # reduce the loop counter so we don't skip a calibration interval period
                    i -= 1

        # if we're dealing with wet conditions then we want to be using positive numbers, and if dry conditions
        # then we need to be using negative numbers, so we introduce a sign variable to help with this
        if wet_or_dry == 'wet':
            sign = 1
        else:  # dry
            sign =-1

        # now for each remaining Z value, recalculate the sum_value based on last value in the list to sum_value and the next Z value
        max_sum = sum_value
        summed_values.add_first(sum_value)
        while (tempZ.size > 0) and (calibration_periods_left > 0):

            # take the next Z-index value off the end of the list
            z = tempZ.pop()

            # reduce by one period for each removal
            calibration_periods_left -= 1

            if not np.isnan(z) and (MISSING_VALUE != z) and not values_to_sum.isEmpty():

                # come up with a new sum_value for this new group of values to sum_value

                # remove the last value from both the sum_value and the values to sum_value array
                sum_value -= values_to_sum.peek()
                values_to_sum.pop()

                # add to the sum_value
                sum_value += z

                # update the values to sum_value and summed values lists
                values_to_sum.add_first(z);
                summed_values.add_first(sum_value);

            # update the maximum sum_value value if we have a new max
            if (sign * sum_value) > (sign * max_sum):
                max_sum = sum_value

        # highest reasonable is the highest (or lowest) value that is not due to some freak anomaly in the data.
        # "freak anomaly" is defined as a value that is either
        #   1) 25% higher than the 98th percentile
        #   2) 25% lower than the 2nd percentile
        if wet_or_dry == 'wet':
            safe_percentile_index = int(summed_values.size * 0.98)
        else:  # DRY
            safe_percentile_index = int(summed_values.size * 0.02)

        # sort the list of sums into ascending order and get the sum_value value referenced by the safe percentile index
        summed_values_list = summed_values.sorted_list()
        sum_at_safe_percentile = summed_values_list[safe_percentile_index]

        # find the highest reasonable value out of the summed values
        highest_reasonable_value = 0.0
        reasonable_tolerance_ratio = 1.25
        while (summed_values.size > 0):
            sum_value = summed_values.pop()
            if (sign * sum_value > 0):
                if ((sum_value / sum_at_safe_percentile) < reasonable_tolerance_ratio):
                    if (sign * sum_value > sign * highest_reasonable_value):
                        highest_reasonable_value = sum_value

        if wet_or_dry == 'wet':
            return highest_reasonable_value
        else:  # DRY
            return max_sum