import numpy as np
import logging

# set up a global logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 
# #     # original code transposed the arrays before returning them, not sure if this is necessary with Python/numpy
# #     alp = alp'
# #     bet = bet'
# #     gam = gam'
# #     delta = delta'
#     
#     return alp, bet, gam, delta

#--------------------------------------------------------------------------------------
def get_coefficients_from_sums(PESUM, ETSUM, RSUM, PRSUM, SPSUM, ROSUM, PLSUM, TLSUM):

    # create "zero limits", i.e. values to use when handling zero or missing values in the following calculations
    two_limits = [1.0, 0.0]
    single_limit = [0.0]

    # divide and limit actuals over potentials    
    alpha = divide_and_limit(ETSUM, PESUM, two_limits)
    beta = divide_and_limit(RSUM, PRSUM, two_limits)
    gamma = divide_and_limit(ROSUM, SPSUM, two_limits)
    delta = divide_and_limit(TLSUM, PLSUM, single_limit)
    
#     # original Matlab code transposed the arrays before returning them, not sure if this is necessary with Python/numpy
#     alp = alp'
#     bet = bet'
#     gam = gam'
#     delta = delta'

    return alpha, beta, gamma, delta

#---------------------------------------------------------------------------
def divide_and_limit(numerators, denominators, zero_limits):

    # make sure we have compatible arrays for the following computations    
    if denominators.shape != numerators.shape:
        message = 'numerator and denominator arrays do not share the same shape'
        logger.error(message)
        raise ValueError(message)
    
    # get the original shape so that later we can reshape the arrays 
    # that we'll flatten back into their original shape
    original_shape = denominators.shape
    denominators = denominators.flatten()
    numerators = numerators.flatten()
    
    # create an array of values corresponding to the shape of the input arrays
    results = np.full(denominators.shape, np.NAN)
    
    # get a column vector of indices where denominator is not zero, 
    # so as to avoid divide by zero in the following calculation
    not_zeros = np.where(denominators != 0)
    if len(not_zeros[0]) > 0:
        results[not_zeros] = numerators[not_zeros] / denominators[not_zeros]
        
    # get array of index values corresponding to the denominators array, 
    # for example if array is 4 elements long the we get an indices array: [0, 1, 2, 3]
    index_values = np.array(range(len(denominators)), np.int)
    
    # perform an XOR on the array indices and the 'not zeros' array of indices 
    # to get the indices where the value is zero
    zeros = np.setxor1d(index_values, not_zeros)
    if zeros.size > 0:

        # we have a zero denominator sum value so we can't perform the normal calculation at these points, 
        # so we limit the value to the zero limits
        
        if len(zero_limits) == 1:
        
            results[zeros] = zero_limits[0]
        
        elif len(zero_limits) == 2:
            
            # find indices where the value is zero, set the value at these indices to the first zero limit 
            limits = np.where(numerators[zeros] == 0)
            if limits[0].size > 0:
                results[zeros[limits]] = zero_limits[0]
    
            # find indices where the value is not zero, set the value at these indices to the second zero limit
            limits = np.where(numerators[zeros] != 0)
            if limits[0].size > 0:
                results[zeros[limits]] = zero_limits[1]
            
        else:
            message = 'Invalid zero limits argument, must contain 1 or 2 values'
            logger.error(message)
            raise valueError(message)

    # reshape the results back to our original shape and return
    return np.reshape(results, original_shape)

#----------------------------------------------------------------------------------        
if __name__ == '__main__':

    numers = np.array([[2., 3.], [8., 6]])
    denoms = np.array([[1., 6.], [4., 12]])
    results = divide_and_limit(numers, denoms, [1.0, 0.0])
    numers[0,1] = 0
    results = divide_and_limit(numers, denoms, [1.0, 0.0])
    denoms[1,0] = 0
    results = divide_and_limit(numers, denoms, [1.0, 0.0])
    pass
    