from __future__ import division
import logging
#import numba
from duration_factors import compute_duration_factors
from netcdf import open_datasets, load_data, extract_coords, write_dataset, close_datasets
import numpy as np
import water_balance_coefficients as wb


# set up a global logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#--------------------------------------------------------------------------------------
#@numba.jit(nopython=True)
def gettingout(U, Ze, nmonths, nmonthc, dow):

    US1 = 0
    US2 = 0
    for i in range(nmonths):
        US1 = US1 + U[nmonthc - nmonths + i]
        if (dow == 1) and (US1 < 0):
            US1 = 0
        if (dow == 0) and (US1 > 0):
            US1 = 0

    if nmonths > 1:
        for i in range(nmonths - 1):
            US2 = US2 + U[nmonthc - nmonths + i]
            if (dow == 1) and (US2 < 0):
                US2 = 0
            if (dow == 0) and (US2 > 0):
                US2 = 0

    Pe = 100 * (US1) / (Ze[nmonthc] + US2)
    if Pe > 100:
        Pe = 100
    if Pe < 0:
        Pe = 0
    return Pe

#--------------------------------------------------------------------------------------
#@numba.jit(nopython=True)
def compute_pdsi(Z):

    '''
    :param Z Z-index values for a location at each time step, 1-D array of values
    '''

    nmonths = 0
    X1 = np.zeros(Z.shape)
    X2 = X1
    X3 = X1
    Uw = X1
    Ud = X1
    Pe = X1
    PDSI = X1
    Ze = X1

    # start off not in dry or wet spell
    XX = Z[0] / 3
    if XX > 0:
        X1[0] = XX
    else:
        X2[0] = XX

    if abs(XX) >= 1:
        X3[0] = XX
        nmonths = 1
    PDSI[0] = XX

    montho = np.zeros_like(Z, dtype=int)

    # loop over time
    for i in range(1, Z.shape[0]):
        XX = Z[i] / 3
        Ud[i] = (XX * 3) - 0.15  # TODO why this value 0.15?
        Uw[i] = (XX * 3) + 0.15  # "
        if XX > 0:
            if X2[i - 1] < 0:
                X2[i] = X2[i - 1] * 0.897 + XX
            else:
                X2[i] = XX

            if (X2[i - 1] < -1) and (X3[i - 1] < -1):
                X2[i] = 0

            if X1[i - 1] > 0:
                X1[i] = X1[i - 1] * 0.897 + XX
            else:
                X1[i] = XX

            if (X2[i - 1] < -1) and (X3[i - 1] < -1):
                X2[i] = 0

            if X3[i - 1] != 0:
                if X3[i - 1] < 0:
                    Ze[i] = -2.691 * X3[i - 1] - 1.5
                    Pe[i] = gettingout(Uw, Ze, nmonths + 1, i, 1)

                    if Pe[i]==100:
                        nmonths = 0
                        X3[i] = 0

                        if X1[i]>1:
                            X3[i] = X1[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

                else:
                    Ze[i] = -2.691 * X3[i - 1] + 1.5
                    Pe[i] = gettingout(Ud, Ze, nmonths + 1, i, 0)

                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0

                        if X1[i] > 1:
                            X3[i] = X1[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

            else:
                if X1[i] > 0.5:
                    X3[i] = X1[i]
                    nmonths = 1

        else:
            if X2[i - 1] <0:
                X2[i] = X2[i - 1] * 0.897 + XX
            else:
                X2[i] = XX

            if X1[i - 1] > 0:
                X1[i] = X1[i - 1] * 0.897 + XX
            else:
                X1[i] = XX

            if X3[i - 1] != 0:

                if X3[i - 1] > 0:

                    Ze[i] = -2.691 * X3[i - 1] + 1.5
                    Pe[i] = gettingout(Ud, Ze, nmonths + 1, i, 0)

                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0

                        if X2[i] < -1:
                            X3[i] = X2[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

                else:

                    Ze[i] = -2.691 * X3[i - 1] - 1.5
                    Pe[i] = gettingout(Uw, Ze, nmonths + 1, i, 1)

                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0

                        if X2[i] > 1:
                            X3[i] = X2[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

            else:

                if X2[i] < -0.5:
                    X3[i] = X2[i]
                    nmonths = 1

        if Pe[i] < 100:

            if X3[i - 1] != 0:
                X3[i] = X3[i - 1] * 0.897 + XX
                nmonths = nmonths + 1

        # decide what PDSI is
        montho[i] = nmonths

        # option 1 no drought going on or being established, nmonths == 0
        if X1[i] < 0:
            X1[i] = 0
        if X2[i] > 0:
            X2[i] = 0

        if nmonths == 0:

            if X3[i] == 0:

                if X1[i] > -X2[i]:
                    PDSI[i] = X1[i]
                else:
                    PDSI[i] = X2[i]

                PDSI[i] = XX

        else:
            PDSI[i] = X3[i]

            if (PDSI[i] >= 1) and (nmonths > 1):
                X1[i] = 0
            if (PDSI[i] <=-1) and (nmonths>1):
                X2[i] = 0

    return PDSI

#--------------------------------------------------------------------------------------
#@numba.jit(nopython=True)
def compute_K_prime(DD, PE, R, RO, precip, L):

    '''
    Computes the climatic characteristic based on equations (1) and (2) in Wells et al 2004.

    :param DD: the moisture departures for a lon x lat grid
    :param PE:
    :param R:
    :param RO:
    :param precip:
    :param L:
    :return: K_prime, mean_DD
    '''

    mean_PE = np.nanmean(PE, axis=0)
    mean_R = np.nanmean(R, axis=0)
    mean_RO = np.nanmean(RO, axis=0)
    mean_precip = np.nanmean(precip, axis=0)
    mean_L = np.nanmean(L, axis=0)
#     mean_DD = np.nanmean(DD, axis=0)
    mean_DD = np.nanmean(DD.flatten())
    term = (mean_PE + mean_R + mean_RO) / (mean_precip + mean_L) + 2.8
    K_prime = (1.5 * np.log10(term / mean_DD))  + 0.5
#     return K_prime, mean_DD
    return K_prime

#--------------------------------------------------------------------------------------
#@numba.jit
def compute_K(K_prime,
              mean_DD):

    '''
    Computes the climatic characteristic based on equation (2) in Wells et al 2004.

    :param K_prime: the first approximation of K for a lon x lat grid
    :param mean_DD: monthly mean moisture departures, of the same dimensions as K_prime
    :return: climatic characteristic K, of the same dimensions as K_prime
    '''

    K = 17.67 * K_prime / np.sum(mean_DD * K_prime, axis=0)

    return K

#--------------------------------------------------------------------------------------
#@numba.jit
# previously named 'PDSI_wb_findnoloop
def perform_water_balance(PET, P, WCBOT, WCTOP, WCTOT, SS, SU, SP):

    """
    :param PET: potential evapotranspiration for a single time step for all lon/lat locations, 1-D array with length == lons * lats
    :param P: precipitation for a single time step for all lon/lat locations, 1-D array with length == lons * lats
    :param WCBOT: bottom layer water capacity
    :param WCTOP: top layer water capacity
    :param WCTOT: total water capacity
    :param SS: surface soil moisture
    :param SU: underlying soil moisture
    :param SP: amount of available moisture in both layers of the soil at the start of the month 
    :return:
    """

    # potential recharge, the amount of mositure required to bring the soil to field capacity
    PR = WCTOT - SP   # eq. (3) from Palmer 1965
    PRS = WCTOP - SS  # surface layer potential recharge
    PRU = WCBOT - SU  # underlying layer potential recharge
    
    # initialize the arrays we'll work with as containing all NaN values
    PL = np.full(len(P), np.NAN)
    ET = np.full(len(P), np.NAN)
    TL = np.full(len(P), np.NAN)
    RO = np.full(len(P), np.NAN)
    R = np.full(len(P), np.NAN)
    RU = np.full(len(P), np.NAN)
    RS = np.full(len(P), np.NAN)
    SSS = np.full(len(P), np.NAN)
    SSU = np.full(len(P), np.NAN)
    SL = np.full(len(P), np.NAN)
    UL = np.full(len(P), np.NAN)

    # get the indices of the values where the surface soil moisture exceeds PET, at these indices make the potential loss = PET
    f1 = np.nonzero(SS >= PET)
    if len(f1) > 0:
        PL[f1] = PET[f1]

    # get the indices of the values where the surface soil moisture is >= PET, at these indices make the PL = (demand * straw) + SS
    f2 = np.flatnonzero(SS < PET)
    if len(f2) > 0:
        straw = SU[f2] / WCTOT[f2]
        demand = PET[f2] - SS[f2]
        PL[f2] = (demand * straw) + SS[f2]

        # get the indices of the values where PL is >= SP, at these indices make the PL = SP (limit PL to SP)
        f4 = np.flatnonzero(PL[f2] > SP[f2])
        if len(f4) > 0:
            PL[f2[f4]] = SP[f2[f4]]

    # find the indices where precipitation exceeds PET
    test2 = np.flatnonzero(P>=PET)
    if len(test2) > 0:
        # precipitation exceeds PET
        ET[test2] = PET[test2]
        TL[test2] = 0.0
        test3 = np.flatnonzero((P[test2] - PET[test2]) > PRS[test2])
        if len(test3) > 0:
            # precipitation is sufficient to recharge both the under and upper layers
            RS[test2[test3]] = PRS[test2[test3]]
            SSS[test2[test3]] = WCTOP[test2[test3]]
            test4 = np.flatnonzero((P[test2[test3]] - PET[test2[test3]] - RS[test2[test3]]) < PRU[test2[test3]])
            if len(test4) > 0:
                # both layers can take the entire excess
                RU[test2[test3[test4]]] = P[test2[test3[test4]]] - PET[test2[test3[test4]]] - RS[test2[test3[test4]]]
                RO[test2[test3[test4]]] = 0.0

            test5 = np.setxor1d(np.array(range(len(test2[test3]))), test4) # some runoff occurs
            if len(test5) > 0:

                RU[test2[test3[test5]]] = WCBOT[test2[test3[test5]]] - SU[test2[test3[test5]]]
                RO[test2[test3[test5]]] = P[test2[test3[test5]]] - PET[test2[test3[test5]]] - RS[test2[test3[test5]]] - RU[test2[test3[test5]]]

            SSU[test2[test3]] = SU[test2[test3]] + RU[test2[test3]]
            R[test2[test3]] = RS[test2[test3]] + RU[test2[test3]]

        test6 = np.setxor1d(np.array(range(len(test2))), test3)
        if len(test6) > 0: # only the top layer is recharged
            R[test2[test6]] = P[test2[test6]] - PET[test2[test6]]
            SSS[test2[test6]] = SS[test2[test6]] + P[test2[test6]] - PET[test2[test6]]
            SSU[test2[test6]] = SU[test2[test6]]
            RO[test2[test6]] = 0.0

    # find the indices where R > PR, at those indices set R = PR
    f12 = np.flatnonzero(R > PR)
    if len(f12) > 0:
        R[f12] = PR[f12]

    testa = np.setxor1d(np.array(range(len(P))), test2)  # evaporation exceeds precipitation
    if len(testa) > 0:
        R[testa] = 0.0
        testb = np.flatnonzero(SS[testa] >= (PET[testa] - P[testa]))
        if len(testb) > 0:
            # evaporation from surface layer only
            SL[testa[testb]] = PET[testa[testb]] - P[testa[testb]]
            SSS[testa[testb]] = SS[testa[testb]] - SL[testa[testb]]
            UL[testa[testb]] = 0.0
            SSU[testa[testb]] = SU[testa[testb]]
        testc = np.setxor1d(np.array(range(len(testa))), testb)  # evaporation from both layers
        if len(testc) > 0:
            SL[testa[testc]] = SS[testa[testc]]
            SSS[testa[testc]] = 0.0
            straw = SU[testa[testc]] / WCTOT[testa[testc]]
            demand = PET[testa[testc]] - P[testa[testc]] - SL[testa[testc]]
            UL[testa[testc]] = demand * straw
            f4 = np.flatnonzero(UL[testa[testc]] > SU[testa[testc]])
            UL[testa[testc[f4]]] = SU[testa[testc[f4]]]
            SSU[testa[testc]] = SU[testa[testc]] - UL[testa[testc]]

        TL[testa] = SL[testa] + UL[testa]
        RO[testa] = 0.0
        ET[testa] = P[testa] + SL[testa] + UL[testa]
        f = np.flatnonzero(PET[testa] < ET[testa])
        if len(f) > 0:
            ET[testa[f]] = PET[testa[f]]

    # set values to NaN wherever the PET values are NaN
    f = np.flatnonzero(np.isnan(PET))
    if len(f) > 0:
        ET[f] = np.NAN
        R[f] = np.NAN
        RO[f] = np.NAN
        SSS[f] = np.NAN
        SSU[f] = np.NAN
        TL[f] = np.NAN

    # return
    return [PL, ET, TL, RO, R, SSS, SSU]

#--------------------------------------------------------------------------------------
def get_cafec_precip(precip_dataset,
                     pet_dataset,
                     soil_dataset,
                     times,
                     lons,
                     lats):

    """

    :rtype : object
    """

    # load data from variable named 'soil' into soil array
    soil = load_data(soil_dataset, 'soil', 0)
    # make sure we're not dealing with all NaN or fill values
    soil[soil <= -999] = np.NaN  # turn all fill values (assumed to be anything less than -999) into NaNs
    if np.all(np.isnan(soil)):
        raise ValueError('Missing soil constant values')
    soil = np.ndarray.flatten(soil) * 25.4  # soil is in inches, multiply by 25.4 to get units into mm

    # water capacity lower level, assumed to be the full available water capacity (minus one inch?)
    WCBOT = soil

    # water capacity upper level, assumed to be one inch (25.4 mm) at start
    WCTOP = np.full((len(soil),), 25.4)  # surface/top one inch (25.4 mm)

    # total soil moisture water capacity
    WCTOT = WCBOT + WCTOP

    # start of month soil moisture levels, in mm
    SS = WCTOP    # surface
    SU = WCBOT    # underlying
    SP = SS + SU  # total

    # allocate data arrays, with dimensionality (time, lon, lat)
    timelonlat_shape = (len(times), len(lons), len(lats),)
    petdat = np.full(timelonlat_shape, np.NaN)
    etdat = np.full(timelonlat_shape, np.NaN)
    spdat = np.full(timelonlat_shape, np.NaN)
    pldat = np.full(timelonlat_shape, np.NaN)
    tldat = np.full(timelonlat_shape, np.NaN)
    rdat = np.full(timelonlat_shape, np.NaN)
    prdat = np.full(timelonlat_shape, np.NaN)
    rodat = np.full(timelonlat_shape, np.NaN)
    prodat = np.full(timelonlat_shape, np.NaN)
    sssdat = np.full(timelonlat_shape, np.NaN)
    ssudat = np.full(timelonlat_shape, np.NaN)

    # go over each time step and perform an ongoing water balance accounting
    for time_index in range(len(times)):

        logger.info('Water balance accounting for time step {step}'.format(step=time_index))
        
        # load data for the current timestep into the precip array
        precip = load_data(precip_dataset, 'ppt', time_index)

        # make sure we're not dealing with all NaN or fill values
        precip[precip <= -999] = np.NaN  # turn all fill values (assumed to be anything less than -999) into NaNs
        if np.all(np.isnan(precip)):
            continue
        data_shape = precip.shape  # we expect this to be (1, #_of_lons, #_of_lats), this should be verified
        precip = np.ndarray.flatten(precip)

        # load data into pet array
        pet = load_data(pet_dataset, 'PET', time_index)
        # make sure our data shape is the same as that of the precipitation array
        if data_shape != pet.shape:
            message = 'Incompatible data shapes -- precipitation: ' + str(data_shape) + '  PET: ' + str(pet.shape)
            logger.error(message)
            raise ValueError(message)

        # make sure we're not dealing with all NaN or fill values
        pet[pet <= -999] = np.NaN  # turn all fill values (assumed to be anything less than -999) into NaNs
        if np.all(np.isnan(pet)):
            continue
        pet = np.ndarray.flatten(pet) * 25.4  # pet is in inches, multiply by 25.4 to get units into mm

        # we now have pet and precip as flat arrays of length (lats * lons)

        PR = WCTOT - SP  # from equation 3, Palmer 1965
        PRO = soil - PR  # from equation 5, Palmer 1965
        [PL, ET, TL, RO, R, SSS, SSU] = perform_water_balance(pet[:],
                                                              precip[:],
                                                              WCBOT[:],
                                                              WCTOP[:],
                                                              WCTOT[:],
                                                              SS[:],
                                                              SU[:],
                                                              SP[:])
        SS = SSS
        SU = SSU
        SP = SS + SU

        # we reshape the water balance data for this time step to (1, len(lons), len(lats)) and insert into the full time series arrays
        petdat[time_index:time_index + 1,:,:] = np.reshape(pet, data_shape)
        etdat[time_index:time_index + 1,:,:] = np.reshape(ET, data_shape)
        spdat[time_index:time_index + 1,:,:] = np.reshape(SP, data_shape)
        pldat[time_index:time_index + 1,:,:] = np.reshape(PL, data_shape)
        tldat[time_index:time_index + 1,:,:] = np.reshape(TL, data_shape)
        rdat[time_index:time_index + 1,:,:] = np.reshape(R, data_shape)
        prdat[time_index:time_index + 1,:,:] = np.reshape(PR, data_shape)
        rodat[time_index:time_index + 1,:,:] = np.reshape(RO, data_shape)
        prodat[time_index:time_index + 1,:,:] = np.reshape(PRO, data_shape)
        sssdat[time_index:time_index + 1,:,:] = np.reshape(SS, data_shape)
        ssudat[time_index:time_index + 1,:,:] = np.reshape(SU, data_shape)

    #TODO make sure ET, PET, R, PR, PR, PRO, L and PL all have equivalent dimensions

    # perform water balance coefficient calculations
    # TODO if we are doing a base/calibration period run then we will compute and later write these values,
    # but if we are doing an incremental run or for a period which is not the base/calibration period then
    # we should read those values from file instead

    # determine potential runoff, needed for later calculations of the gamma water balance coefficient
    PRO = soil - PR  # from equation 5, Palmer 1965

    # reshape the data arrays to the "full years" shape (with times separated into months and years dimensions)
    full_years_shape = get_full_years_shape(times, lons, lats)
    
    logger.info('\nReshaping the data arrays into (year, month, lon, lat)\n')

    # reshape the data arrays into the "full years" shape
    arrays = [etdat, petdat, rdat, prdat, rodat, prodat, spdat, tldat, pldat]
    for i, ary in enumerate(arrays):
        arrays[i] = np.reshape(ary, full_years_shape)
        
    logger.info('\nGetting the mean of the water balance values\n')

    # get the mean for each month, ignoring NaN values
    et_mean = np.nanmean(etdat, axis=0)
    pet_mean = np.nanmean(petdat, axis=0)
    r_mean = np.nanmean(rdat, axis=0)
    pr_mean = np.nanmean(prdat, axis=0)
    ro_mean = np.nanmean(rodat, axis=0)
    sp_mean = np.nanmean(spdat, axis=0)
    tl_mean = np.nanmean(tldat, axis=0)
    pl_mean = np.nanmean(pldat, axis=0)

    logger.info('\nGetting water balance coefficients\n')

    # TODO get the correct sums arrays in order to utilize the water balance 
    # coefficients function below (original Matlab contributed by WRCC/DRI, converted into Python)
    alphas, betas, gammas, deltas = wb.get_coefficients_from_sums(pet_mean, et_mean, r_mean, pr_mean, sp_mean, ro_mean, pl_mean, tl_mean) #TODO test/debug/validate this function
    
    # we now have alphas as the array of monthly coefficient of evapotranspiration values (eq. 7 of Palmer 1965),
    # betas as the array of monthly coefficient of recharge (eq. 7 of Palmer 1965), gammas as the array of
    # monthly coefficient of runoff (eq. 8 of Palmer 1965), and deltas as the array of monthly coefficient
    # of loss (eq. 8 of Palmer 1965)

    # calculate P-hat
    PHAT = (petdat * alphas) + (prdat * betas) + (spdat * gammas) - (pldat * deltas)

    return PHAT, petdat, rdat, rodat, tldat

#--------------------------------------------------------------------------------------
#@numba.jit
def compute_scpdsi(Z, wet_slope, wet_intercept, dry_slope, dry_intercept):

    """

    :param Z: array/list of Z-index values for each month (?)
    :param scalef: scaling factor array, dimensions: ?
    :return: PDSI
    """

    nmonths = 0
    X1 = np.zeros(len(Z))
    X2 = X1
    X3 = X1
    Uw = X1
    Ud = X1
    Pe = X1
    PDSI = X1
    Ze = X1

    # start off not in dry or wet spell

    XX = Z[0] / 3
    if XX > 0:
        X1[0] = XX
    else:
        X2[0] = XX

    if abs(XX) >= 1:
        X3[0] = XX
        nmonths = 1

    PDSI[0] = XX

    montho = np.zeros(Z.shape, dtype=np.int)

    # loop over time
    for i in range(1, len(Z)):

        XX = Z[i] / 3
        Ud[i] = Z[i] - 0.15
        Uw[i] = Z[i] + 0.15

        if XX > 0:

            if X2[i - 1] < 0:
                X2[i] = X2[i - 1] * wet_slope + Z[i] * dry_intercept
            else:
                X2[i] = Z[i] * dry_intercept

            if (X2[i - 1] < -1) and (X3[i - 1] < -1):
                X2[i] = 0

            if X1[i - 1] > 0:
                X1[i] = X1[i - 1] * dry_slope + Z[i] * dry_intercept
            else:
                X1[i] = Z[i] * dry_intercept

            if (X2[i - 1] < -1) and (X3[i - 1] < -1):
                X2[i] = 0

            if X3[i - 1] != 0:
                if X3[i - 1] < 0:

                    Ze[i] = -2.691 * X3[i - 1] - 1.5;
                    Pe[i] = gettingout(Uw, Ze, nmonths + 1, i, 1);

                    if Pe[i]==100:
                        nmonths = 0
                        X3[i] = 0
                        if X1[i] > 1:
                            X3[i] = X1[i]
                            nmonths=1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

                else:

                    Ze[i] = -2.691 * X3[i - 1] + 1.5
                    Pe[i] = gettingout(Ud, Ze, nmonths + 1, i, 0)

                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0
                        if X1[i] > 1:
                            X3[i] = X1[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

            else:
                if X1[i] > 0.5:
                    X3[i] = X1[i]
                    nmonths = 1

        else:
            if X2[i - 1] < 0:
                X2[i] = X2[i - 1] * wet_slope + Z[i] * wet_intercept
            else:
                X2[i] = Z[i] * wet_intercept

            if X1[i - 1] > 0:
                X1[i] = X1[i - 1] * dry_slope + Z[i] * wet_intercept
            else:
                X1[i] = Z[i] * wet_intercept

            if X3[i - 1] != 0:
                if X3[i - 1] > 0:
                    Ze[i] = -2.691 * X3[i - 1] + 1.5
                    Pe[i] = gettingout(Ud, Ze, nmonths + 1 , i, 0)
                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0
                        if X2[i] < -1:
                            X3[i] = X2[i]
                            nmonths = 1

                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0
                else:
                    Ze[i] = -2.691 * X3[i - 1] - 1.5
                    Pe[i] = gettingout(Uw, Ze, nmonths+1, i, 1)
                    if Pe[i] == 100:
                        nmonths = 0
                        X3[i] = 0
                        if X2[i] > 1:
                            X3[i] = X2[i]
                            nmonths = 1
                    if Pe[i] == 0:
                        X1[i] = 0
                        X2[i] = 0

            else:
                if X2[i] < -0.5:
                    X3[i] = X2[i]
                    nmonths = 1

        if Pe[i] < 100:
            if X3[i - 1] > 0:
                if Z[i] > 0:
                    X3[i] = X3[i - 1] * dry_slope + Z[i] * dry_intercept
                else:
                    X3[i] = X3[i - 1] * dry_slope + Z[i] * wet_intercept
            else:
                if Z[i] > 0:
                    X3[i] = X3[i - 1] * wet_slope + Z[i] * dry_intercept
                else:
                    X3[i] = X3[i - 1] * wet_slope + Z[i] * wet_intercept

            nmonths = nmonths + 1

        # decide what PDSI is
        montho[i] = nmonths

        # option 1 no drought ongoing nor being established, nmonths=0
        if X1[i] < 0:
            X1[i] = 0
        if X2[i] > 0:
            X2[i] = 0

        if nmonths == 0:
            if X3[i] == 0:
                if X1[i] > -X2[i]:
                    PDSI[i] = X1[i]
                else:
                    PDSI[i] = X2[i]

                PDSI[i] = XX  #TODO doesn't this just override whatever was done in the above conditional?
        else:
            PDSI[i] = X3[i]

            if (PDSI[i] >= 1) and (nmonths > 1):
                X1[i] = 0
            if (PDSI[i] <= -1) and (nmonths > 1):
                X2[i] = 0

    return PDSI

#--------------------------------------------------------------------------------------
#@numba.jit
def get_full_years_shape(times, lons, lats):
    '''
    This function makes a shape tuple for use when we need to reshape data arrays into (years, months, lons, lats),
    i.e. when we want to use the data by year/month.

    :param times:
    :param lons:
    :param lats:
    :return:
    '''

    remaining_months = 12 - (len(times) % 12)
    if remaining_months == 12:
        remaining_months = 0
    full_years = (len(times) + remaining_months) / 12
    return (full_years, 12, len(lons), len(lats),)

#--------------------------------------------------------------------------------------
def scpdsi(precip_file,
           precip_var_name,
           pet_file,
           soil_file,
           output_file_base):

    # open the NetCDF files as dataset objects
    datasets = open_datasets([precip_file, pet_file, soil_file])
    precip_dataset = datasets[0]
    pet_dataset = datasets[1]
    soil_dataset = datasets[2]

    # get the coordinate values for the data set, all of which should match in size and order between the three files
    times, lons, lats = extract_coords(datasets)

    # TODO how do we account for this spin up in the Python code?
    # i.e. what is being done here, is it adding ten years of mean values to the front of the data arrays?
#     % fake ten year spinup
#     PET(:,:,11:size(PET,3)+10)=PET;
#     PET(:,:,1:10) = repmat(nmean(PET,3), [1 1 10]);
#     ppt(:,:,11:size(ppt,3)+10)=ppt;
#     ppt(:,:,1:10) = repmat(nmean(ppt,3), [1 1 10]);

    logger.info('Computing the CAFEC precipitation')

    # get the CAFEC precipitation (P-hat), plus the PET, R, RO, and L values
    PHAT, petdat, rdat, rodat, tldat = get_cafec_precip(precip_dataset, pet_dataset, soil_dataset, times, lons, lats)

    # reshape the precipitation array to match the P-hat array (i.e. "full years shape": [# years, 12, lon, lat])
    # TODO now putting into shape (total_months, 1, 1)
    precip = np.reshape(precip_dataset.variables[precip_var_name][:], PHAT.shape)

    # get the moisture departure, precipitation - P-hat
    DD = precip - PHAT

    logger.info('Computing K\'')

    # compute the first approximation of the climatic characteristic, K'
#     K_prime, DD_mean = compute_K_prime(DD, petdat, rdat, rodat, precip, tldat)
    K_prime = compute_K_prime(DD, petdat, rdat, rodat, precip, tldat)

#     # compute the climatic characteristic, K
#     K = compute_K(K_prime, DD_mean)

    # Z-index, with shape (months, lons, lats)
    Z_1 = DD / (25.4 * K_prime)  # assume K' is in inches, multiply by 25.4 to get units into mm

    # limit Z scores to +/- 16
    limit = np.nonzero(Z_1 > 16)
    if limit[0].size > 0:
        Z_1[limit] = 16.0
    limit = np.nonzero(Z_1 < -16)
    if limit[0].size > 0:
        Z_1[limit] = -16.0

    # reshape the Z-Index array into (months, lons, lats)
    timelonlat_shape = (len(times), len(lons), len(lats),)
    Z_1 = np.reshape(Z_1, timelonlat_shape)

    # write the values to NetCDF file
    variable_name = 'zindex'
    attributes = {}
    attributes['standard_name'] = variable_name
    attributes['long_name'] = 'Palmer Z-index, computed as part of Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, Z_1, variable_name, attributes)

    # reshape the Z-Index array into dimensions (lons, lats, months)
    Z_1 = np.rollaxis(Z_1, 0, 3)

    # create a PDSI array with dimensions equal to Z-index array (lons, lats, months)
    PDSI = np.full(Z_1.shape, np.NAN)

    # go over every lon/lat cell as a full time series (i.e. first full dimension
    # of the PDSI array and corresponding data arrays) and calculate the PDSI values
    for lon in range(PDSI.shape[0]):
        for lat in range(PDSI.shape[1]):

            logger.info('Computing PDSI for lon/lat: {lon}/{lat}'.format(lon=lon, lat=lat))

            PDSI[lon, lat, :] = compute_pdsi(Z_1[lon][lat])

    #TODO get these values from command line arguments
    calibration_start_year = input_start_year = 1895
    calibration_end_year = 2010

    # compute the slope and intercept values associated with the duration factors
    scalef = compute_duration_factors(Z_1, calibration_start_year, calibration_end_year, input_start_year, 12)
#    scalef = compute_duration_factors(Z_1, calibration_start_year, calibration_end_year, input_start_year, 12, 1)

    # write the values to NetCDF file
    variable_name = 'wet_slope'
    attributes = {}
    attributes['standard_name'] = variable_name
    attributes['long_name'] = 'wet slope duration factor, computed as part of Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, scalef[:, :, 0, 0], variable_name, attributes)
    variable_name = 'wet_intercept'
    attributes = {}
    attributes['standard_name'] = variable_name
    attributes['long_name'] = 'wet intercept duration factor, computed as part of Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, scalef[:, :, 0, 1], variable_name, attributes)
    variable_name = 'dry_slope'
    attributes = {}
    attributes['standard_name'] = variable_name
    attributes['long_name'] = 'dry slope duration factor, computed as part of Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, scalef[:, :, 1, 0], variable_name, attributes)
    variable_name = 'dry_intercept'
    attributes = {}
    attributes['standard_name'] = variable_name
    attributes['long_name'] = 'dry intercept duration factor, computed as part of Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, scalef[:, :, 1, 1], variable_name, attributes)
    
    # allocate an array to hold scPDSI values, with the same shape as the Z-index values array (lons, lats, months), initialized with NaNs
    SCPDSI = np.full(Z_1.shape, np.NAN)

    # for each lon/lat cell we compute the scPDSI, passing in the Z-index values for the cell over the entire
    # time series, as well as the corresponding wet and dry slope and intercepts (duration factors)
    for lon in range(Z_1.shape[0]):
        for lat in range(Z_1.shape[1]):

            logger.info('Computing PDSI for lon/lat: {lon}/{lat}'.format(lon=lon, lat=lat))

            SCPDSI[lon, lat, :] = compute_scpdsi(Z_1[lon, lat, :],
                                                 scalef[lon, lat, 0, 0],
                                                 scalef[lon, lat, 0, 1],
                                                 scalef[lon, lat, 1, 0],
                                                 scalef[lon, lat, 1, 1])

    # reshape the data set into (times, lons, lats) which is the dimension order of the output NetCDF data set variable
    pdsi_data = np.transpose(SCPDSI, (2, 0, 1))

    # write the values to NetCDF file
    variable_name = 'pdsi'
    attributes['standard_name'] = 'SCPDSI'
    attributes['long_name'] = 'Self-calibrated PDSI'
    write_dataset(output_file_base + '_' + variable_name + '.nc', precip_dataset, pdsi_data, variable_name, attributes)

    # close the input NetCDF dataset objects
    close_datasets(datasets)

# -----------------------------------  original Matlab code below ----------------------------------
# function SMB_PDSIupdate(oo);
# d=datevec(date);
# year=d(1);
# month=d(2);
# day=d(3);
# if day<10 month=month-1;end
# if month<1 month=12;year=year-1;end
#
# year1=year;
# month1=month;
#
# [lon,lat]=latlonPRISM;
# [lon,lat]=meshgrid(lon,lat);
# if oo<7
#     lon=lon(1+(oo-1)*100:oo*100,:);
#     lat=lat(1+(oo-1)*100:oo*100,:);
# else
#     lon=lon(601:621,:);
#     lat=lat(601:621,:);
# end
#
#  here we load the precipitation data (multiplied by 25.4 to convert units into mm)
# 
# load (['/data/WWDT/DATAWRITE/precip_1_',num2str(oo)]);
# ppt=DATA3*25.4;
# clear DATA3
#
#
#  here we load the soil water capacity data
# 
# load (['/data/WWDT/DATAWRITE/PET_1_',num2str(oo)]);
# load soil250
# if oo<7
#     soil=soil(1+(oo-1)*100:oo*100,:);
# else
#     soil=soil(601:621,:);
# end
#
#
# % convert all into 3-d with space in 1st dimension
# PET=single(PET);
# ss=size(PET);
# ppt=shiftdim(reshape(shiftdim(ppt,2),ss(3),ss(4),ss(1)*ss(2)),2);
# PET=shiftdim(reshape(shiftdim(PET,2),ss(3),ss(4),ss(1)*ss(2)),2);
# soil=soil(:);
# ff1=find(~isnan(soil)==1 & ~isnan(PET(:,5,5))==1);
# ppt=ppt(ff1,:,:);
# soil=soil(ff1);
# PET=PET(ff1,:,:);
#
#  here we load the soil water capacity data into two arrays, one for the bottom layer and another for the top one inch
# 
# WCBOT=soil*10;
# WCTOP=25.4*ones(size(soil));
# clear soil;
#
#
#
# *** ppt and PET should have dimensions: (lon*lat, month, year)
#
#
#
# %INITIALIZE VARS
# WCTOT=WCBOT+WCTOP;
# SS=WCTOP;
# SU=WCBOT;
# SP=SS+SU;
#

# create a PET and ppt arrays with an additional 10 years of mean values

# % fake ten year spinup
# PET(:,:,11:size(PET,3)+10)=PET;
# PET(:,:,1:10)=repmat(nmean(PET,3),[1 1 10]);
# ppt(:,:,11:size(ppt,3)+10)=ppt;
# ppt(:,:,1:10)=repmat(nmean(ppt,3),[1 1 10]);
#
# clear tmean
# SP=SS+SU;
# %BEGIN REAL STUFF
# % alter stuff and only run data for last 2 years max
#
#
# for yr=1:size(PET,3)%1:200 for GCM
#     for mo=1:12
#         PR=WCTOT-SP;
#         [PL,ET,TL,RO,R,SSS,SSU]=PDSI_wb_findnoloop(PET(:,mo,yr),ppt(:,mo,yr),WCBOT(:),WCTOP(:),WCTOT(:),SS(:),SU(:),SP(:));
#         SS=SSS;
#         SU=SSU;
#         SP=SS+SU;
#         spdat(:,mo,yr)=single(SP);
#         pldat(:,mo,yr)=single(PL);
#         prdat(:,mo,yr)=single(PR);
#         %        rdat(:,mo,yr)=single(R);
#         %        tldat(:,mo,yr)=single(TL);
#         %        etdat(:,mo,yr)=single(ET);
#         %        rodat(:,mo,yr)=single(RO);
#         sssdat(:,mo,yr)=single(SS);
#         ssudat(:,mo,yr)=single(SU);
#
#         %fprintf ('\n Month %d Year %d\n',mo,yr);
#     end;
# end;
#
#
# %        c1=find(rdat>=prdat);rdat(c1)=prdat(c1);
# %        c1=find(tldat>=pldat);tldat(c1)=pldat(c1);
#
# whos
#
# load(['/data/WWDT/DATAREAD/PDSIv_',num2str(oo)],'gam','bet','alp','del','AK');
#
# % we already calculated coefficients, only data required is PET, prdat, spdat, pldat and ppt
#
# PHAT=PET.*repmat(alp,[1 1 size(PET,3)])+prdat.*repmat(bet,[1 1 size(PET,3)])+spdat.*repmat(gam,[1 1 size(PET,3)])-pldat.*repmat(del,[1 1 size(PET,3)]);
# clear spdat prdat pldat
# DD=ppt-PHAT;
# clear PHAT ppt
#
#
#  *** what is the shape of AK?
#  *** what does repmat(AK,[1 1 size(DD,3)] look like?
#
#
# Z_1=DD/25.4.*repmat(AK,[1 1 size(DD,3)]);
# clear DD
# % limit Z scores to +/- 10
# limit=find(Z_1>16);Z_1(limit)=16;
# limit=find(Z_1<-16);Z_1(limit)=-16;
#
#
#  *** what is the shape of Z_1 at this point? same as ppt and PET, no? (lons*lats, month, year)
#
#
# PDSI=NaN*ones(size(Z_1,1),size(Z_1,2)*size(Z_1,3));
#
#
# *** PDSI has shape == (lons*lats, 12*years)
#
#
# *** here we loop over each lon/lat point and compute PDSI for that point
#
# for i=1:size(PDSI,1)
#     %if max(Z_1(i,:))>0
#     [PDSI(i,:),X1,X2,X3,Pe,montho]=PDSIr((Z_1(i,:)));
# end;%end
#
# *** here we make an array of pdsidata conaining all NaN values, dimensions: (months, years, lons*lats)
#
# pdsidata=NaN*single(ones(12,size(PDSI,2)/12,ss(1)*ss(2)));
#
# *** here we reshape the PDSI array of pdsidata conaining all NaN values, dimensions: (months, years, lons*lats)
#
# S2=size(PDSI,1);
# PDSI=reshape(PDSI,S2,12,size(PDSI,2)/12);
#
# *** PDSI has shape == (lons*lats, 12, years), shift dimensions to the left by one
#
# pdsidata(:,:,ff1)=shiftdim(PDSI,1);
#
# *** pdsidata now has shape: (12, years, lons*lats), containing values from the previous PDSI array at the ff1 indices
#
#
# *** get rid of original PDSI array
#
# clear PDSI
#
# zdata=pdsidata;
#
#
# *** Z-index array gets shifted from (lons*lats, month, year) to (month, year, lons*lats),
#     then new zdata array created containing values from the shifted Z-index array at the ff1 indices
#
# zdata(:,:,ff1)=shiftdim(Z_1,1);
#
# *** get rid of original Z-index array
#
# clear Z_1
#
# *** trim off the first ten years from the Z-index and PDSI data arrays
#
# zdata=zdata(:,11:size(zdata,2),:);
# pdsidata=pdsidata(:,11:size(pdsidata,2),:);
#
# *** zdata and pdsidata get reshaped to (month, year, lon, lat) then shifted to (lon, lat, month, year)
#
# zdata=shiftdim(reshape(zdata,ss(3),ss(4),ss(1),ss(2)),2);
# pdsidata=shiftdim(reshape(pdsidata,ss(3),ss(4),ss(1),ss(2)),2);
#
# *** add in remaining months for the year to the current month
#
# if month1<12
#     pdsidata(:,:,month1+1:12,size(pdsidata,4))=NaN;
#     zdata(:,:,month1+1:12,size(pdsidata,4))=NaN;
# end
# save(['/data/WWDT/DATAWRITE/PDSIv_',num2str(oo)],'-v7.3','AK','alp','bet','del','gam','pdsidata','zdata');
#
# clear pdsidata AK alp bet del gam
#
#
# *** load some scale information from file into arrays scalef and A
#
# load(['/data/WWDT/DATAREAD/SCPDSIv_',num2str(oo)],'scalef','A');
# s4=size(zdata);
#
# *** save the original zdata array as zdata2
#
# zdata2=zdata;
#
# *** turn all positive values to zeros in the zdata array
#
# f=find(zdata>0);
# zdata(f)=0;
#
# *** turn all scale values above 0.98 to 0.98 in the scalef array
#
# tt=find(scalef>.98);
# scalef(tt)=.98;
#
# *** we are broadcasting part of the A array against the zdata array, not sure what is really happening here?
#
# zdata=zdata.*repmat(-4./A(:,:,1),[1 1 size(zdata,3) size(zdata,4)]);
#
# *** turn all negative values to zeros in the zdata2 array
#
# f=find(zdata2<0);
# zdata2(f)=0;
#
# *** we are broadcasting part of the A array against the zdata2 array, not sure what is really happening here?
#
# zdata2=zdata2.*repmat(4./A(:,:,2),[1 1 size(zdata,3) size(zdata,4)]);
#
# *** add the two Z-index matrices together
#
# zdata=zdata+zdata2;
#
# *** get rid of the zdata2 array
#
# clear zdata2
#
# *** create SCPDSI array with dimensions: (year, month, lon*lat), assuming that A has dimensions: (year, month, lon, lat)
#
# SCPDSI=NaN*ones(size(A,1),size(A,2),size(zdata,3)*size(zdata,4));
#
# *** reshape the zdata array to dimensions: (year, month, lon*lat)
#
# zdata=reshape(zdata,size(zdata,1),size(zdata,2),size(zdata,4)*size(zdata,3));
# for i=1:size(A,1);
#     for j=1:size(A,2)
#         if max(zdata(i,j,:))>0
#             [SCPDSI(i,j,:)]=SCPDSIr(squeeze(zdata(i,j,:)),squeeze(scalef(i,j,:,:)));
#         end;
#     end;
# end
# clear zdata
#
# SC2=SCPDSI;
# f=find(SC2>0);
# SC2(f)=0;
# SC2=SC2.*repmat(-4./A(:,:,3),[1 1 size(SCPDSI,3)]);
# f=find(SCPDSI<0);
# SCPDSI(f)=0;
# SCPDSI=SCPDSI.*repmat(4./A(:,:,4),[1 1 size(SCPDSI,3)]);
# SCPDSI=reshape(SCPDSI+SC2,s4);
# clear SC2;
# if month1<12
#     SCPDSI(:,:,month1+1:12,size(SCPDSI,4))=NaN;
# end
# save(['/data/WWDT/DATAWRITE/SCPDSIv_',num2str(oo)],'-v7.3','SCPDSI','scalef','A');
# clear SC* scalef A zdata
#
#
#
# function [PDSI,X1,X2,X3,Pe,montho]=PDSIr(Z);
# nmonths=0;
# X1=zeros(size(Z));X2=X1;X3=X1;Uw=X1;Ud=X1;Pe=X1;PDSI=X1;Ze=X1;
# % start off not in dry or wet spell
#
# XX=Z(1)/3;
# if XX>0 X1(1)=XX;else X2(1)=XX;end
# if abs(XX)>=1 X3(1)=XX;nmonths=1;end
# PDSI(1)=XX;
#
#
# % loop over time
#
# for i=2:length(Z)
#     XX=Z[i]/3;
#     Ud[i]=XX*3-.15;
#     Uw[i]=XX*3+.15;
#     if XX>0
#         if X2(i-1)<0 X2[i]=X2(i-1)*.897+XX;else X2[i]=XX;end;
#         if X2(i-1)<-1 & X3(i-1)<-1 X2[i]=0;end
#         if X1(i-1)>0 X1[i]=X1(i-1)*.897+XX;else X1[i]=XX;end; if X2(i-1)<-1 & X3(i-1)<-1 X2[i]=0;end
#         if (X3(i-1))~=0
#             if X3(i-1)<0
#                 Ze[i]=-2.691*X3(i-1)-1.5;
#                 Pe[i]=gettingout(Uw,Ze,nmonths+1,i,1);
#                 if Pe[i]==100 nmonths=0; X3[i]=0; if X1[i]>1 X3[i]=X1[i];nmonths=1; end;end
#                 if Pe[i]==0 X1[i]=0;X2[i]=0;end
#             else
#                 Ze[i]=-2.691*X3(i-1)+1.5;
#                 Pe[i]=gettingout(Ud,Ze,nmonths+1,i,0);
#                 if Pe[i]==100 nmonths=0; X3[i]=0; if X1[i]>1 X3[i]=X1[i];nmonths=1; end;end
#                 if Pe[i]==0 X1[i]=0;X2[i]=0;end
#             end
#         else
#             if X1[i]>0.5 X3[i]=X1[i];nmonths=1;end
#         end
#     else
#         if X2(i-1)<0 X2[i]=X2(i-1)*.897+XX;else X2[i]=XX;end
#         if X1(i-1)>0 X1[i]=X1(i-1)*.897+XX;else X1[i]=XX;end
#         if (X3(i-1))~=0
#             if X3(i-1)>0
#                 Ze[i]=-2.691*X3(i-1)+1.5;
#                 Pe[i]=gettingout(Ud,Ze,nmonths+1,i,0);
#                 if Pe[i]==100 nmonths=0; X3[i]=0;if X2[i]<-1 X3[i]=X2[i]; nmonths=1;end;end
#                 if Pe[i]==0  X1[i]=0;X2[i]=0;end
#             else
#                 Ze[i]=-2.691*X3(i-1)-1.5;
#                 Pe[i]=gettingout(Uw,Ze,nmonths+1,i,1);
#                 if Pe[i]==100 nmonths=0; X3[i]=0; if X2[i]>1 X3[i]=X2[i];nmonths=1; end;end
#                 if Pe[i]==0 X1[i]=0;X2[i]=0;end
#             end
#         else
#             if X2[i]<-0.5 X3[i]=X2[i];nmonths=1;end
#         end
#     end
#     if Pe[i]<100
#         if (X3(i-1))~=0 X3[i]=X3(i-1)*.897+XX;nmonths=nmonths+1;end
#     end
#     % decide what PDSI is
#     montho[i]=nmonths;
#     % option 1 no drought going on or being established, nmonths=0
#     %PDSI[i]=PDSI[i]*.897+XX;
#     if X1[i]<0 X1[i]=0;end
#     if X2[i]>0 X2[i]=0;end
#
#     if nmonths==0
#         if X3[i]==0
#             if X1[i]>-X2[i] PDSI[i]=X1[i];else PDSI[i]=X2[i];end
#             PDSI[i]=XX;
#         end
#     else
#         % if in wet/dry spell and it has not ended PDSI=X3
#         %  if abs(X3[i])<1
#         %      if X1[i]>-X2[i] PDSI[i]=X1[i];else PDSI[i]=X2[i];end
#         %  else
#         % if nmonths>1
#         PDSI[i]=X3[i];
#         % end
#
#         if PDSI[i] >=1 & nmonths>1 X1[i]=0; end
#         if PDSI[i] <=-1 & nmonths>1 X2[i]=0; end
#     end
# end% if beginning wet/dry spell PDSI=X1 or X2
#
#
# function [Pe]=gettingout(U,Ze,nmonths,nmonthc,dow);
# US1=0;US2=0;
# for i=1:nmonths
#     US1=US1+U(nmonthc-nmonths+i);
#     if dow==1 & US1<0 US1=0;end
#     if dow==0 & US1>0 US1=0;end
# end
# if nmonths>1
#     for i=1:nmonths-1
#         US2=US2+U(nmonthc-nmonths+i);
#         if dow==1 & US2<0 US2=0;end
#         if dow==0 & US2>0 US2=0;end
#     end
# end
# Pe=100*(US1)/(Ze(nmonthc)+US2);
# if Pe>100 Pe=100;end
# if Pe<0 Pe=0;end


if __name__ == '__main__':

    # # get command line arguments
    # precip_file = sys.argv[1]
    # pet_file = sys.argv[2]
    # soil_file = sys.argv[3]
    # output_file = sys.argv[4]

#    # paths relevant to laptop PC
#    data_dir = 'C:/Users/James/Dropbox/scpdsi/data/'   # home PC
    data_dir = 'C:/home/wrcc/data/'   # work PC
    precip_var_name = 'ppt'

    # # paths relevant to EC2 instance
    # data_dir = '/home/ubuntu/Dropbox/scpdsi/data/'

    # paths relevant to climgrid-dev
#     data_dir = '/home/james.adams/scpdsi_wrcc/data/'

    precip_file = data_dir + 'point_ppt.nc'
    pet_file = data_dir + 'point_PET.nc'
    soil_file = data_dir + 'point_soil.nc'
    output_file_base = data_dir + 'ncei_scpdsi_point_1.nc'

    scpdsi(precip_file, precip_var_name, pet_file, soil_file, output_file_base)

    print 'Done'