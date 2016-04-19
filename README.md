**Master**
[![Build Status](https://img.shields.io/travis/github.com/nidis/climate_indices/master.svg)](https://travis-ci.org/github.com/nidis/climate_indices)
[![Test Coverage](https://img.shields.io/codecov/c/gitlab/github.com/nidis/climate_indices/master.svg)](https://codecov.io/github.com/nidis/climate_indices)
[![Code Climate](https://img.shields.io/codeclimate/github.com/nidis/climate_indices.svg)](https://codeclimate.com/github.com/nidis/climate_indices)
[![Dependencies](https://img.shields.io/gemnasium/github.com/nidis/climate_indices.svg)](https://gemnasium.com/github.com/nidis/climate_indices)

## Climate Indicators for everyone

This project includes code to compute climate indicators, and is an attempt to provide implementations which are useful to a wide range of users. 

Below are the currently provided climate indicators, which provide a geographical and temporal picture of the severity of precipitation and temperature anomalies:

* [SPI](https://www.ncdc.noaa.gov/oa/climate/research/prelim/drought/spi.html): indicator of the sigma from the scaled values' fitted to a distribution 
(either gamma or Pearson type III) for each time step, input is total precipitation. This is a transformation of the probability of observing a given 
amount of precipitation in xx months.  A zero index value reflects the median of the distribution of precipitation, a -3 indicates a very extreme dry 
spell, and a +3 indicates a very extreme wet spell.  The more the index value departs from zero, the drier or wetter an event lasting xx months is when 
compared to the long-term climatology of the location.  The index allows for comparison of precipitation observations at different locations with markedly 
different climates; an index value at one location expresses the same relative departure from median conditions at one location as at another location.  
It is calculated for different time scales since it is possible to experience dry conditions over one time scale while simultaneously experiencing wet 
conditions over a different time scale. 
* [SPEI](https://www.researchgate.net/publication/252361460_The_Standardized_Precipitation-Evapotranspiration_Index_SPEI_a_multiscalar_drought_index): 
indicator of the sigma from the scaled values' fitted to a distribution (either gamma or Pearson type III) for each time step, input is total precipitation 
and temperature, as an intermediate step PET values are computed from temperature and can be provided as secondary output. The SPEI is based on precipitation 
and temperature data, and it has the advantage of combining multiscalar  character  with  the  capacity  to  include  the  effects  of  temperature  variability 
on  drought  assessment. The procedure to calculate the index is detailed and involves a climatic water balance, the accumulation of deficit/surplus at 
different time scales, and adjustment to a log-logistic probability distribution.Mathematically, the SPEI is similar to the standardized precipitation index 
(SPI), but it includes the role of temperature. 
* [PET](https://www.ncdc.noaa.gov/monitoring-references/dyk/potential-evapotranspiration): potential evapotranspiration, computed using [Thornthwaite's equation](https://en.wikipedia.org/wiki/Potential_evaporation). The water demand or maximum amount of water that would be evapotranspired if enough water were available 
(from precipitation and soil moisture), computed based on temperature and latitude.

## Get involved
We welcome you to use, make suggestions, and contribute to our code. 

- Read our [contributing guidelines](https://github.com/nidis/climate_indices/CONTRIBUTING.md). 
Then, [file an issue](https://github.com/nidis/climate_indices/issues) or submit a pull request.
- [Send us an email](mailto:james.adams@noaa.gov).

## Set up

### Dependencies
This library and the sample applications use the [NetCDF](https://www.unidata.ucar.edu/software/netcdf/), 
[Numpy](http://www.numpy.org/), and [SciPy](https://www.scipy.org/) modules.

It uses Python version 2.7. It's recommended that you create a [virtualenv](http://docs.python-guide.org/en/latest/dev/virtualenvs/) 
before installing Python dependencies, or use an installation of the [Anaconda](https://www.continuum.io/why-anaconda) 
Python distribution. 

Install Python dependencies:

    virtualenv:  $ pip install -r requirements.txt
    Anaconda:    $ conda create -n <environment_name> --file requirements.txt

### Run Tests

#### Python unit tests

The below should use the Python executable from the virtualenv or Anaconda installation configured above.

    $ cd src/test
    $ py.test
    
### Run indicator processor codes

**Example**: Compute PET from temperature dataset in NetCDF file 'nclimgrid\_tavg.nc' with a temperature variable named 'tavg', output to NetCDF file named 'nclimgrid\_pet.nc':

    $ python src/scripts/pet.py nclimgrid_tavg.nc tavg nclimgrid_pet.nc

**Example**: Compute SPI with a gamma distribution from precipitation dataset in NetCDF file 'nclimgrid\_prcp.nc' with a precipitation variable named 'prcp' at 3-month scale, output to NetCDF file named 'nclimgrid\_spi\_gamma\_03.nc':

    $ python src/scripts/spi_gamma.py nclimgrid_prcp.nc prcp nclimgrid_ 3

## Copyright and licensing
This project is in the public domain within the United States, and we waive worldwide copyright and related rights 
through [CC0 universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). Read more on our license page.
