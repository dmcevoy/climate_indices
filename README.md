**Master**
[![Build Status](https://img.shields.io/travis/<account>/<project>/master.svg)](https://travis-ci.org/<account>/<project>)
[![Test Coverage](https://img.shields.io/codecov/c/gitlab/<account>/<project>/master.svg)](https://codecov.io/gitlab/<account>/<project>)
[![Code Climate](https://img.shields.io/codeclimate/gitlab/<account>/<project>.svg)](https://codeclimate.com/gitlab/<account>/<project>)
[![Dependencies](https://img.shields.io/gemnasium/<account>/<project>.svg)](https://gemnasium.com/<account>/<project>)

## Climate Indicators for everyone

This project includes code to compute climate indicators, and is an attempt to provide implementations which are useful 
to a wide range of users. 

We welcome you to explore, make suggestions, and contribute to our code. 

## Get involved
We’re happy for you to get involved! 
- Read our [contributing guidelines](https://k3.cicsnc.org/jadams/climate-indices/CONTRIBUTING.md). 
Then, [file an issue](https://k3.cicsnc.org/jadams/climate-indices/issues) or submit a pull request.
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

#### Python Unit Tests

    py.test

## Copyright and licensing
This project is in the public domain within the United States, and we waive worldwide copyright and related rights 
through [CC0 universal public domain dedication](https://creativecommons.org/publicdomain/zero/1.0/). Read more on our license page.
