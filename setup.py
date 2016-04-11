from setuptools import setup

setup(
    name='climate_indices',
    version='0.0.1',
    description=(
        'Library for computing climate indices.'
    ),
    long_description=open('README.rst').read(),
    author='James Adams',
    author_email='monocongoREMOVETHIS@gmail.com',
    license='Public Domain, Creative Commons',
    url='None',
    packages=['climate_indices'],
    package_data={'': ['*.rst', '*.txt']},
    test_suite='tests',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.7',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Development Status :: 3 - Alpha',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
)