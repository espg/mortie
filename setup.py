# A minimal setup.py file to make a Python project installable.

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

#with open("environment.yml", "r") as fh:
#    env = yaml.safe_load(fh)
#requirements = [a.split('=', 1)[0].strip() for a in env['dependencies'] ]

setuptools.setup(
    name             = "mortie",
    version          = "0.2.0",
    author           = "Shane Grigsby",
    author_email     = "refuge@rocktalus.com",
    description      = "Morton numbering for healpix grids",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages         = setuptools.find_packages(),
    url              = 'https://github.com/espg/mortie',
    keywords         = ['healpix', 'geohashing', 'morton index'], 
    classifiers      = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires  = '>= 3.5',
    #install_requires = requirements,
    install_requires = [
        'healpy',
        'numpy',
        'numba',
        'cython',
        'llvmlite'
        ],
)
