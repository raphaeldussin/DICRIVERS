.. DICRIVERS documentation master file, created by
   sphinx-quickstart on Wed Jun  5 14:42:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

DICRIVERS: Proudly creating BGC river concentration gridded files since 2019
============================================================================

The purpose of this package is to create gridded files of river biogeochemical
properties. Such properties are measured locally or inferred at the mouth of
the river and provided in spreadsheet-style databases. To force ocean model, we
need gridded files of BGC concentrations, that can be multiplied by river and
coastal runoff to create the riverine flux into the ocean. DICRIVERS will find
the closest gridpoint to the true river mouth geographical location, then
create a "plume" (i.e. a binary mask) on which to apply the concentration of
each given river. Principal features:

#. works on global or regional domains, not model specific
#. adjust the spreading of the plume for each river
#. operates on pandas dataframe and returns xarray.Dataset


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   dicrivers_example_global_mom6
   dicrivers_example_regional_ROMS


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
