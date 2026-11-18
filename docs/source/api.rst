.. currentmodule:: flox

API Reference
-------------

Functions
~~~~~~~~~

.. autosummary::
    :toctree: generated/

    groupby_reduce
    groupby_scan
    xarray.xarray_reduce

Exceedance events
~~~~~~~~~~~~~~~~~

Detect consecutive exceedance events along an ordered time axis and reduce
their observations with the existing groupby engine.

.. autosummary::
    :toctree: generated/

    label_events
    reduce_events
    find_events
    EventTable

Rechunking
~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    rechunk_for_blockwise
    rechunk_for_cohorts
    xarray.rechunk_for_blockwise
    xarray.rechunk_for_cohorts

Visualization
~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    visualize.draw_mesh
    visualize.visualize_groups_1d
    visualize.visualize_cohorts_2d

Aggregation Objects
~~~~~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/

    Aggregation
    Scan

    aggregations.sum_
    aggregations.nansum
