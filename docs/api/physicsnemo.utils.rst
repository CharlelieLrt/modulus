PhysicsNeMo Utils
==================

.. automodule:: physicsnemo.utils
.. currentmodule:: physicsnemo.utils

The PhysicsNeMo Utils module provides a comprehensive set of utilities that support various aspects of scientific computing,
machine learning, and physics simulations. These utilities range from optimization helpers and distributed computing tools
to specialized functions for weather and climate modeling, and geometry processing. The module is designed to simplify common
tasks while maintaining high performance and scalability.

.. autosummary::
   :toctree: generated

Optimization Utils
------------------

The optimization utilities provide tools for capturing and managing training states, gradients, and optimization processes.
These are particularly useful when implementing custom training loops or specialized optimization strategies.

.. automodule:: physicsnemo.utils.capture
    :members:
    :show-inheritance:


GraphCast Utils
---------------

A collection of utilities specifically designed for working with the GraphCast model, including data processing,
graph construction, and specialized loss functions. These utilities are essential for implementing and
training GraphCast-based weather prediction models.

.. automodule:: physicsnemo.utils.graphcast.data_utils
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.graph
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.graph_utils
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.graphcast.loss
    :members:
    :show-inheritance:

Filesystem Utils
----------------

Utilities for handling file operations, caching, and data management across different storage systems.
These utilities abstract away the complexity of dealing with different filesystem types and provide
consistent interfaces for data access.

.. automodule:: physicsnemo.utils.filesystem
    :members:
    :show-inheritance:


Weather / Climate Utils
-----------------------

Specialized utilities for weather and climate modeling, including calculations for solar radiation
and atmospheric parameters. These utilities are used extensively in weather prediction models.

.. automodule:: physicsnemo.utils.insolation
    :members:
    :show-inheritance:

.. automodule:: physicsnemo.utils.zenith_angle
    :show-inheritance:

.. _patching_utils:


Domino Utils
------------

Utilities for working with the Domino model, including data processing and grid construction.
These utilities are essential for implementing and training Domino-based models.

.. automodule:: physicsnemo.utils.domino.utils
    :members:
    :show-inheritance:

Profiling Utils
---------------

Utilities for profiling the performance of a model.

.. automodule:: physicsnemo.utils.profiling
    :members:
    :show-inheritance:
