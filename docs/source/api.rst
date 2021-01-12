API Reference
=============

This is the API documentation for ``StreamAD``.


Multivariate Anomaly Models
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: streamad.model
    :no-members:
    :no-inherited-members:

.. currentmodule:: streamad

.. autosummary::
    :nosignatures:
    :template: class.rst
    :toctree: generated

    model.xStreamDetector
    model.LSTMDetector
    model.RandomDetector


Univariate Anomaly Models
^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :nosignatures:
    :template: class.rst
    :toctree: generated


    model.KNNDetector
    model.SpotDetector



Core
^^^^

.. automodule:: streamad.base
    :no-members:
    :no-inherited-members:

.. currentmodule:: streamad

.. autosummary::
    :nosignatures:
    :template: class.rst
    :toctree: generated

    base.BaseDetector
    base.BaseMetrics


Util
^^^^

.. automodule:: streamad.util
    :no-members:
    :no-inherited-members:

.. currentmodule:: streamad

.. autosummary::
    :nosignatures:
    :template: class.rst
    :toctree: generated

    util.StreamGenerator
    util.MultivariateDS
    util.UnivariateDS
    util.StreamStatistic
    util.AUCMetric
