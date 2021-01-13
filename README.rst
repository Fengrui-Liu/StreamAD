.. image:: docs/source/images/logo_htmlwithname.svg
    :align: center


**StreamAD** Online anomaly detection for streaming data.
===========================================================

`Documentation <https://streamad.readthedocs.io/en/latest/>`_


.. image:: https://readthedocs.org/projects/streamad/badge/?version=latest
    :target: https://streamad.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://img.shields.io/pypi/v/streamad
    :target: https://pypi.org/project/streamad/
    :alt: PyPI

.. image:: https://travis-ci.org/Fengrui-Liu/StreamAD.svg?branch=main
    :target: https://travis-ci.org/Fengrui-Liu/StreamAD


Why StreamAD
=============


Purpose & Advantages
^^^^^^^^^^^^^^^^^^^^^^^^^^^

StreamAD focuses on streaming settings, where data features evolve and distributions change over time. To prevent the failure of static models, StreamAD can correct its model as needed.

Incremental & Continual
^^^^^^^^^^^^^^^^^^^^^^^^^^^

StreamAD loads static datasets to a stream generator and feed a single observation at a time to any model in StreamAD. Therefore it can be used to simulate real-time applications and process streaming data.


Models & Algorithms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

StreamAD collects open source implementations and reproduce state-of-the-art papers. Thus, it can also be used as an benchmark for academic.


Efficient & Scalability:
^^^^^^^^^^^^^^^^^^^^^^^^^^^

StreamAD concerns about the running time, resources usage and usability of different models. It is implemented by python and you can design your own algorithms and run with StreamAD.



Free & Open Source Software (FOSS)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`StreamAD` is distributed under `BSD License 3.0 <https://github.com/Fengrui-Liu/StreamAD/blob/main/LICENSE>`_ and favors FOSS principles.


Installation
============


The StreamAD framework can be installed via:


.. code-block:: bash

    pip install -U StreamAD


Alternatively, you can install the library directly using the source code in Github repository by:


.. code-block:: bash

    git clone https://github.com/Fengrui-Liu/StreamAD.git
    cd StreamAD
    pip install .


Versioning
==========

`Semantic versioning <http://semver.org/>`_ is used for this project.


Models
===================


* `KNN CAD <https://arxiv.org/abs/1608.04585>`_
* `xStream <https://cmuxstream.github.io/>`_
* `SPOT <https://dl.acm.org/doi/10.1145/3097983.3098144>`_
* LSTMAutoencoder

