

<!--
 * @Author: liufr
 * @Github: https://github.com/Fengrui-Liu
 * @LastEditTime: 2021-01-05 19:17:12
 * @Copyright 2021 liufr
 * @Description:
-->

# StreamAD

An anomaly detection package for data streams in Python.


* [Installation](installation.md)
* [Quick start](quick_start.md)
* [API document](api.md)



## Why StreamAD

* Purpose & Advantages: StreamAD focuses on streaming settings, where data features evolve and distributions change over time. To prevent the failure of static models, StreamAD can correct its model as needed.
* Incremental & Continual:StreamAD loads static datasets to a stream generator and feed a single observation at a time to any model in StreamAD. Therefore it can be used to simulate real-time applications and process streaming data.
* Models & Algorithms:StreamAD collects open source implementations and reproduce state-of-the-art papers. Thus, it can also be used as an benchmark for academic.
* Efficient & Scalability: StreamAD concerns about the running time, resources usage and usability of different models. It is implemented by python and you can design your own algorithms and run with StreamAD.


---


## Algorithm:

* [KNN CAD](https://github.com/numenta/NAB/tree/master/nab/detectors/knncad)
* [xStream](https://cmuxstream.github.io/)
* [SPOT](https://dl.acm.org/doi/10.1145/3097983.3098144)
* [LSTMAutoencoder]
