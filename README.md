

<!--
 * @Author: liufr
 * @Github: https://github.com/Fengrui-Liu
 * @LastEditTime: 2020-12-27 16:16:59
 * @Copyright 2020 liufr
 * @Description:
-->

# Online stream anomaly detection(Outlier detection).


实时数据流异常检测。

[[English version](./README.md)][[中文版本](./README_zh_CN.md)]


## Why streamAD

Anomaly detection aims to finding unexpected patterns or outlier points among plenty of data. In practical application, such as Intrusion Detection System(IDS), anomalies must be identified and stopped as soon as possible, we tend to provide **streamAD** library for anomaly detection on-line; Another reason is that concept drift problem which refers the distribution and features of data change over time. The performance of some trained models may decrease and no longer suitable when concept drift occurs, thus, we need models that can be continuously updated to overcome this problem.


---


## TODO:

- Anomaly detector
    - [x] [KNN CAD](https://github.com/numenta/NAB/tree/master/nab/detectors/knncad)
    - [x] [xStream](https://cmuxstream.github.io/)
    - [x] [SPOT](https://dl.acm.org/doi/10.1145/3097983.3098144)
    - [x] [LSTMAutoencoder]
    - [] [IsolationForest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
    - [] [Numenta HTM](https://github.com/numenta/nupic)
    - [] [CAD OSE](https://github.com/smirmik/CAD)
    - [] [earthgecko Skyline](https://github.com/earthgecko/skyline)

    - [] [Relative Entropy](http://www.hpl.hp.com/techreports/2011/HPL-2011-8.pdf)
    - [] [Random Cut Forest](http://proceedings.mlr.press/v48/guha16.pdf)
    - [] [Twitter ADVec v1.0.0](https://github.com/twitter/AnomalyDetection)
    - [] [Windowed Gaussian](https://github.com/numenta/NAB/blob/master/nab/detectors/gaussian/windowedGaussian_detector.py)
    - [] [Etsy Skyline](https://github.com/etsy/skyline)



---

### Overview tutorials
* [Tutorial at WWW 20202 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)