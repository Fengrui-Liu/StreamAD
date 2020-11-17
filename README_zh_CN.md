# 基于FLink的流式时序异常检测

Online streaming time series anomaly detection, including anomaly detector and prediction.

基于FLink和Kafka的实时在线时序数据异常检测。

[[English version](./README.md)][[中文版本](./README_zh_CN.md)]

----

## 安装

Flink: 详细安装步骤请参照[Flink](https://ci.apache.org/projects/flink/flink-docs-release-1.11/try-flink/local_installation.html)

```zsh
$ tar -xzf flink-1.11.2-bin-scala_2.11.tgz
$ cd flink-1.11.2-bin-scala_2.11
# 启动 cluster
$ ./bin/start-cluster.sh
# Starting cluster.
# Starting standalonesession daemon on host.
# Starting taskexecutor daemon on host.

# (可选)在运行程序时，查看输出
$ tail -f log/flink-*.out
```

Kafka: 详细安装步骤请参照[Kafka](http://kafka.apache.org/downloads)

```zsh
$ tar -xzf kafka_2.13-2.6.0.tgz
$ cd ./kafka/bin
# 启动zookeeper服务
$ ./zookeeper-server-start.sh ../config/zookeeper.properties
# 另开新终端，启动kafka server
$ ./kafka-server-start.sh ../config/server.properties
# 另开新终端，启动kafka producer, 实时输入数据
$ ./bin/kafka-console-producer.sh --broker-list localhost:9092 --topic test
# (可选) 另开新终端，启动kafka consumer，实时打印输入的数据
$ ./bin/kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic test --from-beginning
```

streamingTS

```
git clone git@github.com:Fengrui-Liu/streamingAD.git
```


---

TODO:

- Anomaly detector
    - [] [IsolationForest](https://cs.nju.edu.cn/zhouzh/zhouzh.files/publication/icdm08b.pdf)
    - [] [Numenta HTM](https://github.com/numenta/nupic)
    - [] [CAD OSE](https://github.com/smirmik/CAD)
    - [] [earthgecko Skyline](https://github.com/earthgecko/skyline)
    - [] [KNN CAD](https://github.com/numenta/NAB/tree/master/nab/detectors/knncad)
    - [] [Relative Entropy](http://www.hpl.hp.com/techreports/2011/HPL-2011-8.pdf)
    - [] [Random Cut Forest](http://proceedings.mlr.press/v48/guha16.pdf)
    - [] [Twitter ADVec v1.0.0](https://github.com/twitter/AnomalyDetection)
    - [] [Windowed Gaussian](https://github.com/numenta/NAB/blob/master/nab/detectors/gaussian/windowedGaussian_detector.py)
    - [] [Etsy Skyline](https://github.com/etsy/skyline)

- Prediction
    - [] [prophet](https://facebook.github.io/prophet/)

---

### Overview tutorials
* [Tutorial at WWW 20202 (with videos)](https://lovvge.github.io/Forecasting-Tutorial-WWW-2020/)
* [Tutorial at SIGMOD 2019](https://lovvge.github.io/Forecasting-Tutorials/SIGMOD-2019/)
* [Tutorial at KDD 2019](https://lovvge.github.io/Forecasting-Tutorial-KDD-2019/)
* [Tutorial at VLDB 2018](https://lovvge.github.io/Forecasting-Tutorial-VLDB-2018/)