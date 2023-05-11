import os
import subprocess
import pandas as pd
import numpy as np
from typing import Literal
import json
import ast
import gdown
import zipfile

DS = {
    "AIOPS_KPI": ["preliminary_train", "finals_train", "finals_ground_truth"],
    "MICRO": [],
    "AWSCloudwatch": [],
    "GAIA": [
        "changepoint_data",
        "concept_drift_data",
        "linear_data",
        "low_signal-to-noise_ratio_data",
        "partially_stationary_data",
        "periodic_data",
        "staircase_data",
    ],
    "MSL": ["test"],
    "SMD": [],
    "CHM": [],
}


def check(ds_name, path="./streamad-benchmark-dataset"):
    assert ds_name in DS, f"Unavailable dataset {ds_name}, only support {list(DS.keys())}"

    if not os.path.exists(path):
        os.makedirs(path)


def download_ds(ds_name, path="./streamad-benchmark-dataset"):
    check(ds_name, path)

    if os.path.exists(path + "/" + ds_name):
        print("Dataset {} already exists".format(ds_name))
        return

    if str.lower(ds_name) == "aiops_kpi":
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/NetManAIOps/KPI-Anomaly-Detection.git",
                path + "/AIOPS_KPI",
            ]
        )
        subprocess.check_call(
            [
                "unzip",
                path + "/AIOPS_KPI/Finals_dataset/phase2_ground_truth.hdf.zip",
                "-d",
                path + "/AIOPS_KPI/Finals_dataset/",
            ]
        )
        subprocess.check_call(
            [
                "unzip",
                path + "/AIOPS_KPI/Finals_dataset/phase2_train.csv.zip",
                "-d",
                path + "/AIOPS_KPI/Finals_dataset/",
            ]
        )
    elif str.lower(ds_name) == "micro":
        os.makedirs(path + "/MICRO/", exist_ok=True)
        gdown.download(
            id="1nkEsD1g7THm_T58KwUQZ7o-b174fdx-n",
            output=path + "/MICRO/data.zip",
        )

        with zipfile.ZipFile(path + "/MICRO/data.zip") as zip_ref:
            zip_ref.extractall(path + "/MICRO/")

        for root, dirs, files in os.walk(path + "/MICRO/AIOps挑战赛数据"):
            for filename in files:
                if filename.endswith(".zip"):
                    fileSpec = path + "/MICRO/AIOps挑战赛数据/" + filename
                    with zipfile.ZipFile(fileSpec) as zip_ref:
                        zip_ref.extractall(path + "/MICRO/")

    elif str.lower(ds_name) == "awscloudwatch":
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=tree:0",
                "--sparse",
                "https://github.com/numenta/NAB.git",
                path + "/AWSCloudwatch",
            ]
        )
        subprocess.check_call(
            [
                "cd "
                + path
                + "/AWSCloudwatch/ && git sparse-checkout set data/realAWSCloudwatch && wget https://raw.githubusercontent.com/numenta/NAB/master/labels/combined_labels.json"
            ],
            shell=True,
        )
    elif str.lower(ds_name) == "gaia":
        subprocess.check_call(
            [
                "wget",
                "https://raw.githubusercontent.com/CloudWise-OpenSource/GAIA-DataSet/main/Companion_Data/metric_detection.zip",
                "-P",
                path + "/GAIA",
            ]
        )
        subprocess.check_call(
            [
                "unzip",
                path + "/GAIA/metric_detection.zip",
                "-d",
                path + "/GAIA/",
            ]
        )

    elif str.lower(ds_name) == "msl":
        subprocess.check_call(
            [
                "wget",
                "https://s3-us-west-2.amazonaws.com/telemanom/data.zip",
                "-P",
                path + "/MSL",
            ]
        )

        subprocess.check_call(
            [
                "unzip",
                path + "/MSL/data.zip",
                "-d",
                path + "/MSL/",
            ]
        )
        subprocess.check_call(
            [
                "rm",
                path + "/MSL/data.zip",
            ]
        )

        subprocess.check_call(
            [
                "wget",
                "https://raw.githubusercontent.com/khundman/telemanom/master/labeled_anomalies.csv",
                "-P",
                path + "/MSL",
            ]
        )
    elif str.lower(ds_name) == "chm":
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/Fengrui-Liu/Cloud-host-metrics-dataset",
                path + "/CHM",
            ]
        )
        subprocess.check_call(
            ["unzip", path + "/CHM/data.zip", "-d", path + "/CHM/"]
        )
        subprocess.check_call(["rm", "-rf", path + "/CHM/.git"])
    elif str.lower(ds_name) == "smd":
        subprocess.check_call(
            [
                "git",
                "clone",
                "--depth=1",
                "https://github.com/NetManAIOps/OmniAnomaly",
                path + "/SMD",
            ]
        )


def prepare_ds(
    ds_name: Literal["AIOPS_KPI"], path="./streamad-benchmark-dataset"
):
    # check(ds_name, path)

    download_ds(ds_name, path)


def read_ds(ds_name, ds_file, path="./streamad-benchmark-dataset"):
    check(ds_name, path)

    if str.lower(ds_name) == "aiops_kpi":
        if ds_file == "preliminary_train":
            df = pd.read_csv(
                path + "/" + ds_name + "/Preliminary_dataset/train.csv"
            )

        elif ds_file == "finals_train":
            df = pd.read_csv(
                path + "/" + ds_name + "/Finals_dataset/phase2_train.csv"
            )

        elif ds_file == "finals_ground_truth":
            df = pd.read_hdf(
                path + "/" + ds_name + "/Finals_dataset/phase2_ground_truth.hdf"
            )
        else:
            raise FileNotFoundError(
                "Unavailable dataset file, only support {}".format(DS[ds_name])
            )

        df_groups = df.groupby("KPI ID")
        keys = df_groups.groups.keys()
        dfs = {}
        for key in keys:
            df_key = df_groups.get_group(key)
            df_key = df_key[["timestamp", "value", "label"]]
            df_label = df_key["label"]
            dfs[key] = (df_key, df_label)

        return dfs

    elif str.lower(ds_name) == "micro":
        labels = pd.read_csv(path + "/MICRO/故障整理（预赛）.csv", index_col=["index"])
        labels = labels.dropna(subset=["kpi", "start_time"])
        dfs = {}
        for idx, fault in labels.iterrows():
            start_time = fault["start_time"]
            duration = fault["duration"]
            folder = pd.to_datetime(start_time).strftime("%Y_%m_%d")
            start_time = pd.to_datetime(start_time + "+0800", utc=True)
            end_time = start_time + pd.Timedelta(duration)

            df_lst = []
            for root, dirs, files in os.walk(
                path + "/MICRO/" + folder + "/平台指标/"
            ):
                for filename in files:
                    if filename.endswith(".csv"):
                        df = pd.read_csv(
                            path + "/MICRO/" + folder + "/平台指标/" + filename
                        )
                        df_lst.append(df)

            df = pd.concat(df_lst, axis=0)

            for kpi in fault["kpi"].split(";"):
                df_kpi = df[
                    (df["name"] == kpi) & (df["cmdb_id"] == fault["name"])
                ][["timestamp", "value"]]
                df_kpi["label"] = 0
                df_kpi.loc[
                    (df_kpi["timestamp"] > start_time.timestamp() * 1000)
                    & (df_kpi["timestamp"] < end_time.timestamp() * 1000),
                    "label",
                ] = 1
                dfs[kpi + "_" + fault["name"]] = (df_kpi, df_kpi["label"])

        return dfs

    elif str.lower(ds_name) == "awscloudwatch":
        labels = json.load(open(path + "/AWSCloudwatch/combined_labels.json"))
        dfs = {}
        for f in os.listdir(path + "/AWSCloudwatch/data/realAWSCloudwatch"):
            if f.endswith(".csv"):
                df = pd.read_csv(
                    path + "/AWSCloudwatch/data/realAWSCloudwatch/" + f
                )
                df = df[["timestamp", "value"]]
                key = "realAWSCloudwatch/" + f
                label = labels[key]
                df["label"] = 0
                df.loc[df["timestamp"].isin(label), "label"] = 1
                df_label = df["label"]

                dfs[f.split(".")[0]] = (df, df_label)
        return dfs

    elif str.lower(ds_name) == "gaia":
        if ds_file in DS[ds_name]:
            dfs = {}
            folder = path + "/GAIA/metric_detection/" + ds_file
            for root, dirs, files in os.walk(folder):
                for item in files:
                    df = pd.read_csv(root + "/" + item)
                    df_label = df["label"]
                    dfs[item.split(".csv")[0]] = (df, df_label)
            return dfs
        elif ds_file == 'all':
            dfs = {}
            for ds_file in DS[ds_name]:
                folder = path + "/GAIA/metric_detection/" + ds_file
                for root, dirs, files in os.walk(folder):
                    for item in files:
                        df = pd.read_csv(root + "/" + item)
                        df_label = df["label"]
                        dfs[item.split(".csv")[0]] = (df, df_label)
            return dfs
        else:
            raise FileNotFoundError

    elif str.lower(ds_name) == "msl":
        labels = pd.read_csv(path + "/MSL/labeled_anomalies.csv")
        if ds_file in DS[ds_name]:
            dfs = {}
            folder = path + "/MSL/data/" + ds_file
            for root, dirs, files in os.walk(folder):
                for item in files:
                    name = item.replace(".npy", "")
                    df = pd.DataFrame(np.load(root + "/" + item))
                    df.columns = df.columns.astype(str)
                    anomalies = labels[labels["chan_id"] == name][
                        "anomaly_sequences"
                    ]
                    df["label"] = 0
                    if len(anomalies) > 0:
                        anomalies = ast.literal_eval(anomalies.values[0])
                        for seg in anomalies:
                            seg_begin = seg[0]
                            seg_end = seg[1]
                            df.iloc[seg_begin:seg_end] = 1

                    dfs[name] = (df, df["label"])

            return dfs

        else:
            raise FileNotFoundError

    elif str.lower(ds_name) == "chm":
        dfs = {}
        for root, dirs, files in os.walk(path + "/CHM/data"):
            for item in files:
                df = pd.read_csv(root + "/" + item, index_col=["timestamp"])
                df = df.sort_index()
                dfs[item.split(".csv")[0]] = (df, df["label"])

        return dfs
    elif str.lower(ds_name) == "smd":
        dfs = {}
        for root, dirs, files in os.walk(
            path + "/SMD/ServerMachineDataset/test"
        ):
            for item in files:
                df = pd.read_csv(root + "/" + item, header=None)
                label = pd.read_csv(
                    path + "/SMD/ServerMachineDataset/test_label/" + item,
                    header=None,
                    names=["label"],
                )
                df.columns = df.columns.astype(str)
                df["label"] = label
                dfs[item.split(".txt")[0]] = (df, df["label"])
        return dfs


if __name__ == "__main__":
    ds_name = "SMD"
    df_file = ""
    prepare_ds(
        ds_name=ds_name,
        path="./benchmark/streamad-benchmark-dataset",
    )
    dfs = read_ds(
        ds_name=ds_name,
        ds_file=df_file,
        path="./benchmark/streamad-benchmark-dataset",
    )

    dfs
