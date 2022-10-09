import os
import subprocess
import pandas as pd
from typing import Literal
import json

DS = {
    "AIOPS_KPI": ["preliminary_train", "finals_train", "finals_ground_truth"],
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
}


def check(ds_name, path="./streamad-benchmark-dataset"):

    assert ds_name in DS, "Unavailable dataset, only support {}".format(DS)

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


def prepare_ds(
    ds_name: Literal["AIOPS_KPI"], path="./streamad-benchmark-dataset"
):

    check(ds_name, path)

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
        else:
            raise FileNotFoundError
