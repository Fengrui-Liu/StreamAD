import os
import subprocess
import pandas as pd
from typing import Literal

DS = {"AIOPS_KPI": ["preliminary_train", "finals_train", "finals_ground_truth"]}


def check(ds_name, path="./streamad-benchmark-dataset"):

    assert ds_name in DS, "Unavailable dataset, only support {}".format(DS)

    if not os.path.exists(path):
        os.makedirs(path)


def download_ds(ds_name, path="./streamad-benchmark-dataset"):

    check(ds_name, path)

    if os.path.exists(path + "/" + ds_name):
        print("Dataset {} already exists".format(ds_name))
        return

    if ds_name == "AIOPS_KPI":
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


def prepare_ds(
    ds_name: Literal["AIOPS_KPI"], path="./streamad-benchmark-dataset"
):

    check(ds_name, path)

    download_ds(ds_name, path)


def read_ds(ds_name, ds_file, path="./streamad-benchmark-dataset"):

    check(ds_name, path)

    if ds_name == "AIOPS_KPI":

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
