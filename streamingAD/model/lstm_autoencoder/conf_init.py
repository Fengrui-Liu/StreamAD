#!/usr/bin/env python
# coding=utf-8
"""
Author: liufr
Github: https://github.com/Fengrui-Liu
LastEditTime: 2020-12-11 09:12:38
Copyright 2020 liufr
Description:
"""

import pandas as pd


class InitConfig:
    def __init__(
        self, dataPath, modelPath, train_data_source="file", decode_without_input=False
    ) -> None:
        self.batch_num = 8
        self.hidden_num = 5
        self.step_num = 10
        self.train_set_size = self.step_num * 10000
        self.input_path = dataPath
        self.iteration = 300
        self.model_save_path = modelPath
        self.modelmeta_path = (
            self.model_save_path
            + "_"
            + str(self.batch_num)
            + "_"
            + str(self.hidden_num)
            + "_"
            + str(self.step_num)
            + "_.ckpt.meta"
        )
        self.modelpara_path = (
            self.model_save_path
            + "_"
            + str(self.batch_num)
            + "_"
            + str(self.hidden_num)
            + "_"
            + str(self.step_num)
            + "_para.ckpt"
        )
        self.modelmetapara_path = (
            self.model_save_path
            + "_"
            + str(self.batch_num)
            + "_"
            + str(self.hidden_num)
            + "_"
            + str(self.step_num)
            + "_para.ckpt.meta"
        )
        self.decode_without_input = decode_without_input

        self.log_path = self.model_save_path + "log.txt"

        self.train_data_source = train_data_source


class InitData:
    def __init__(
        self,
        input_path,
        train_set_size,
        step_num,
        batch_num,
        log_path,
    ) -> None:
        self.path = input_path
        self.batch_num = batch_num
        self.step_num = step_num
        self.trainset_size = train_set_size
