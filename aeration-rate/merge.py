# !/usr/bin/env python
# -*-coding:utf-8 -*-
"""
# Time       ：2022/10/12 18:00
# Author     ：heyingjie
# Description：
"""
import pandas as pd
import numpy as np
import gc

xgboost_res = pd.read_csv("./res/xgboost_res.csv")
lightgbm_res = pd.read_csv("./res/lightgbm_res.csv")

res = pd.DataFrame()
res['time'] = lightgbm_res['time']
lightgbm_po = 0.65
res[['Label1', 'Label2']] = lightgbm_po * lightgbm_res[['Label1', 'Label2']] + (1 - lightgbm_po) * xgboost_res[
    ['Label1', 'Label2']]
res.to_csv("./res/merge_res.csv", index=False)
