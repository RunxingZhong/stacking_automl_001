#!/usr/bin/env python
#-*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import runningz_automl

def stacking_automl(train, test, col_id, col_label, cols_fea, cols_cat, meta_train_list, meta_test_list):
    try:
        sub = runningz_automl.automl_adapter(train, test, col_id, col_label, cols_fea, cols_cat, 
            meta_train_list, meta_test_list, key = 'f1c1592588411002af340cbaedd6fc33')
        print('[+] done')
    except:
        print('[+] error')
        sub = test[[col_id]]
        sub[col_label] = 0
    return sub

def show():
    print('show test info...')
    