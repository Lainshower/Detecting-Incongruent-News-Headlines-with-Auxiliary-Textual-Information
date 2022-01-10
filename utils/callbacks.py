'''
Customized Callbacks for Trainer
'''

import numpy as np
import matplotlib.pyplot as plt
import datetime

def EarlyStopping(taccList,vaccList, min_delta=0.01, patience=3):
    #No early stopping for 2*patience epochs 
    if len(vaccList)//patience < 2 :
        return False

    vacc_mean_recent = np.mean(vaccList[::-1][:patience]) #last # valid Data
    tacc_mean_recent = np.mean(taccList[::-1][:patience]) #last # train Data

    if tacc_mean_recent - vacc_mean_recent > min_delta :
        print("*EarlyStopping* Valid Accuracy didn't change much from last %d epochs than Train Accuracy"%(patience))
        print("*...EarlyStopping...*")
        return True
    else:
        return False
    
def ModelSave(vaccList, vacc):
    if vacc == max(vaccList):
        return True
    else:
        return False