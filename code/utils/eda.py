import pandas as pd
from config import *

def show_class_dist(trainData):
    dist = trainData.loc[:, RESPONSE_COL_NAME].value_counts()
    print("==== Value counts of training data ====")
    print(dist)
    ("====================================")
    
def show_wordcount(trainData):
    govtFilter = trainData[RESPONSE_COL_NAME]==RESPONSE_ANTIGOV_NAME
    privFilter = trainData[RESPONSE_COL_NAME]==RESPONSE_PRIVACY_NAME
    trainData["wordcount"] = trainData.loc[:, INPUT_NAME].apply(lambda x: len(str(x).split()))
    
    print("==== Mean word counts of anti-government responses in training data ====")
    print(trainData.loc[govtFilter, "wordcount"].mean())
    ("====================================")
    print()
    
    print("==== Mean word counts of privacy responses in training data ====")
    print(trainData.loc[privFilter, "wordcount"].mean())
    ("====================================")
    
def run_eda(trainData):
    show_class_dist(trainData)
    show_wordcount(trainData)
