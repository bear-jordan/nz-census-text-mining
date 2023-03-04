import pandas as pd
from utils.config import *

def show_class_dist(data):
    dist = data.loc[:, RESPONSE_COL_NAME].value_counts()
    print("==== Value counts of training data ====")
    print(dist)
    print()
    
def show_wordcount(data):
    govtFilter = data[RESPONSE_COL_NAME]==RESPONSE_ANTIGOV_NAME
    privFilter = data[RESPONSE_COL_NAME]==RESPONSE_PRIVACY_NAME
    data["wordcount"] = data.loc[:, INPUT_NAME].apply(lambda x: len(str(x).split()))
    
    print("==== Mean word counts of anti-government responses in training data ====")
    print(round(data.loc[govtFilter, "wordcount"].mean(), 2))
    print()
    
    print("==== Mean word counts of privacy responses in training data ====")
    print(round(data.loc[privFilter, "wordcount"].mean(), 2))
    print()
    
def run_eda(data):
    show_class_dist(data)
    show_wordcount(data)
