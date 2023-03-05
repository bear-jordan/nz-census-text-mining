import pandas as pd

from utils.config import *
from utils.eda import run_eda
from utils.tpp import run_tpp
from utils.vct import run_vct
from utils.model import run_model
from utils.model import run_tune
from utils.imbalance import run_imbalance

def loadData():
    rawData = pd.read_csv(DUMMY_DATA_FILEPATH)
    rawNewData = pd.read_csv(DUMMY_DATA_NEW_FILEPATH)
    
    return (rawData, rawNewData)

def main():
    rawData, rawNewData = loadData()
    data = run_tpp(rawData.copy())
    newData = run_tpp(rawNewData.copy())
    X, Xnew = run_vct(data, newData)
    y = data[RESPONSE_COL_NAME]
    X, y = run_imbalance(X, y)
    bestModel = run_tune(X, y)
    results = run_model(bestModel, Xnew)
    
    
if __name__ == "__main__":
    main()