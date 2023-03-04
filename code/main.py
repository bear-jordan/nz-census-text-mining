import pandas as pd

from utils.config import *
from utils.eda import run_eda
from utils.tpp import run_tpp
from utils.vct import run_vct
from utils.model import run_model

def loadData():
    rawData = pd.read_csv(DUMMY_DATA_FILEPATH)
    rawNewData = pd.read_csv(DUMMY_DATA_NEW_FILEPATH)
    
    return (rawData, rawNewData)

def main():
    rawData, rawNewData = loadData()
    # run_eda(rawData)
    data = run_tpp(rawData.copy())
    newData = run_tpp(rawNewData.copy())
    X, Xnew = run_vct(data, newData)
    y = data[RESPONSE_COL_NAME]
    predictions = run_model(X, Xnew, y)
    results = rawNewData.copy()
    results.insert(loc=0, column="predictions", value=predictions)
    
    results.to_csv("../results/predictions.csv", index=False)

    
    
if __name__ == "__main__":
    main()