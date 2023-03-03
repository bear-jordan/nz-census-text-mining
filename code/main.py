import pandas as pd

from utils.config import *
from utils.eda import run_eda
from utils.tpp import run_tpp

def loadData():
    trainData = pd.read_csv(TRAINING_DATA_FILEPATH)
    testData = pd.read_csv(TESTING_DATA_FILEPATH)
    
    return (trainData, testData)

def main():
    rawTrainData, rawTestData = loadData()
    run_eda(rawTrainData)
    trainData = run_tpp(rawTrainData)
    testData = run_tpp(rawTestData)
    
    
if __name__ == "__main__":
    main()