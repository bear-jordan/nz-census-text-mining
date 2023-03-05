import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from utils.models import define_models

def tune_models(model, grid, X, y):
    skf = StratifiedKFold(shuffle=True)
    search = GridSearchCV(model, grid, cv=skf)
    search.fit(X, y)
    
    return search
    
    
def predict():
    ...
    
def report_tune(search):
    print("= Model Results =")
    print(search.best_estimator_)
    print(round(search.best_score_, 4))
    print()

def report_cv(cvResults):
    print("== CV Scores ==")
    print(round(cvResults.mean(), 4))
    print(cvResults)
    print()
    print()
    
def get_best_model(modelResults):
    bestIndex = 0
    bestScore = 0
    for i, model in enumerate(modelResults):
        score = model["score"]
        if score > bestScore:
            bestIndex = i
            bestScore = score
            
    return modelResults[i]["model"]
            
    
def run_tune(X, y):
    modelResults = list()
    models = define_models()
    for model, grid in models:
        search = tune_models(model, grid, X, y)
        bestModel = search.best_estimator_
        report_tune(search)    
        report_cv(cross_val_score(bestModel, X, y, cv=5))
        modelResults.append({"model": bestModel, "score": search.best_score_})
    
    bestModel = get_best_model(modelResults)
    bestModel.fit(X, y)
    
    return bestModel
    
    
def run_model(bestModel, Xnew):
    return bestModel.predict(Xnew)