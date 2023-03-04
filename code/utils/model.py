import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

def fit_model(X, y):
    model = LogisticRegression(solver="lbfgs", multi_class="multinomial")
    skf = StratifiedKFold(shuffle=True)
    results = cross_val_score(model, X, y, cv=skf)
    model.fit(X, y)
    
    return (model, results)
    
def report(model, X, y, results):
    print("== Results across five runs ==")
    print([round(r, 4) for r in results])
    print(round(results.mean(), 5))
    print()
    print(confusion_matrix(y, model.predict(X)))
    print(classification_report(y, model.predict(X)))
    print()
    
def run_model(X, Xnew, y):
    model, results = fit_model(X, y)
    report(model, X, y, results)
    predictions = model.predict(Xnew)
    
    return predictions