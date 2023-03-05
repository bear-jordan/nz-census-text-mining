from imblearn.over_sampling import SMOTE

def run_imbalance(X, y):
    sm = SMOTE()
    X, y = sm.fit_resample(X, y)
    
    return (X, y)
    