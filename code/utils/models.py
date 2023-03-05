from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def define_models():
    knn = KNeighborsClassifier()
    knnGrid = {
        "n_neighbors": list(range(5, 20)),
        "weights": ["uniform", "distance"]
    }
        
    rfc = RandomForestClassifier()
    rfcGrid = {
        "n_estimators": list(range(50, 100, 10)),
        "min_samples_split": list(range(10, 20, 5))
    }
    return [(knn, knnGrid), (rfc, rfcGrid)]