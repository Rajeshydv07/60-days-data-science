from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, accuracy_score
import pickle

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        report = classification_report(y_test, predictions)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy, report
        
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)


class CustomerSegmenter:
    def __init__(self, n_clusters=3):
        self.model = KMeans(n_clusters=n_clusters, random_state=42)
        
    def fit_predict(self, X):
        clusters = self.model.fit_predict(X)
        return clusters
        
    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
