import pandas as pd
from scipy.io import arff
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self):
        data = arff.loadarff('./resources/HTRU_2.arff')
        df = pd.DataFrame(data[0]).astype(float)

        X = df.loc[: , df.columns != 'class']
        y = df['class']
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        scaler = StandardScaler().fit(X_train)
        self.X_train_scaled = scaler.transform(X_train)
        self.X_test_scaled = scaler.transform(X_test)

    def knn(self, n_neighbors, weights):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)

    def svc(self, C, kernel):
        self.model = SVC(C=C, kernel=kernel)

    def random_forest(self, n_estimators, criterion, max_depth):
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)

    def fit(self):
        self.model.fit(self.X_train_scaled, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test_scaled)

    def confusion_matrix(self):
        return confusion_matrix(self.y_test, self.y_pred)
    
    def accuracy(self):
        return round(metrics.accuracy_score(self.y_test, self.y_pred), 4)
    
    def precision(self):
        return round(metrics.precision_score(self.y_test, self.y_pred), 4)
    
    def recall(self):
        return round(metrics.recall_score(self.y_test, self.y_pred), 4)