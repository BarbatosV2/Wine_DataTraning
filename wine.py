import sys
import pandas as pd
from sklearn.neighbors import KDTree, KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

def preprocess_data(train_file, test_file):
    train_data = pd.read_csv(train_file, delimiter="\s+", header=0)
    train_data.columns = ['f_acid', 'v_acid', 'c_acid', 'res_sugar', 'chlorides',
                          'fs_dioxide', 'ts_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    test_data = pd.read_csv(test_file, delimiter="\s+", header=0)
    test_data.columns = ['f_acid', 'v_acid', 'c_acid', 'res_sugar', 'chlorides',
                         'fs_dioxide', 'ts_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    # Filter out quality values not in [5, 6, 7]
    train_data = train_data[train_data['quality'].isin([5, 6, 7])]
    
    return train_data, test_data

def train_and_predict(train_data, test_data, k):
    train_X = train_data.iloc[:, :-1].values
    train_y = train_data.iloc[:, -1].values
    test_X = test_data.values
    
    # Standardize features
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    
    # Train KNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(train_X_scaled, train_y)
    
    # Predict
    predictions = knn_classifier.predict(test_X_scaled)
    
    return predictions

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    k = int(sys.argv[3])

    train_data, test_data = preprocess_data(train_file, test_file)
    predictions = train_and_predict(train_data, test_data, k)
    
    print(*predictions, sep="\n")
