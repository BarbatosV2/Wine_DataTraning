import sys
import pandas as pd
import numpy as np

class Node:
    def __init__(self, point, val, d):
        self.point = point
        self.val = val
        self.d = d
        self.left = 0
        self.right = 0

def kdtree(points, depth=0, dimension=0):
    """algorithmn 1 using sudo code"""

    if len(points) == 0:        #if point is empty return null
        return None
    
    if dimension == 0:       #if there is no dimension
        dimension = len(points[0])
        
    axis = depth % dimension        #need dimension input
    sorted_points = sorted(points, key=lambda point: point[axis])
    median_idex = len(sorted_points) // 2           
    median_point = sorted_points[median_idex]
    node = Node(point=median_point, val=median_point[axis], d=axis)
    node.left = kdtree(sorted_points[:median_idex], depth+1, dimension)     #KdTree if d less than or equals val D+1
    node.right = kdtree(sorted_points[median_idex+1:], depth+1, dimension)  #if d greater than or val D+1
    return node

def distance(p1, p2):
    """to check the distance of the nodes for kNN"""
    return np.sqrt(sum([(p1[i] - p2[i])**2 for i in range(len(p1))]))

def knn(node, point, best = 0):
    """search form nearest neighbor"""
    if node == 0:
        return best
    
    if best == 0 or distance(node.point, point) < distance(best, point):
        best = node.point
    
    if node.left == 0 and node.right == 0:
        return best
    
    if node.left == 0:
        return knn(node.right, point, best)
    elif node.right == 0:
        return knn(node.left, point, best)
    
    if point[node.d] < node.val:
        best = knn(node.left, point, best)
        if (node.val - point[node.d])**2 < distance(best, point):
            best = knn(node.right, point, best)
    else:
        best = knn(node.right, point, best)
        if (point[node.d] - node.val)**2 < distance(best, point):
            best = knn(node.left, point, best)
    
    return best

def predict_quality(train_data, test_data, dimension, k):
    """using kNN algorithmn to predict wine quailty"""

    train_X = train_data.iloc[:, :-1].values
    train_y = train_data.iloc[:, -1].values
    test_X = test_data.values
    
    quality = []                    # puting the data quality of wine 
    for i in range(len(test_X)):
        distances = []
        for j in range(len(train_X)):
            dist = distance(test_X[i], train_X[j])
            distances.append((dist, train_y[j]))
        distances.sort()
        k_nn = distances[:k]            #using neighboring method
        k_nn_quality = [neighbor[1] for neighbor in k_nn]
        prediction = max(set(k_nn_quality), key=k_nn_quality.count)
        quality.append(prediction)
        
    return quality

if __name__ == '__main__':
    train_input = sys.argv[1]            #train file input
    test_input = sys.argv[2]             #test file input 
    dimension = int(sys.argv[3])        #dimension input
    
    #as the train and test files are .file extension the columns were 900x1, to be readable as 900x12 or 11
    train_data = pd.read_csv(train_input, delimiter="\s+", header=0)
    train_data.columns = ['f_acid', 'v_acid', 'c_acid', 'res_sugar', 'chlorides',
                'fs_dioxide', 'ts_dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'quality']
    test_data = pd.read_csv(test_input, delimiter="\s+", header=0)
    test_data.columns = ['f_acid', 'v_acid', 'c_acid', 'res_sugar', 'chlorides',
                'fs_dioxide', 'ts_dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    train_data = train_data[train_data['quality'].isin([5, 6, 7])]        #as does not want the quality of 4,8,9
    
    predictions = predict_quality(train_data, test_data, dimension, k=1)      #kNN is 1NN
    
    print(*predictions, sep="\n")       #to print in rows 