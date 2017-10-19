import numpy
from scipy.sparse import csr_matrix
from sklearn import metrics
from bisect import bisect_left
import os.path
import pickle

#saves a spare csr matrix as .npz
def save_sparse_csr(filename, x):
    numpy.savez(filename, data=x.data, indices=x.indices,
                indptr=x.indptr, shape=x.shape)

#loads a spare csr matrix
def load_sparse_csr(filename):
    csr = numpy.load(filename)
    return csr_matrix((csr['data'], csr['indices'], csr['indptr']),
                      shape=csr['shape'])

#calculates and prints silhuette score
def silhuette(matrix,labels):
    score = metrics.silhouette_score(matrix, labels)
    print("Silhouette Coefficient: " + str(score))
    return score

#uses bisection to find item in arr and returns false if item is not present in arr
def binsearch(arr,item):
    index = bisect_left(arr,item)

    if index == len(arr) or arr[index] != item:
        return False
    return index

#creates the directory if it does not exist
def mkdir(dir):
    try:
        os.mkdir(dir)
        return True
    except OSError:
        return False

#checks if a file exists
def file_exists(filename):
    return os.path.isfile(filename)


#saves an object in binary
def save_model(model,out):
    file = open(out+ ".obj", "wb")
    pickle.dump(model, file)

#loads an object from binary file
def load_model(filename,enc):
    file = open(filename+ ".obj", "rb")
    model = pickle.load(file,encoding=enc)
    return model

