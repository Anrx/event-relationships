import numpy as np
from scipy.sparse import csr_matrix
from sklearn import metrics
from bisect import bisect_left
from datetime import date
from scipy.stats import threshold
import os.path
import pickle
import json
import ast
import math
from datetime import date,timedelta
#saves a spare csr matrix as .npz
def save_sparse_csr(filename, x):
    np.savez(filename, data=x.data, indices=x.indices,
                indptr=x.indptr, shape=x.shape)

#loads a spare csr matrix
def load_sparse_csr(filename):
    csr = np.load(filename+".npz")
    return csr_matrix((csr['data'], csr['indices'], csr['indptr']),
                      shape=csr['shape'])

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
    with open(out+ ".obj", "wb") as f:
        pickle.dump(model, f)

#loads an object from binary file
def load_model(filename,enc):
    with open(filename+ ".obj", "rb") as f:
        model = pickle.load(f,encoding=enc)
    return model

#normalizes given value to a range of 1
def normalize(val,min,max):
    return 0 if (max-min)==0 else (val-min)/(max-min)

#returns a date object from a string like YYYY-MM-DD
def str_to_date(d):
    d=d.split("-")
    return date(int(d[0]),int(d[1]),int(d[2]))

#returns a string object from a date like YYYY-MM-DD
def date_to_str(d):
    return d.strftime('%Y-%m-%d')

def save_as_json(obj,out):
    with open(out + ".json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def load_as_json(fileName):
    with open(fileName + ".json", "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj

def HasConcept(concepts,cid):
    clist=ast.literal_eval(concepts)
    for id,_ in clist:
        if id==cid:
            return True
    return False

def AverageDate(cluster):
    dates = cluster.date
    dateMin = str_to_date(dates.min())
    dateMax = str_to_date(dates.max())
    dateDelta = (dateMax - dateMin).days
    avgDate = dateMin + timedelta(days=dateDelta / 2)
    cluster["avgDate"]=avgDate
    return cluster

def GroupBySorter(cluster,sortby):
    cluster.sort_values(sortby, inplace=True)
    return cluster

# compute mean by dataframe row
def NoZeroMean(d):
    return np.nanmean(d.values, axis=1)

def sigmoid(z):
    return [1 / (1 + math.exp(-n)) for n in z]

def bin_encode(x,limit=0.5):
    x=threshold(x,threshmin=limit,newval=0)
    x=threshold(x,threshmax=limit,newval=1)
    return x

def string_to_object(x):
    return ast.literal_eval(x)
