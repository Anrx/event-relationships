from eventregistry import *
from sklearn.decomposition import NMF
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.metrics import hamming_loss
from sklearn.metrics import silhouette_score
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from sklearn.metrics import calinski_harabaz_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from mlxtend.frequent_patterns import apriori
from sklearn.model_selection import train_test_split
from scipy.cluster.hierarchy import dendrogram, linkage, cophenet, fcluster
from scipy.spatial.distance import pdist
from collections import OrderedDict
from src.helpers import *
from src.apriori import *
from datetime import date,timedelta
from collections import defaultdict
from itertools import islice
import json
import pandas as pd
import scipy
import scipy.stats
import numpy as np
import logging
import time
import math
import ast
import gc
import pickle
import os.path
import matplotlib.pyplot as plt

class EventRelationships:
    tab = "\t"
    nl = "\n"
    sep = ";"
    enc = "utf-8"
    results_subdir = "results"
    models_subdir = "models"
    data_subdir = "data"
    temp_subdir = "temp"
    events_filename = "events"
    concepts_filename = "concepts"
    categories_filename = "categories"
    seed=32435424

    def __init__(self, events_filename="events",concepts_filename="concepts",categories_filename="categories"):
        self.events_filename = events_filename
        self.concepts_filename = concepts_filename
        self.categories_filename = categories_filename
        mkdir(self.results_subdir)
        mkdir(self.models_subdir)
        mkdir(self.data_subdir)



    ### Get Data #########################################################################################################################

    def ConceptIdToName(self,conceptIds):
        if len(conceptIds)>1:
            concepts=self.GetConcepts()
            selected = pd.Series()
            for cid in conceptIds:
                concept = concepts.loc[concepts["conceptId"] == str(cid)]
                selected = selected.append(concept, ignore_index=True)
            return selected.title.values
        else:
            return []

    def GetConceptIdToNameLookupDict(self):
        concepts=self.GetConcepts()
        ids = concepts.conceptId.values
        names = concepts.title.values
        return {id:name for id,name in zip(ids,names)}

    def GetEvents(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.events_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col) # read events into dataframe

    def GetConcepts(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.concepts_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col) # read concepts into dataframe

    def GetCategories(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.categories_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col)  # read categories into dataframe

    def GetHeatmap(self):
        return pd.read_csv(os.path.join(self.data_subdir,"conceptHeatmapFreq")+".csv", sep=self.sep, encoding=self.enc, index_col=0)  # read frequencies into dataframe

    def GetConceptCount(self):
        with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
            conceptCount = json.load(f)
        return conceptCount

    def GetValidConcepts(self,min_events):
        conceptCount = self.GetConceptCount()
        validConcepts=set()
        for id,props in conceptCount.items():
            if props[1]>=min_events: validConcepts.add(id)
        return validConcepts

    def GetClusters(self,labels=None,events=None,groupLabels=None,sorted=True,inplace=True):
        if events is None:
            events=self.GetEvents()
        if not inplace:
            events =events.copy()
        if labels is not None:
            events["cluster"]=labels

        if groupLabels is not None:
            groupClusterLabels=[]
            for lab in labels:
                groupClusterLabels.append(groupLabels[lab])
            events["groupCluster"]=groupClusterLabels

        if sorted:
            events=events.groupby("cluster").apply(GroupBySorter,("date"))
            if groupLabels is not None:
                events=events.groupby("cluster").apply(AverageDate)
                events=events.groupby("groupCluster").apply(GroupBySorter,("avgDate"))

        return events

    def LoadData(self):
        self.concepts = self.GetConcepts(index_col=0)
        self.conceptCount = self.GetConceptCount()
        self.events = self.GetEvents(index_col=0)
        self.categores=self.GetCategories(index_col=0)



    ### Sparse Matrix #########################################################################################################################

    #compute spare matrix from data
    def SparseMatrix(self,events=None,normalized=False):
        if events is None:
            events = self.GetEvents()
        eventsConcepts = events["concepts"]  # get just the event concepts

        concepts = self.GetConcepts()
        uniqueConceptIds = concepts["conceptId"].astype(int)  # get just the concept ids
        uniqueConceptIds.sort_values(inplace=True)  # sort the list for bisect search

        matrix = []
        for eventConcepts in eventsConcepts:
            conceptsList = ast.literal_eval(eventConcepts)  # list of tuples like (conceptId,score)
            bins = np.zeros(uniqueConceptIds.size, dtype=int)
            for id, score in conceptsList:
                index = binsearch(uniqueConceptIds.tolist(), int(id))
                if normalized: score = normalize(score, 0, 100)
                bins[index] = score
            matrix.append(bins)
        return matrix

    #compute csr matrix from data
    def CsrMatrix(self,out=None,normalized=False,concept_wgt=1,include_date=False,date_wgt=1,date_max=None,min_events=0,events=None,excluded_types = None):
        print("Building CSR matrix")
        if events is None:
            events = self.GetEvents()
        eventsConcepts = events["concepts"]  # get just the event concepts
        if excluded_types is not None:
            concepts = self.GetConcepts(index_col = 0)
        if include_date:
            eventDates=events["date"]
            dateMin=str_to_date(eventDates.min())
            dateMax=str_to_date(date_max) if date_max is not None else str_to_date(eventDates.max())
            dateDelta = (dateMax-dateMin).days
            eventDates=eventDates.values

        indptr=[0]
        indices = []
        data=[]
        vocabulary = {}
        if min_events>1:
            with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
                conceptCount = json.load(f)
        excluded_concepts=0
        excluded_events=0
        for i,eventConcepts in enumerate(eventsConcepts):
            conceptsList = ast.literal_eval(eventConcepts)  # list of tuples like (conceptId,score)
            included=False
            #build csr vectors
            for id,score in conceptsList:
                if (min_events<2 or conceptCount[str(id)][1]>=min_events) and (excluded_types is None or concepts.loc[id, :].type not in excluded_types):
                    included=True
                    index = vocabulary.setdefault(id,len(vocabulary))
                    indices.append(index)
                    if normalized: score = normalize(score,0,100)
                    data.append(score*concept_wgt)
                else:
                    excluded_concepts+=1

            #include date dimension
            if include_date and included:
                event_date=(str_to_date(eventDates[i])-dateMin).days
                index = vocabulary.setdefault("date",len(vocabulary))
                indices.append(index)
                if normalized: event_date = normalize(event_date, 0, dateDelta)
                data.append(event_date*date_wgt)

            if included:
                indptr.append(len(indices))
            else:
                excluded_events+=1

        matrix = csr_matrix((data, indices, indptr), dtype=int,shape=(len(indptr)-1,len(vocabulary)))
        if out is not None:
            save_sparse_csr(out,matrix)
            #save_as_json(vocabulary,os.path.join(self.data_subdir, out))

        cc = ((len(indptr)-1)*len(vocabulary))/len(data) if len(indptr)-1>0 else 0
        print("n samples: " + str(len(indptr)-1))
        print("n dimensions: "+str(len(vocabulary)))
        print("n excluded concepts: "+str(excluded_concepts))
        print("n excluded events: "+str(excluded_events))
        print("cover coeficient: "+str(cc))

        return (matrix,vocabulary)



    ### Clustering Algorithms ######################################################################################################################

    # nmf clustering
    def NMF(self,x,nClusters,init="nndsvd",seed=None,maxiter=100,out=None):
        model = NMF(n_components=nClusters, init=init, random_state=seed, max_iter=maxiter)
        model.fit(x)
        W = model.transform(x)
        labels=[]
        for sample in W:
            labels.append(np.argmax(sample))
        #H = model.components_

        if out is not None:
            save_model(model,out)
        return (model,labels)

    # dbscan clustering
    def DBSCAN(self,x,eps=0.5, min_samples=5, metric="euclidean", metric_params=None, algorithm="auto", leaf_size=30, p=None, n_jobs=1,out=None):
        db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, metric_params=metric_params, algorithm=algorithm, leaf_size=leaf_size, p=p, n_jobs=n_jobs).fit(x)
        if out is not None:
            save_model(db,out)
        return (db,db.labels_)

    #k-means clustering
    def KMeans(self, x, n_clusters,seed=None, init="k-means++", max_iter=300, n_init=10, n_jobs=1,verbose=0, useMiniBatchKMeans=False,out=None, batch_size=100):
        from sklearn.cluster import KMeans

        print("Training K-Means")
        if useMiniBatchKMeans:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                verbose=verbose,
                batch_size=batch_size)
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init,
                random_state=seed,
                max_iter=max_iter,
                n_init=n_init,
                n_jobs=n_jobs,
                verbose=verbose)
        kmeans=kmeans.fit(x)

        if out is not None:
            save_model(kmeans,out)
        return (kmeans,kmeans.labels_)

    def CosineKmeans(self,x,n_clusters):
        from sklearn.cluster import k_means_
        from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

        # Manually override euclidean
        def euc_dist(X, Y=None, Y_norm_squared=None, squared=False):
            # return pairwise_distances(X, Y, metric = 'cosine', n_jobs = 10)
            return cosine_similarity(X, Y)

        k_means_.euclidean_distances = euc_dist

        kmeans = k_means_.KMeans(n_clusters=n_clusters, random_state=1,max_iter=100,n_init=10)
        _ = kmeans.fit(x)
        return kmeans,kmeans.labels_

    # affinity propagation clustering
    def AffinityPropagation(self, x, damping=0.5,max_iter=200,convergence_iter=15,verbose=False):
        print("AffinityPropagation")
        affinity = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter,
            verbose=verbose)
        affinity = affinity.fit(x)
        return (affinity, affinity.labels_)

    # agglomerative (hierarichal) clustering
    def AgglomerativeClustering(self,x,n_clusters=2,affinity="euclidean",caching_dir=None,connectivity=None,full_tree="auto", method="ward",imp="sklearn",out=None,criterion="distance",score=False):
        print("AgglomerativeClustering")
        if imp.lower()=="scipy":
            z = linkage(x,method=method,metric=affinity)

            if out is not None:
                save_model(z,out)

            if score:
                score, coph_dists = cophenet(z, pdist(x))
                print(str(score))

            if n_clusters is not None:
                labels = fcluster(z, n_clusters, criterion=criterion)
                if(criterion=="distance"): print("N clusters: "+str(len(np.unique(labels))))
                return z,labels
            else: return z

        elif imp.lower()=="sklearn":
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                memory=caching_dir,
                connectivity=connectivity,
                compute_full_tree=full_tree,
                linkage=method)
            model.fit(x)

            if out is not None:
                save_model(model,out)

            return (model,model.labels_)

    # birch clustering
    def Birch(self,x,threshold=0.5,branching_factor=50,n_clusters=3,compute_labels=True,copy=True,out=None):
        brc = Birch(branching_factor=branching_factor, n_clusters=n_clusters, threshold=threshold,compute_labels = compute_labels,copy=copy)
        brc.fit(x)

        if out is not None:
            save_model(brc,out)

        return (brc,brc.predict(x))

    # mean shift clustering
    def MeanShift(self,x,bandwidth=None, seeds=None, bin_seeding=False, min_bin_freq=1, cluster_all=True, n_jobs=1,out=None):
        model = MeanShift(bandwidth=bandwidth, seeds=seeds, bin_seeding=bin_seeding, min_bin_freq=min_bin_freq, cluster_all=cluster_all, n_jobs=n_jobs).fit(x)
        if out is not None:
            save_model(model,out)

        return (model,model.label_)

    # spectral clustering
    def SpectralClustering(self,x,n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, n_jobs=1,out=None):
        model=SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma,affinity=affinity, n_neighbors = n_neighbors, eigen_tol = eigen_tol, assign_labels =assign_labels, degree = degree, coef0 = coef0, kernel_params = kernel_params, n_jobs = n_jobs)
        model.fit(x)

        if out is not None:
            save_model(model,out)

        return (model,model.fit_predict(x))

    def Clustering2KNN(self,labels,csrMatrix,out=None,n_neighbors=5, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=1,verbose=False):
        print("Clustering2KNN")
        from sklearn.neighbors import KNeighborsClassifier
        model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, leaf_size=leaf_size, p=p, metric=metric, metric_params=metric_params, n_jobs=n_jobs)
        model.fit(csrMatrix, labels)

        if out is not None:
            save_model(model,out)

        return model


    ### Dimensionality Reduction ######################################################################################################################

    def TruncatedSVD(self,x,n_components = 2,algorithm = "randomized",n_iter = 5,random_state=None,tol=0.0,out=None):
        print("TruncatedSVD")
        svd = TruncatedSVD(n_components=n_components,algorithm=algorithm, n_iter=n_iter, random_state=random_state,tol=tol)
        fitted = svd.fit_transform(x)
        print(svd.explained_variance_ratio_.sum())

        if out is not None:
            save_model(svd,out)

        return fitted,svd

    def PCA(self,x,n_components=None, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", random_state=None,verbose=False):
        from sklearn.decomposition import PCA
        print("pca")
        pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
        fitted=pca.fit_transform(x)
        #print(pca.explained_variance_ratio_)
        print(pca.explained_variance_ratio_.sum())
        #print(pca.singular_values_)
        return fitted,pca

    # deprecated
    def ClusterByTimeWindow(self,events=None,window_size=14,min_events=50,clusters=None):
        if events is None:
            events=self.GetEvents()

        dateMin = str_to_date(events.date.min())
        dateMax = str_to_date(events.date.max())
        dates = pd.to_datetime(events.date)

        startDate = dateMin
        endDate = dateMin + timedelta(days=window_size)

        if clusters is None: clusters = {}
        else:
            startDate=clusters.keys().max()
            endDate = startDate+timedelta(days=window_size)
            if endDate > dateMax and startDate != dateMax: endDate = dateMax

        rest=pd.DataFrame()
        while (endDate <= dateMax):
            print(date_to_str(startDate) + " - " + date_to_str(endDate))

            #get events
            date_filter = np.logical_and(dates >= startDate, dates < endDate)
            group = events.loc[date_filter]
            if rest.size>0: group=group.append(rest,ignore_index=True)

            # skip until next cluster if <10 samples
            if group.size < 10:
                print("skipping due to low sample size")
                rest = rest.append(group, ignore_index=True)

                startDate = endDate
                endDate = (endDate + timedelta(days=window_size))
                continue
            else:
                rest = pd.DataFrame()

            #build matrix
            csr,vocab= self.CsrMatrix(events=group,min_events=min_events,verbose=True)

            #cluster
            model,labels=self.AffinityPropagation(csr,verbose=True)
            #model, labels = self.KMeans(csr, int(nclust), useMiniBatchKMeans=False,seed=1)
            #model,labels = self.AgglomerativeClustering(csr,n_clusters=int(nclust))

            group["cluster"]=labels
            clusters[startDate] = (model, group)

            #info
            #self.CountClusterSize(labels,plot=False,events=group)
            #silhuette(csr,labels)
            #count = self.CountConcepts(events=group, method="score", sorted=False, min_score=0)

            startDate = endDate
            endDate = (endDate + timedelta(days=window_size))
            if endDate>dateMax and startDate!=dateMax: endDate=dateMax

        return clusters

    # deprecated
    def DatasetByTimeWindow(self,clusters,window_size=14,min_events=50,min_score=10):
        print("Making dataset")

        for model,group in clusters.values():
            for cluster in group.groupby("cluster"):
                pass



    ### Cross validation ######################################################################################################################

    # deprecated
    def PredictByCluster(self,date_start="2017-09-04",date_end="2017-09-18",window_size=7,min_events=50,min_score=10):
        dateRange = pd.date_range(str_to_date(date_start), str_to_date(date_end)).tolist()

        for date in dateRange:
            print(date_to_str(date))
            train,test=self.TrainTestSplit(date_to_str(date))

            clusters=self.ClusterByTimeWindow(events=train,window_size=window_size,min_events=min_events)

            print("k")

    def CrossValidateByCluster(self,date_start="2016-08-29",date_end="2016-09-11",min_score=0,n_clusters=1600,n_dims=1000,window_size=14,verbose=True,separate=False,min_events=10,lazy=True):
        from src.dset import Dset

        dateRange = pd.date_range(str_to_date(date_start), str_to_date(date_end)).tolist()

        hammingScores=[]
        hammingLossScores=[]
        precisionScores = []
        recallScores = []
        fscoreScores = []

        constantHammingScores=[]
        constantHammingLossScores = []
        constantPrecisionScores = []
        constantRecallScores = []
        constantFscoreScores = []

        train=None
        trainModel=None
        train_set=None
        clfs=None
        tsvd=None
        for date in dateRange:
            #split dset by date
            date = date.strftime('%Y-%m-%d')
            print(date)

            if lazy and train is not None:
                _, test = self.TrainTestSplit(date)
            else:
                train,test = self.TrainTestSplit(date)

            #create matrix and train clustering
            trainMatrix,trainVocab=self.CsrMatrix(events=train,min_events=min_events)
            #trainMatrix, trainVocab = self.CsrMatrix(events=train,min_events=min_events,date_wgt=100,concept_wgt=100,normalized=True,include_date=True,verbose=verbose)

            if not lazy or tsvd is None:
                try:
                    tsvd = load_model("tsvd" + date, enc=self.enc)
                    trainMatrixFull = tsvd.transform(trainMatrix)
                except FileNotFoundError:
                    trainMatrixFull,tsvd = self.TruncatedSVD(trainMatrix, n_components=n_dims, random_state=self.seed, algorithm="randomized",out="tsvd"+date)
            #else:
            #    trainMatrixFull = tsvd.transform(trainMatrix)


            # cluster train set
            if not lazy or trainModel is None:
                try:
                    trainModel=load_model("scipyAgg"+date,enc=self.enc)
                    #trainLabels=trainModel.labels_
                    trainLabels = fcluster(trainModel, n_clusters, criterion="distance")
                    print("n clusters:"+str(len(set(trainLabels))))
                except FileNotFoundError:
                    #trainModel,trainLab els= self.KMeans(trainMatrix,n_clusters,useMiniBatchKMeans=False,nar=verbose,seed=1,verbose=False,out="crossValCluster"+date)
                    #trainModel,trainLabels = self.AgglomerativeClustering(trainMatrix,n_clusters=n_clusters,out="crossValClusterAgg"+date)
                    trainModel, trainLabels = self.AgglomerativeClustering(trainMatrixFull, n_clusters=n_clusters, affinity="euclidean", method="ward", out="scipyAgg" + date, imp="scipy")
                train["cluster"]=trainLabels

            # performance info
            #silhuette(trainMatrix,trainLabels)
            #self.ShowRelationships(None, "crossValClustering",train)
            #self.CountClusterSize(trainLabels,plot=False,events=train)

            # build dataset
            if not lazy or train_set is None:
                try:
                    train_set=load_model("dset"+date,self.enc)
                    train_set.Analyze()
                except (FileNotFoundError,EOFError):
                    train_set = Dset(min_events=min_events)
                    train_set.Compile(train,out="dset"+date)

            # train model
            if not lazy or clfs is None:
                try:
                    clfs = load_model("crossValPred"+date,self.enc)
                except (FileNotFoundError,EOFError):
                    #clfs=self.RandomForest(train_set,random_state=1,n_estimators=10,bootstrap=True,out="crossValPred"+date)
                    #clfs=self.NeuralNetwork(train_set)
                    clfs = self.OneVsRest(train_set,out="crossValPred"+date)
                constantClfs=self.MultiLabelMostFrequent(train,n_labels=10,filter=list(train_set.yset.keys()))

            # cluster today's events
            today=test.loc[test.date==date]
            todayCsr=self.CsrFromVocab(trainVocab,today,verbose=verbose)
            knnModel = self.Clustering2KNN(train.cluster, trainMatrix)
            todayLabels=knnModel.predict(todayCsr)
            #todayLabels=trainModel.predict(todayCsr)
            today.loc[:,"cluster"]=todayLabels

            print("Building test set")
            test_set = Dset(train_set)
            for todayLabel in set(todayLabels):
                trainCluster=train.loc[train.cluster==todayLabel]
                todayCluster=today.loc[today.cluster==todayLabel]

                startDate = (str_to_date(date) - timedelta(days=window_size)).strftime('%Y-%m-%d')
                cond = np.logical_and(trainCluster.date>=startDate, trainCluster.date<date)
                todayTrainCluster=trainCluster.loc[cond]

                test_set.Concat(todayTrainCluster,todayCluster)

            print("Predicting")
            train_x,test_y = test_set.ToArray()
            #train_x=tsvd.transform(train_x)
            train_y = clfs.predict(train_x)
            #train_y = bin_encode(clfs.predict(train_x),limit=0.1)
            constant_train_y = self.MLMFPredict(constantClfs, test_set.yset)

            print("Evaluation")
            hammingScore = self.HammingScore(test_y, train_y, verbose=verbose, xset=test_set.xset, yset=test_set.yset,separate=separate,print_names=False)
            hammingLoss = hamming_loss(test_y,train_y)
            prfs = precision_recall_fscore_support(test_y,train_y,average="samples")

            constantHammingScore = self.HammingScore(test_y, constant_train_y, verbose=verbose, xset=test_set.xset, yset=test_set.yset,separate=separate)
            constantHammingLoss = hamming_loss(test_y, constant_train_y)
            constantPrfs = precision_recall_fscore_support(test_y, constant_train_y, average="samples")

            hammingScores.append(hammingScore)
            hammingLossScores.append(hammingLoss)
            precisionScores.append(prfs[0])
            recallScores.append(prfs[1])
            fscoreScores.append(prfs[2])

            constantHammingScores.append(constantHammingScore)
            constantHammingLossScores.append(constantHammingLoss)
            constantPrecisionScores.append(constantPrfs[0])
            constantRecallScores.append(constantPrfs[1])
            constantFscoreScores.append(constantPrfs[2])

            #print(classification_report(test_y, train_y)) #,target_names=self.ConceptIdToName([yset.keys()])
            print("Hamming score for " + date + " is " + str(hammingScore))
            print("Hammming loss score for " + date + " is " + str(hammingLoss))
            print("precision_recall_fscore_support for " + date + " is " + str(prfs))

            print("Constant Hamming score for " + date + " is " + str(constantHammingScore))
            print("Hammming loss score for " + date + " is " + str(constantHammingLoss))
            print("precision_recall_fscore_support for " + date + " is " + str(constantPrfs))

            if lazy:
                train = train.append(today,ignore_index=True)
                train_set.Merge(test_set)

        # final average scores
        print("Mean Hamming Score: " +str(np.mean(hammingScores)))
        print("Mean Hamming loss score: "+ str(np.mean(hammingLossScores)))
        print("Mean Precision score: "+ str(np.mean(precisionScores)))
        print("Mean Recall score: "+ str(np.mean(recallScores)))
        print("Mean Fscore score: "+ str(np.mean(fscoreScores)))

        print("Mean constant Hamming score: "+ str(np.mean(constantHammingScores)))
        print("Mean constant Hamming loss score: "+ str(np.mean(constantHammingLossScores)))
        print("Mean constant Precision score: "+ str(np.mean(constantPrecisionScores)))
        print("Mean constant Recall score: "+ str(np.mean(constantRecallScores)))
        print("Mean constant Fscore score: "+ str(np.mean(constantFscoreScores)))

    ### Multi-label prediction ######################################################################################################################

    def OneVsRest(self,dset,out=None):
        print("OneVsRest")
        x,y = dset.ToArray()
        model = OneVsRestClassifier(LinearSVC())
        model.fit(x,y)

        if out is not None:
            save_model(model,out)

        return model

    # random forest prediction
    def RandomForest(self, dset,n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None,out=None):
        print("Training random forest")

        x,y = dset.ToArray()
        model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                       min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                       min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                       max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                       min_impurity_split=min_impurity_split, bootstrap=bootstrap,
                                       oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                       verbose=verbose, warm_start=warm_start, class_weight=class_weight)

        model.fit(x, y)
        if out is not None:
            save_model(model,out)
        return model

    def MultiLabelMostFrequent(self, events, n_labels=10,min_score=0,filter=None):
        print("Training most frequent classfier")
        freq=self.CountConcepts(events=events,desc=True,min_score=0,filter=filter)
        model=list(freq.keys())[:n_labels]
        return model

    def MLMFPredict(self,model,yset):
        concepts=list(yset.keys())
        train_y=OrderedDict()
        n_samples=len(yset[concepts[0]])
        for cid in concepts:
            train_y[cid]=[1 if cid in model else 0 for _ in range(n_samples)]

        return np.asarray(list(train_y.values())).T

    # apriori association rule prediction
    def Apriori(self, support=0.01, minlen=1):
        data = pd.read_csv(os.path.join(self.data_subdir, "AssociationRules") + ".csv", sep=self.sep,
                           encoding=self.enc, index_col=0)
        frequent_itemsets = filter_frequent(data, min_support=0.05, use_colnames=True)
        print(frequent_itemsets)

    # compute dataset from heatmap for association rule learning
    def CompileTransactions(self,labels,events=None,out=None,min_score=50,conceptNames=None,conceptIds=None):
        if events is None:
            events = self.GetEvents()
        events["cluster"] = labels

        if conceptIds is None:
            conceptIds=self.ConceptNameToId(conceptNames)

        conceptSet=self.ConceptVocab(events,min_score)

        # construct dataset
        dset={id:[] for id in conceptSet}
        for cluster in events.groupby("cluster"):
            for _,event in cluster.iterrows():
                concepts = ast.literal_eval(event.concepts)
                for id,score in concepts:
                    if id in dset:
                        if score>=min_score:
                            dset[id].append(1)
                        else:
                            dset[id].append(0)

        for id in conceptIds:
            with open(os.path.join(self.data_subdir, str(id)) + "_x.txt", "w", encoding=self.enc,newline=self.nl) as x, \
                    open(os.path.join(self.data_subdir, str(id)) + "_y.txt", "w", encoding=self.enc, newline=self.nl) as y:
                x.write(self.sep.join(conceptSet) + self.nl)
                y.write(self.sep.join(conceptSet) + self.nl)
                for _, cluster in events.groupby("cluster"):
                    events = self.EventsByConceptScore(cluster,id,min_score=min_score)
                    for eventId,event in events.items():
                        prev=cluster.loc[cluster.date<event.date]
                        fol=cluster.loc[cluster.date>event.date]
                        #TODO
                        #if len(prev)>=min_events and len(fol)>=min_events:
                        #    self.Compile(OrderedDict(usedConcepts),events=prev,f=x,min_score=min_score)
                        #    self.Compile(events=fol,OrderedDict(usedConcepts),events=prev,f=y,min_score=min_score)

    def ConceptVocab(self,events,min_score):
        conceptVocab=OrderedDict()
        for concepts in events.concepts:
            id,score = ast.literal_eval(concepts)
            if score>=min_score:
                conceptVocab[id]=None

        return list(conceptVocab.keys())

    def AssociationRulesByWeek(self,out=None):
        heatmap = pd.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0, na_values=0)  # read frequencies into dataframe
        events = self.GetEvents()
        eventDates = events["date"]
        dateMin = str_to_date(eventDates.min())
        dateMax = str_to_date(eventDates.max())
        weeks = pd.date_range(dateMin, dateMax, freq="W").tolist()

        averages = self.NoZeroMean(heatmap)

        df = pd.DataFrame(columns=heatmap.index, index=weeks)

        columns=pd.to_datetime(heatmap.columns)
        for i,week in enumerate(weeks):
            print(i)
            nextWeek = weeks[-1] if i+1==len(weeks) else weeks[i+1]
            cond = np.logical_and(columns > week, columns <= nextWeek)
            those = heatmap.loc[:,cond]
            for j,id in enumerate(heatmap.index):
                avg = averages[j]
                df.at[week,id] = 1 if any(those.loc[id,:]>avg) else 0

        if out is not None:
            df.to_csv(os.path.join(self.data_subdir,out)+".csv",sep=self.sep,na_rep=0,encoding=self.enc)

        return df

    def NeuralNetwork(self,dset,out=None):
        from keras.models import Sequential
        from keras.layers import Dense
        print("Training neural network")

        x,y = dset.ToArray()
        nn = Sequential()
        nn.add(Dense(x.shape[1], activation="relu", input_shape=(x.shape[1],)))
        nn.add(Dense(y.shape[1], activation="sigmoid"))
        nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        nn.fit(x, y, batch_size=16, epochs=5,verbose=0, validation_split=0.1)

        if out is not None:
            save_model(nn,out)

        return nn



    ### Outlier Detection ######################################################################################################################

    def IsolationForest(self,x,n_estimators=100,max_samples="auto",contamination=0.1,max_features=1.0,bootstrap=False,n_jobs=1,seed=None,verbose=0):
        model = IsolationForest()
        model.fit(x)
        return model.predict(x)



    ### Validation metrics ######################################################################################################################

    def HammingScore(self,y_true, y_pred,verbose=False,xset=None,yset=None,print_names=False,separate=True):
        '''
        Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
        https://stackoverflow.com/q/32239577/395857
        '''
        if verbose:
            print("Calculating Hamming Score / Jaccard Index")
        if not separate:
            nsamples=0
            npred=0
            ntrue=0
            ncorrect=0
            nblank=0
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])

            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
                nblank+=1
            else:
                if verbose:
                    #print("N Predicted: "+str(len(set_pred)))
                    #print("N True: "+str(len(set_true)))
                    #print("N Correctly predicted: "+str(len(set_true.intersection(set_pred))))
                    nsamples+=1
                    npred+=len(set_pred)
                    ntrue+=len(set_true)
                    ncorrect+=len(set_true.intersection(set_pred))
                tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))

            acc_list.append(tmp_a)
            if print_names:
                train_x = np.asarray(list(xset.values()))

                set_attr = set(np.where(train_x.T[i])[0])
                ids = [list(xset.keys())[i] for i in set_attr]
                names = self.ConceptIdToName(ids)
                print("Attributes: ")
                print(names)

                ids = [list(yset.keys())[i] for i in set_pred]
                names=self.ConceptIdToName(ids)
                print("Predicted classes: ")
                print(names)

                ids = [list(yset.keys())[i] for i in set_true]
                names = self.ConceptIdToName(ids)
                print("Actual classes: ")
                print(names)

        if not separate:
            print("N Samples: "+str(nsamples))
            print("N Predicted: " + str(npred))
            print("N True: " + str(ntrue))
            print("N Correctly predicted: " + str(ncorrect))
            print("N Blank: " + str(nblank))
        return np.mean(acc_list)



    ### Machine Learning ######################################################################################################################

    # show cluster shapes
    def CountClusterSize(self,labels,plot=True,frequency=True,events=None):
        cs = dict.fromkeys(set(labels),0)
        for label in labels:
            cs[label]+=1
        print(len(cs))
        if(frequency):
            largest=max(cs.items(), key=lambda cpt: cpt[1])[0]
            largestCluster = events.loc[events.cluster == largest]
            counts = self.CountConcepts(events=largestCluster, desc=True, min_score=0,n_concepts=10,include_names=True)
            print(counts)
        print(list(cs.values()))

        if plot:
            self.Plot(list(cs.values()))

        return cs

    # compute top concepts for each cluster
    def GetClusterConcepts(self,clusters,n_concepts=20,method="frequency",out=None):
        df=pd.DataFrame(columns=["cluster", "concepts"])
        incoming = []
        for label,cluster in clusters.groupby("cluster"):
            print(label)
            count = self.CountConcepts(events=cluster,method=method,desc=True,min_score=0)
            count = OrderedDict(islice(count.items(), min(n_concepts,len(count))))
            if method.lower() == "score".lower():
                concepts = ",".join(map(lambda c: "(" + str(c[0]) + "," + str(c[1][0]) + ")", count.items()))
            elif method.lower() == "frequency".lower():
                concepts = ",".join(map(lambda c: "(" + str(c[0]) + "," + str(c[1][1]) + ")", count.items()))
            elif method.lower() == "average".lower():
                concepts = ",".join(map(lambda c: "(" + str(c[0]) + "," + str(c[1][0] / c[1][1]) + ")", count.items()))
            incoming.append({"cluster": label, "concepts":concepts})
        df = df.append(incoming,ignore_index=True)

        if out is not None:
            df.to_csv(os.path.join(self.data_subdir, out) + ".csv", sep=self.sep, na_rep=0, encoding=self.enc,index=False)

        return df

    # split dataset into train and test sets by date
    # date: "yyyy-mm-dd"
    def TrainTestSplit(self,date,out=False):
        print("Splitting dataset to train and test sets by date: " +date)
        events = self.GetEvents()

        train = events.loc[events.date<date]
        test = events.loc[events.date>=date]
        if out:
            train.to_csv(os.path.join(self.data_subdir, self.events_filename) + "_train.csv",sep=self.sep,index=False,encoding=self.enc)
            test.to_csv(os.path.join(self.data_subdir, self.events_filename) + "_test.csv",sep=self.sep,index=False,encoding=self.enc)
        return train,test

    #average of average distances within clusters/average of average distances within clusters in chronological order
    def AverageChronologicalLinkage(self,events,metric="euclidean"):
        from sklearn.metrics import pairwise
        csr, vocab = self.CsrMatrix(events=events)
        matrix = csr.toarray()
        chrons=0
        nc=0
        for _,cluster in events.groupby("cluster"):
            rows = [(x[1] if type(x) is tuple else x) for x in cluster.index.values]
            n=len(rows)
            if n <= 1:
                #print("n==1, skipping")
                continue
            clusterArray = matrix[rows,]
            try:
                distances = pairwise.pairwise_distances(clusterArray,metric = metric)
            except Exception as e:
                continue
            chronAvg = 0
            normAvg = 0
            for i,eventDistances in enumerate(distances):
                if i>0:
                    chronAvg+=eventDistances[i-1]
                    normAvg+=sum(eventDistances[:i])
            chronAvg/=n-1
            normAvg/=(((n-1)*n)/2)
            chron = (normAvg/chronAvg)-1
            #print("chron: "+str(chron))
            chrons+=chron
            #chrons+=chronAvg
            nc+=1
        avgChron = chrons/nc
        #print("avg chron: "+str(avgChron))
        return avgChron



    #optimize a clustering algorithm using silhouette score
    def Cluster(self, events=None, min=1000, max=2000, step=100, min_events=0,out=None,eval=True, date = "2016-04-25",plot=True):
        if events is None:
            events = self.GetEvents()
        if date is not None:
            events,test = self.TrainTestSplit(date)
        matrix, vocab = self.CsrMatrix(events=events, min_events=min_events)
        #matrix, vocab = self.CsrMatrix(events=events,min_events=min_events,date_wgt=100,concept_wgt=100,normalized=True,include_date=True,verbose=True)
        #fitted,model = self.TruncatedSVD(matrix,n_components=2000,random_state=self.seed,algorithm="randomized")
        #save_model(fitted,"scipyFitted"+date)
        fitted = load_model("scipyFitted" + date, self.enc)
        #matrix = csr_matrix(fitted)
        bestClusters=0
        bestScore=-1
        bestModel=None
        scores = []
        clusters = []
        #model = self.AgglomerativeClustering(fitted, n_clusters=None, affinity="euclidean", method="ward",caching_dir="hierarichal_caching_dir",out="scipyAgg"+date, imp="scipy")
        model = load_model("scipyAgg"+date, self.enc)
        #self.ShowDendrogram(model,p=500)

        for i in range(min, max, step):
            #n_clusters=2**i
            #n_clusters=i/100
            n_clusters=i
            print("n-clusters: "+str(n_clusters))

            #model, labels = self.KMeans(fitted, n_clusters, useMiniBatchKMeans=False, seed=self.seed, verbose=False)
            #model,labels = self.CosineKmeans(matrix,n_clusters)
            #model,labels = self.AgglomerativeClustering(fitted,n_clusters=n_clusters,affinity="cosine",method="single",caching_dir="hierarichal_caching_dir")
            #model,labels = self.DBSCAN(matrix,n_clusters)

            labels = fcluster(model, n_clusters, criterion="distance")
            print("n clusters: "+str(len(set(labels))))

            #self.ShowRelationships(labels, "chron1400aggClust2016-04-25",events)
            if eval:
                try:
                    print("scoring...")
                    #score=silhouette_score(matrix.toarray(), labels, metric="euclidean",sample_size=1000, random_state=self.seed)
                    score = self.AverageChronologicalLinkage(self.GetClusters(labels,events,inplace=False))
                    #score = calinski_harabaz_score(matrix.toarray(),labels)
                    #score, coph_dists = cophenet(model, pdist(fitted))
                    print("score: " + str(score))
                    scores.append(score)
                    clusters.append(n_clusters)

                    if score>bestScore:
                        bestScore=score
                        bestClusters=n_clusters
                        bestModel = model
                except ValueError:
                    print("n labels = 1, pass")
                    raise

        print("Best score: "+str(bestScore) + " for n-clusters: "+str(bestClusters))
        if out is not None:
            save_model(bestModel,out)

        if plot:
            plt.plot(clusters,scores)
            plt.ylabel("score")
            plt.xlabel("n_clusters")
            plt.show()

    # write all the clusters to a file in a human readable format
    def ShowRelationships(self,labels,out,events):
        with open(os.path.join(self.results_subdir,out)+".txt", "w", encoding=self.enc, newline=self.nl) as f:
            clusters = self.GetClusters(labels,events)
            for _,cluster in clusters.groupby("cluster"):
                f.write("New cluster: "+self.nl)
                for _,event in cluster.iterrows():
                    f.write(str(event["title"])+" ["+event["date"]+"] => ")
                f.write(self.nl)

    # same as above, but with clusters of clusters
    def ShowClusterRelationships(self,labels,clusterLabels,out):
        clusters=self.GetClusters(labels,groupLabels=clusterLabels)

        with open(os.path.join(self.results_subdir, out) + ".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for _, group in clusters.groupby("groupCluster"):
                f.write("New group: "+self.nl)
                for _,cluster in group.groupby("cluster"):
                    f.write(self.tab+"New cluster: " + self.nl+self.tab+self.tab)
                    for _, event in cluster.iterrows():
                        f.write(str(event["title"]) + " [" + event["date"] + "] => ")
                    f.write(self.nl)

    # adjust the cluster centers according to the event with the highest social score in each cluster and recluster
    def AdjustCentroids(self,x,labels,method="KMeans"):
        clusters=self.GetClusters(labels)

        # find event with the highest social score in each cluster
        centers = pd.DataFrame()
        for _,cluster in clusters.groupby(level=0):
            best=cluster.loc[cluster["socialScore"].idxmax(),:]
            centers=centers.append(best,ignore_index=True)

        centers = np.asarray(self.SparseMatrix(events=centers))

        if method.lower() == "KMeans".lower():
            model,labels=self.KMeans(x, len(set(labels)),init=centers,n_init=1)
        return (model,labels)

    # compile the clustered events in a multiindex dataframe
    def GetClustersMultiindex(self,labels,sorted=True,events=None,clusterLabels=None,avgDate=False):
        if events is None:
            events=self.GetEvents()
            events["cluster"]=labels
        if clusterLabels is not None:
            clusterConcepts = pd.read_csv(os.path.join(self.data_subdir, "KmeansClusterConcepts") + ".csv",
                                          sep=self.sep, encoding=self.enc)
            clusterConcepts["group"] = clusterLabels

            clusters = {}

            for i,clusterLabel in enumerate(set(clusterLabels)):
                print(i)
                g = clusterConcepts.loc[clusterConcepts["group"] == clusterLabel]
                groupLabels = g["cluster"].values
                e = events.loc[events["cluster"].isin(set(groupLabels))]

                clusters[clusterLabel] = self.GetClustersMultiindex(events=e,labels=groupLabels,avgDate=avgDate)
            print("done")
        else:
            clusters = {label: pd.DataFrame() for label in set(labels)}

            for label in set(labels):
                e = events.loc[events["cluster"]==label]

                if avgDate:
                    eventDates = e["date"]
                    dateMin = str_to_date(eventDates.min())
                    dateMax = str_to_date(eventDates.max())
                    dateDelta = (dateMax - dateMin).days
                    avgDate= dateMin+timedelta(days=dateDelta/2)
                    e["avgDate"]=avgDate.strftime('%Y-%m-%d')

                clusters[label] = clusters[label].append(e, ignore_index=True)
                if sorted: clusters[label].sort_values("date",inplace=True)

        res=pd.concat(clusters)

        return res

    # compute concept score in a set of events
    def GetTopConcepts(self,events):
        counter = defaultdict(int)
        for _,event in events.iterrows():
            conceptsList = ast.literal_eval(event["concepts"])
            for id, score in conceptsList:
                counter[id]+=score

        return counter

    # compute dictionary of events with a specific concept
    def EventsByConceptScore(self,events,conceptId,min_score=50):
        dset=[]
        for _,event in events.iterrows():
            conceptsList = ast.literal_eval(event["concepts"])
            for id, score in conceptsList:
                if id==conceptId and score>=min_score:
                    dset.append(event)
                    break
        return dset

    # search clusters by list of concepts
    def FindEventRelationships(self,labels,conceptNames, n_clusters=10):
        # get selected concepts
        selectedIds = self.ConceptNameToId(conceptNames)

        clusters = self.GetClusters(labels)

        scores = []
        clusterIndexes = []
        for i,cluster in clusters.groupby(level=0):
            conceptCount = self.GetTopConcepts(cluster)
            score = 0
            for conceptId in selectedIds:
                score+=conceptCount[conceptId]
            scores.append(score)
            clusterIndexes.append(i)

        topn = sorted(zip(scores, clusterIndexes), reverse=True)[:n_clusters]

        #write to file
        out = self.sep.join(conceptNames)
        with open(os.path.join(self.results_subdir,out) + ".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for score, index in topn:
                cluster = clusters.loc[index,:]
                f.write("New cluster: " + self.nl)
                for _,event in cluster.iterrows():
                    f.write(str(event["title"]) + " [" + event["date"] + "] => ")
                f.write(self.nl)

    # compute dataset of concepts by date for heatmap visualisation
    def ConceptHeatmap(self,out=None):
        events = self.GetEvents()
        eventDates=events["date"]
        dateMin = str_to_date(eventDates.min())
        dateMax = str_to_date(eventDates.max())
        dateRange = pd.date_range(dateMin, dateMax).tolist()

        heatmap = {date.strftime('%Y-%m-%d'): defaultdict(int) for date in dateRange}
        for _,event in events.iterrows():
            date = event.date
            conceptsList = ast.literal_eval(event.concepts)
            for id,score in conceptsList:
                heatmap[date][id] += 1

        df = pd.DataFrame.from_dict(heatmap)
        if out is not None:
            df.to_csv(os.path.join(self.data_subdir,out)+".csv",sep=self.sep,na_rep=0,encoding=self.enc)
        return df

    def CsrFromVocab(self,vocab,events,concept_wgt=100,date_wgt=5000,include_date=False,verbose=False):
        if verbose:
            print("Making CSR Matrix from input vocab")
        indptr = [0]
        indices = []
        data = []
        for _,event in events.iterrows():
            conceptsList = ast.literal_eval(event.concepts)
            #build csr vectors
            for id,score in conceptsList:
                if id in vocab:
                    index = vocab[id]
                    indices.append(index)
                    score = normalize(score,0,100)
                    data.append(score*concept_wgt)

            #include date dimension
            if include_date:
                index = vocab["date"]
                indices.append(index)
                data.append(1*date_wgt)
            indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=int,shape=(len(events),len(vocab)))

    def TopConcepts(self,events,min_score):
        concepts=set()
        for _,event in events.iterrows():
            conceptList=ast.literal_eval(event.concepts)
            for id,score in conceptList:
                if score>=min_score:
                    concepts.add(id)

        return concepts



    ### Data Visualisation #########################################################################################################################

    # plot data
    def Plot(self, vals, type="bar", x_labels=None):
        if type.lower()=="bar":
            fig, ax = plt.subplots()
            ax.bar(range(len(vals)), vals, align="center")
            if x_labels is not None:
                ax.set_xticks(range(len(vals)))
                plt.xticks(rotation=90)
                ax.set_xticklabels(x_labels)
        elif type.lower()=="line":
            plt.plot(vals)
        plt.show()

    def ShowDendrogram(self,z,p=100):
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('sample index')
        plt.ylabel('distance')
        dendrogram(
            z,
            truncate_mode='lastp',  # show only the last p merged clusters
            p=p,  # show only the last p merged clusters
            show_leaf_counts=True,  # otherwise numbers in brackets are counts
            leaf_rotation=90.,
            leaf_font_size=6,
            show_contracted=False  # to get a distribution impression in truncated branches
        )
        plt.show()

    # show concept frequency line graph by date
    def ConceptFrequencyLineGraph(self,conceptNames,method=None):
        heatmap = self.GetHeatmap()
        conceptIds = self.ConceptNameToId(conceptNames)
        for id,name in zip(conceptIds,conceptNames):
            data = heatmap.loc[id,:]
            if method is not None:
                data = self.SmoothFunction(data,method).values
            plt.plot(data,label = name)
            data[data==0] = np.nan
            plt.axhline(np.nanmean(data))
        plt.ylabel("concept score frequency")
        plt.xlabel("days")
        plt.legend()

        plt.show()

    # show concept frequency heatmap by date
    def ConceptFrequencyHeatmap(self,n_days=100,n_concepts=50):
        heatmap = self.GetHeatmap()

        conceptCount = OrderedDict(self.GetConceptCount())
        conceptIds=[id for id,props in islice(reversed(conceptCount.items()),n_concepts)]
        conceptNames=[props[2] for id,props in islice(reversed(conceptCount.items()),n_concepts)]

        shown = np.empty((n_concepts, n_days))
        for i,id in enumerate(conceptIds):
            t = heatmap.loc[int(id),heatmap.columns[:n_days]].values
            normalizer = np.vectorize(normalize)
            t=normalizer(t,t.min(),t.max())
            shown[i] = t

        # map elements
        fig, ax = plt.subplots()
        map = ax.pcolor(shown)

        # put the major ticks at the middle of each cell
        ax.set_yticks(np.arange(shown.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(shown.shape[1]) + 0.5, minor=False)

        ax.invert_yaxis()

        # labels
        ax.set_yticklabels(conceptNames, minor=False)
        plt.xticks(rotation=90)
        ax.set_xticklabels(heatmap.columns[:n_days],minor=False)

        plt.show()
        #fig.savefig("heatmap.eps", format="eps", dpi=1000)
        #fig.savefig("heatmap.svg", format="svg", dpi=1200)

    def PlotBinomialDistribution(self):
        conceptCount = self.GetConceptCount()
        freqs=[]
        for id,props in conceptCount.items():
            freqs.append(props[1])

        self.Plot(freqs,type="line")

    def PlotEventDateRange(self):
        events=self.GetEvents()
        dateMin = str_to_date(events.date.min())
        dateMax = str_to_date(events.date.max())
        dateRange = pd.date_range(dateMin, dateMax).tolist()
        dates=pd.to_datetime(events.date)
        counts = []
        for date in dateRange:
            cond = (dates == date).values
            counts.append(sum(cond))

        self.Plot(counts,x_labels=dateRange)

    def PlotDistribution(self,y):
        size = len(y)
        x=range(size)
        h = plt.hist(y, bins="auto", color="w")

        dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']

        for dist_name in dist_names:
            dist = getattr(scipy.stats, dist_name)
            param = dist.fit(y)
            pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
            plt.plot(pdf_fitted, label=dist_name)
            plt.xlim(0, len(h)-2 )
        plt.legend(loc='upper right')
        plt.show()

    def PlotConceptTypes(self):
        events = self.GetEvents()
        concepts = self.GetConcepts(index_col=0)
        data = OrderedDict({"wiki":0,"loc":0,"org":0,"person":0})
        #for _,event in events.iterrows():
        #    cpts = string_to_object(event.concepts)
        #    for id,score in cpts:
        #        ctype = concepts.loc[id, :].type
        #        data[ctype] +=1

        for _,concept in concepts.iterrows():
            ctype = concept.type
            data[ctype] +=1

        self.Plot(list(data.values()),x_labels=list(data.keys()))



    ### Data Analysis #########################################################################################################################

    # compute correlations between concept frequencies
    def ComputeCorrelations(self,method=None):
        heatmap = pd.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, dtype=str,
                              index_col=0)  # read frequencies into dataframe

        correlations = {}
        for i,index in enumerate(heatmap.index):
            a = heatmap.loc[index, :].astype(float)
            for j,index2 in enumerate(heatmap.index):
                if j<=i:
                    continue
                b = heatmap.loc[index2, :].astype(float)
                if method is not None:
                    a = self.SmoothFunction(a)
                    b = self.SmoothFunction(b)
                correlations[(index,index2)]=self.Correlation(a,b)
            best = OrderedDict(sorted(correlations, key=lambda cpt: abs(correlations[cpt])))
            print(best)
            break

    # compute time series correlation between a and b series
    def Correlation(self,a,b):
        correlation = a.corr(b)
        return correlation

    # smooth data with rolling window
    def SmoothFunction(self,df,method="mean",window=3):
        roll =df.rolling(window=window,min_periods=1)
        if method.lower() == "mean".lower():
            return roll.mean()
        elif method.lower() == "median".lower():
            return roll.median()

    # compute concept frequency and order by method
    def CountConcepts(self,out=None,events=None,method="frequency",sort=True,desc=False,min_score=0,values_only=False,include_names=False,filter=None,n_concepts=0):
        if events is None:
            events = self.GetEvents()
        if include_names:
            idnamed=self.GetConceptIdToNameLookupDict()

        v=defaultdict(list)
        for i,event in events.iterrows():
            try:
                conceptsList = ast.literal_eval(event["concepts"])
            except TypeError as e:
                print(str(i))
                print(event["concepts"])
            for id,score in conceptsList:
                if score>=min_score and (filter is None or id in filter):
                    if len(v[id])==0:
                        v[id].append(0)
                        v[id].append(0)
                        if include_names:
                            v[id].append(idnamed[str(id)])
                    v[id][0]+=score
                    v[id][1]+=1

        if method.lower()=="score":
            if sort:
                v=OrderedDict(sorted(v.items(), key=lambda cpt: cpt[1][0],reverse=desc))
            if values_only:
                v = [e[0] for e in v.values()]
        elif method.lower()=="frequency":
            if sort:
                v=OrderedDict(sorted(v.items(), key=lambda cpt: cpt[1][1],reverse=desc))
            if values_only:
                v = [e[1] for e in v.values()]
        elif method.lower()=="average":
            if sort:
                v=OrderedDict(sorted(v.items(), key=lambda cpt: cpt[1][0]/cpt[1][1],reverse=desc))
            if values_only:
                v = [e[0]/e[1] for e in v.values()]

        if n_concepts > 0:
            if values_only:
                v[:n_concepts]
            else:
                v = OrderedDict(islice(v.items(), min(n_concepts, len(v))))

        if out is not None: save_as_json(v,out)
        return v



    ### Debugging ######################################################################################################################

    # find the ids of duplicates
    def FindDuplicates(self):
        #concepts = self.GetConcepts()
        #ids = concepts["conceptId"].astype(int)  # get just the concept ids

        #categories= self.GetCategories()
        #ids = categories["categoryId"].astype(int)

        events= self.GetEvents()
        ids = events["eventId"].astype(int)

        check=defaultdict(int)
        for id in ids:
            check[id]+=1

        for id, val in check.items():
            if val>1:
                print(id)
                print(val)

    # check the dataset for duplicates
    def TestUnique(self):
        events=self.GetConcepts()
        uniqueConceptIds = events["conceptId"].astype(int)

        all=len(uniqueConceptIds)
        unique=len(uniqueConceptIds.unique())

        print("Total items in dataset: "+str(all))
        print("Unique items in dataset: "+str(unique))
        return all==unique