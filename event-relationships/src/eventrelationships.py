from eventregistry import *
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.stats import threshold
from scipy.cluster.hierarchy import dendrogram, linkage
from src.helpers import *
from src.apriori import *
from datetime import date,timedelta
from collections import defaultdict
from itertools import islice
import json
import pandas as pd
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

    def __init__(self, events_filename,concepts_filename,categories_filename, connect=True,key=None, settings=None):
        self.events_filename = events_filename
        self.concepts_filename = concepts_filename
        self.categories_filename = categories_filename
        mkdir(self.results_subdir)
        mkdir(self.models_subdir)
        mkdir(self.data_subdir)
        if connect:
            self.er = EventRegistry()  # gets key from settings.json file in module root by default
            self.returnInfo=ReturnInfo(eventInfo=EventInfoFlags(socialScore=True)) # set return info



    ### Event Registry API #########################################################################################################################

    # returns a dictionary-like object and writes the json to file
    def GetEventsObj(self, categories=None, toFile=True,eventUris=None,dateStart=None,dateEnd=None):
        if categories!=None: # get events by category
            categoryUris = [self.er.getCategoryUri(cat) for cat in categories]
            q = QueryEvents(categoryUri=QueryItems.OR(categoryUris),dateStart=dateStart,dateEnd=dateEnd,lang="eng")
        elif eventUris!=None: # get events by event uris
            q = QueryEvents.initWithEventUriList(eventUris)

        q.setRequestedResult(RequestEventsInfo(returnInfo=self.returnInfo))
        res = self.er.execQuery(q)

        if (toFile):
            with open(os.path.join(self.data_subdir, self.events_filename)+".json", "w", encoding=self.enc) as f:
                json.dump(res, f, indent=2)

        return res

    # returns an iterator
    def GetEventsIter(self, categories):
        categoryURIs = [self.er.getCategoryUri(cat) for cat in categories]

        q = QueryEventsIter(categoryUri=QueryItems.OR(categoryURIs),lang="eng")
        res = q.execQuery(self.er,returnInfo=self.returnInfo)

        return res

    # returns ids of given concept names
    def ConceptNameToId(self,conceptNames):
        conceptUris = [self.er.getConceptUri(con) for con in conceptNames]
        concepts = self.GetConcepts()
        selected = pd.Series()
        for uri in conceptUris:
            selected = selected.append(concepts.loc[concepts["uri"] == uri], ignore_index=True)
        return selected["conceptId"].astype(int)

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



    ### Get Data #########################################################################################################################

    def GetEvents(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.events_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col) # read events into dataframe

    def GetConcepts(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.concepts_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col) # read concepts into dataframe

    def GetCategories(self,index_col=None):
        return pd.read_csv(os.path.join(self.data_subdir, self.categories_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str,index_col=index_col)  # read categories into dataframe

    def GetHeatmap(self):
        return pd.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0)  # read frequencies into dataframe

    def GetConceptCount(self):
        with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
            conceptCount = json.load(f)
        return conceptCount

    def GetClusters(self,labels,events=None,groupLabels=None,sorted=True):
        if events is None:
            events=self.GetEvents()
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



    ### Events CSV #########################################################################################################################

    def GenerateEventsCsv(self, eventIter):
        headers = ["eventId", "uri", "title","summary", "date", "location","socialScore","articleCount", "concepts","categories"]

        if file_exists(os.path.join(self.data_subdir, self.events_filename)+".csv"):
            self.UpdateEvents(eventIter)
        else:
            self.CreateEvents(eventIter, headers)
    def GenerateEventCsvLine(self, event):
        eventId=str(event["id"])
        uri = event["uri"]
        title = event["title"]
        title = title["eng"].replace(self.sep, ",") if "eng" in title else title[list(title.keys())[0]].replace(
            self.sep, ",")
        summary = event["summary"]
        summary= summary["eng"].replace(self.sep, ",") if "eng" in summary else summary[list(summary.keys())[0]].replace(
            self.sep, ",")
        summary = summary.replace(self.nl, " ")
        summary = summary.replace("\"","")
        date = event["eventDate"]
        location = json.dumps(event["location"])
        socialScore = str(event["socialScore"])
        articleCount = str(event["totalArticleCount"])
        concepts = ",".join(map(lambda c: "(" + str(c["id"]) + "," + str(c["score"]) + ")", event["concepts"])) if event[
            "concepts"] else "null"
        categories = ",".join(map(lambda c: "(" + str(c["id"]) + "," + str(c["wgt"]) + ")", event["categories"])) if event[
            "categories"] else "null"
        return (
            eventId + self.sep +
            uri + self.sep +
            title + self.sep +
            summary + self.sep +
            date + self.sep +
            location + self.sep +
            socialScore + self.sep +
            articleCount + self.sep +
            concepts + self.sep +
            categories + self.nl
        )
    def CreateEvents(self, eventIter, headers):
        with open(os.path.join(self.data_subdir, self.events_filename)+".csv", "w", encoding=self.enc, newline=self.nl) as f:
            f.write(self.sep.join(headers) + self.nl)
            for event in eventIter:
                try:
                    if "warning" not in event and event["concepts"]:
                        f.write(self.GenerateEventCsvLine(event))
                        self.GenerateConceptsCsv(event["concepts"])
                        self.GenerateCategoriesCsv(event["categories"])
                except KeyError as e:
                    print(json.dumps(event))
                    raise
    def UpdateEvents(self, eventIter):
        existingEvents = self.GetEvents()
        existingIds = existingEvents["eventId"].astype(int)
        existingIds.sort_values(inplace=True)  # sort the list for bisect
        existingIds = existingIds.tolist()

        with open(os.path.join(self.data_subdir, self.events_filename)+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for event in eventIter:
                try:
                    if 'warning' not in event and event["concepts"] and binsearch(existingIds,int(event["id"])) is False:
                        dd = self.GenerateEventCsvLine(event)
                        f.write(dd)
                        self.GenerateConceptsCsv(event["concepts"])
                        self.GenerateCategoriesCsv(event["categories"])
                except KeyError as e:
                    print(json.dumps(event))
                    raise
    #todo updates dataset with new columns but will not work with events older than 1 month
    def ExpandEvents(self):
        existingEvents = self.GetEvents()
        existingUris = existingEvents["uri"].tolist()

        events = self.GetEventsObj(eventUris=existingUris)

        with open(self.events_filename + "2.csv", "w", encoding=self.enc, newline=self.nl) as f:
            for event in events:
                try:
                    dd = self.GenerateEventCsvLine(event)
                    f.write(dd)
                    self.GenerateConceptsCsv(event["concepts"])
                    self.GenerateCategoriesCsv(event["categories"])
                except KeyError as e:
                    print(json.dumps(event))
                    raise
    # concatenates two event datasets
    def ConcatEvents(self):
        newEvents=pd.read_csv(os.path.join("old_data", self.events_filename) + ".csv", sep=self.sep, encoding=self.enc,
                              dtype=str)  # read events into dataframe
        existingEvents = self.GetEvents()
        existingIds = existingEvents["eventId"].astype(int)
        existingIds.sort_values(inplace=True)  # sort the list for bisect
        existingIds = existingIds.tolist()

        with open(os.path.join(self.data_subdir, self.events_filename)+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for _,event in newEvents.iterrows():
                try:
                    if binsearch(existingIds,int(event["eventId"])) is False:
                        eventId = str(event["eventId"])
                        uri = event["uri"]
                        title = event["title"]
                        if pd.isnull(title):
                            continue
                        summary = "null"
                        date = event["date"]
                        location = event["location"]
                        socialScore = "null"
                        articleCount = "null"
                        concepts = event["concepts"]
                        categories = "null"
                        f.write(eventId + self.sep +
                                uri + self.sep +
                                title + self.sep +
                                summary + self.sep +
                                date + self.sep +
                                location + self.sep +
                                socialScore + self.sep +
                                articleCount + self.sep +
                                concepts + self.sep +
                                categories + self.nl)
                except (KeyError,TypeError) as e:
                    print(event)
                    raise



    ### Concepts CSV #########################################################################################################################

    def GenerateConceptsCsv(self, concepts):
        headers = ["conceptId", "uri", "title", "type"]

        if file_exists(os.path.join(self.data_subdir, self.concepts_filename)+".csv"):
            self.UpdateConcepts(concepts)
        else:
            self.CreateConcepts(concepts, headers)
    def GenerateConceptCsvLine(self, concept):
        conceptId = str(concept["id"])
        uri = concept["uri"].replace(self.sep, ",")
        label = concept["label"]
        label = label["eng"].replace(self.sep, ",") if "eng" in label else label[list(label.keys())[0]].replace(self.sep, ",")
        type = concept["type"]
        return (
            conceptId + self.sep +
            uri + self.sep +
            label + self.sep +
            type + self.nl
        )
    def CreateConcepts(self, concepts, headers):
        with open(os.path.join(self.data_subdir, self.concepts_filename)+".csv", "w", encoding=self.enc, newline=self.nl) as f:
            f.write(self.sep.join(headers) + self.nl)
            for concept in concepts:
                try:
                    f.write(self.GenerateConceptCsvLine(concept))
                except KeyError as e:
                    print(json.dumps(concept))
                    raise
    def UpdateConcepts(self, concepts):
        existingConcepts = self.GetConcepts()
        existingIds = existingConcepts["conceptId"].astype(int)
        existingIds.sort_values(inplace=True)  # sort the list for bisect
        existingIds = existingIds.tolist()

        with open(os.path.join(self.data_subdir, self.concepts_filename)+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for concept in concepts:
                try:
                    if binsearch(existingIds,int(concept["id"])) is False:
                        f.write(self.GenerateConceptCsvLine(concept))
                except KeyError as e:
                    print(json.dumps(concept))
                    raise
    # concatenates two concept datasets
    def ConcatConcepts(self):
        newConcepts = pd.read_csv(os.path.join("old_data", self.concepts_filename) + ".csv",
                                  sep=self.sep, encoding=self.enc,
                                  dtype=str)  # read concepts into dataframe
        existingConcepts = self.GetConcepts()
        existingIds = existingConcepts["conceptId"].astype(int)
        existingIds.sort_values(inplace=True)  # sort the list for bisect
        existingIds = existingIds.tolist()

        with open(os.path.join(self.data_subdir, self.concepts_filename) + ".csv", "a", encoding=self.enc,
                  newline=self.nl) as f:
            for _, concept in newConcepts.iterrows():
                try:
                    if binsearch(existingIds, int(concept["conceptId"])) is False:
                        conceptId = str(concept["conceptId"])
                        uri = concept["uri"]
                        title = concept["title"]
                        type = concept["type"]
                        f.write(conceptId + self.sep +
                                uri + self.sep +
                                title + self.sep +
                                type + self.nl)
                except (KeyError, TypeError) as e:
                    print(concept)
                    raise



    ### Categories CSV #########################################################################################################################

    def GenerateCategoriesCsv(self, categories):
        headers = ["categoryId", "uri"]
        if file_exists(os.path.join(self.data_subdir, self.categories_filename) + ".csv"):
            self.UpdateCategories(categories)
        else:
            self.CreateCategories(categories, headers)
    def GenerateCategoryCsvLine(self, category):
        categoryId = str(category["id"])
        uri = category["uri"]
        return (
            categoryId + self.sep +
            uri + self.nl
        )
    def CreateCategories(self, categories, headers):
        with open(os.path.join(self.data_subdir, self.categories_filename) + ".csv", "w", encoding=self.enc, newline=self.nl) as f:
            f.write(self.sep.join(headers) + self.nl)
            for category in categories:
                try:
                    f.write(self.GenerateCategoryCsvLine(category))
                except KeyError as e:
                    print(json.dumps(category))
                    raise
    def UpdateCategories(self, categories):
        existingCategories = self.GetCategories()
        existingIds = existingCategories["categoryId"].astype(int)
        existingIds.sort_values(inplace=True)  # sort the list for bisect
        existingIds = existingIds.tolist()

        with open(os.path.join(self.data_subdir, self.categories_filename) + ".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for category in categories:
                try:
                    if binsearch(existingIds, int(category["id"])) is False:
                        f.write(self.GenerateCategoryCsvLine(category))
                except KeyError as e:
                    print(json.dumps(category))
                    raise



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
    def CsrMatrix(self,out=None,normalized=False,concept_wgt=1,include_date=False,date_wgt=1,date_max=None,min_events=5,events=None,verbose=False):
        print("Building CSR matrix")
        if events is None:
            events = self.GetEvents()
        eventsConcepts = events["concepts"]  # get just the event concepts
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
        excluded=0
        for i,eventConcepts in enumerate(eventsConcepts):
            conceptsList = ast.literal_eval(eventConcepts)  # list of tuples like (conceptId,score)
            included=False
            #build csr vectors
            for id,score in conceptsList:
                if min_events<2 or conceptCount[str(id)][1]>=min_events:
                    included=True
                    index = vocabulary.setdefault(id,len(vocabulary))
                    indices.append(index)
                    if normalized: score = normalize(score,0,100)
                    data.append(score*concept_wgt)

            #include date dimension
            if include_date and included:
                event_date=(str_to_date(eventDates[i])-dateMin).days
                index = vocabulary.setdefault("date",len(vocabulary))
                indices.append(index)
                if normalized: event_date = normalize(event_date, 0, dateDelta)
                data.append(event_date*date_wgt)
            #if included:
            indptr.append(len(indices))
            #else:
            #    excluded+=1

        matrix = csr_matrix((data, indices, indptr), dtype=int,shape=(len(indptr)-1,len(vocabulary)))
        if out is not None:
            save_sparse_csr(out,matrix)
            #save_as_json(vocabulary,os.path.join(self.data_subdir, out))
        if verbose:
            cc = ((len(indptr)-1)*len(vocabulary))/len(data) if len(indptr)-1>0 else 0
            print("n samples: " + str(len(indptr)-1))
            print("n dimensions: "+str(len(vocabulary)))
            print("n excluded: "+str(excluded))
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
    def DBSCAN(self,x,max_distance=0.5,min_samples=10,metric="euclidian",leaf_size=30):
        db = DBSCAN(eps=max_distance,min_samples=min_samples,metric=metric,leaf_size=leaf_size).fit(x)
        return (db,db.labels_)

    #k-means clustering
    def KMeans(self, x, n_clusters, nar=False,seed=None, init="k-means++", max_iter=300, n_init=10, n_jobs=1,verbose=0, useMiniBatchKMeans=False,out=None,dim_red=False):
        print("Training K-Means")
        if useMiniBatchKMeans:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                verbose=verbose,
                init_size=3*n_clusters) #to shut up the runtime warning
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
    def AgglomerativeClustering(self,x,n_clusters=2,affinity="euclidean",caching_dir=None,connectivity=None,full_tree="auto", method="ward",imp="sklearn",out=None):
        print("AgglomerativeClustering")
        if imp.lower()=="scipy":
            Z = linkage(x.toarray(),method=method,metric=affinity)
            #c, coph_dists = cophenet(Z, pdist(x))
            #print(c)
            #print(coph_dists)
            return Z
        elif imp.lower()=="sklearn":
            model = AgglomerativeClustering(
                n_clusters=n_clusters,
                affinity=affinity,
                memory=caching_dir,
                connectivity=connectivity,
                compute_full_tree=full_tree,
                linkage=method)
            model.fit(x.toarray())

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

    def TruncatedSVD(self,x,n_components = 2,algorithm = "randomized",n_iter = 5,seed=None,tol=0.0):
        svd = TruncatedSVD(n_components=n_components,algorithm=algorithm, n_iter=n_iter, random_state=42,tol=tol)
        svd.fit(x)
        print(svd.explained_variance_ratio_)
        print(svd.explained_variance_ratio_.sum())
        print(svd.singular_values_)

    def PCA(self,x,n_components=None, copy=True, whiten=False, svd_solver="auto", tol=0.0, iterated_power="auto", random_state=None,verbose=False):
        print("pca")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components, copy=copy, whiten=whiten, svd_solver=svd_solver, tol=tol, iterated_power=iterated_power, random_state=random_state)
        fitted=pca.fit_transform(x)
        if verbose:
            print(pca.explained_variance_ratio_)
            print(pca.singular_values_)
        return fitted,pca

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

    def DatasetByTimeWindow(self,clusters,window_size=14,min_events=50,min_score=10):
        print("Making dataset")

        for model,group in clusters.values():
            for cluster in group.groupby("cluster"):
                pass



    ### Cross validation ######################################################################################################################

    def CrossValidateByConcept(self,date_start="2017-09-01",date_end="2017-09-30",print_rel=False,min_score=30,n_clusters=1000,date_wgt=2500,seed=1):
        dateRange = pd.date_range(str_to_date(date_start), str_to_date(date_end)).tolist()
        scores = defaultdict(list)
        for date in dateRange:
            date = date.strftime('%Y-%m-%d')
            print(date)

            train,test = self.TrainTestSplit(date,out=False)

            trainMatrix,trainVocab=self.CsrMatrix(normalized=True,concept_wgt=100,include_date=True,date_wgt=date_wgt,events=train,min_events=5,date_max=date)
            trainModel,trainLabels = self.KMeans(trainMatrix,n_clusters,useMiniBatchKMeans=True,seed=seed)
            train["cluster"]=trainLabels

            testMatrix = self.CsrMatrix(normalized=True, concept_wgt=100, include_date=True, date_wgt=date_wgt, events=test,min_events=5)[0]
            testModel, testLabels = self.KMeans(testMatrix, n_clusters, useMiniBatchKMeans=True,seed=seed)
            test["cluster"]=testLabels

            today=test.loc[test.date==date]

            conceptIds=self.TopConcepts(today,min_score)
            dset=self.DatasetByConcept(events=train,conceptIds=conceptIds,min_events=2,out=True)
            #dset=None
            clfs=self.RandomForest(dset, keys=conceptIds, from_file=True)

            for conceptId in conceptIds:
                if clfs[conceptId] is not None:
                    print("Processing concept: "+str(conceptId))
                    xset = OrderedDict({cid: [] for cid in list(dset[conceptId][0].keys())})
                    yset = OrderedDict({cid: [] for cid in list(dset[conceptId][1].keys())})
                    for _,event in today.iterrows():
                        if HasConcept(event.concepts,conceptId):
                            testCluster=test.loc[test.cluster==event.cluster]
                            sampleCsr=self.CsrFromVocab(trainVocab,event,concept_wgt=100,date_wgt=date_wgt)
                            sampleCluster= trainModel.predict(sampleCsr)
                            trainCluster = train.loc[train.cluster==sampleCluster[0]]
                            self.Compile(xset,min_score,events=trainCluster)
                            self.Compile(yset,min_score,events=testCluster)

                            #set_attr = set(np.where(list(xset.values)()[i])[0])
                            #ids = [[xset.keys()][i] for i in set_pred]
                            #names = self.ConceptIdToName(ids)
                            #print("Attributes: " + names)
                    train_x = np.asarray(list(xset.values()))
                    test_y = np.asarray(list(yset.values()))

                    train_y=clfs[conceptId].predict(train_x.T)
                    score=self.HammingScore(test_y.T,train_y,verbose=True,cids_y=[yset.keys()])
                    print("Hamming score for " + str(conceptId) + " is " + str(score))
                    scores[conceptId].append(score) #todo

    def PredictByCluster(self,date_start="2017-09-04",date_end="2017-09-18",window_size=7,min_events=50,min_score=10):
        dateRange = pd.date_range(str_to_date(date_start), str_to_date(date_end)).tolist()

        for date in dateRange:
            print(date_to_str(date))
            train,test=self.TrainTestSplit(date_to_str(date))

            clusters=self.ClusterByTimeWindow(events=train,window_size=window_size,min_events=min_events)

            print("k")

    def CrossValidateByCluster(self,date_start="2017-09-04",date_end="2017-09-18",min_score=10,n_clusters=500,window_size=14,verbose=True,separate=False,min_events=100,dim_red=False,exclude_entities=True,knn_cluster_assignment=True,caching=False):
        dateRange = pd.date_range(str_to_date(date_start), str_to_date(date_end)).tolist()
        if separate:
            scores = defaultdict(list)
            constantScores = defaultdict(list)
        else:
            scores=[]
            constantScores=[]
        for date in dateRange:

            #split dset by date
            date = date.strftime('%Y-%m-%d')
            print(date)
            train,test = self.TrainTestSplit(date)

            #create matrix and train clustering
            trainMatrix,trainVocab=self.CsrMatrix(events=train,min_events=min_events,verbose=verbose)
            #trainMatrix, trainVocab = self.CsrMatrix(events=train,min_events=min_events,date_wgt=100,concept_wgt=100,normalized=True,include_date=True,verbose=verbose)
            if caching:
                trainModel=load_model("crossValCluster",enc=self.enc)
                trainLabels=trainModel.labels_
            else:
                #trainModel,trainLabels= self.KMeans(trainMatrix,n_clusters,useMiniBatchKMeans=False,nar=verbose,seed=1,verbose=False,out="crossValCluster")
                trainModel,trainLabels = self.AgglomerativeClustering(trainMatrix,n_clusters=n_clusters)
            train["cluster"]=trainLabels

            #performance info
            #silhuette(trainMatrix,trainLabels)
            self.ShowRelationships(trainLabels, "crossValCLustering",train)
            self.CountClusterSize(trainLabels,plot=False,events=train)

            #assign cluster to today's events
            today=test.loc[test.date==date]
            todayCsr=self.CsrFromVocab(trainVocab,today,verbose=verbose)
            if knn_cluster_assignment:
                knnModel = self.Clustering2KNN(trainLabels, trainMatrix)
                todayLabels=knnModel.predict(todayCsr)
            else:
                todayLabels= trainModel.predict(todayCsr)
            today.loc[:,"cluster"]=todayLabels
            todayClusters=set(todayLabels)

            keys=None
            if separate:
                keys=todayClusters
            if caching:
                dset=load_model("dset",self.enc)
            else:
                dset = self.DatasetByCluster(events=train, window_size=window_size,min_score=min_score,min_events=min_events, exclude_entities=exclude_entities,out="dset")
            if dim_red:
                clfs,pcaModel=self.RandomForest(dset, keys=todayClusters, from_file=False,nar=verbose,separate=separate,random_state=1,dim_red=dim_red)
            else:
                clfs=self.RandomForest(dset, keys=todayClusters, from_file=False,nar=verbose,separate=separate,random_state=1,dim_red=dim_red)
                #clfs=self.NeuralNetwork(dset,separate=separate,nar=verbose)
            constantClfs=self.MultiLabelMostFrequent(train,keys=todayClusters,verbose=verbose,from_file=False,min_score=min_score,separate=separate,n_labels=10,filter=list(dset[1].keys()))

            if not separate:
                xset = OrderedDict({cid: [] for cid in list(dset[0].keys())})
                yset = OrderedDict({cid: [] for cid in list(dset[1].keys())})
            for todayCluster in todayClusters:
                if not separate or clfs[todayCluster] is not None and constantClfs[todayCluster] is not None:
                    trainCluster=train.loc[train.cluster==todayCluster]
                    group=today.loc[today.cluster==todayCluster]
                    if separate:
                        xset = OrderedDict({cid: [] for cid in list(dset[todayCluster][0].keys())})
                        yset = OrderedDict({cid: [] for cid in list(dset[todayCluster][1].keys())})

                    startDate = (str_to_date(date) - timedelta(days=window_size)).strftime('%Y-%m-%d')
                    cond = np.logical_and(trainCluster.date>=startDate, trainCluster.date<date)
                    self.Compile(xset, min_score,events=trainCluster.loc[cond],exclude_entities=False)
                    self.Compile(yset, min_score,events=group,exclude_entities=False)

                    if separate:
                        train_x = np.asarray(list(xset.values()))
                        test_y = np.asarray(list(yset.values()))

                        train_y=clfs[todayCluster].predict(train_x.T)
                        constant_train_y=self.MLMFPredict(constantClfs[todayCluster],yset)
                        score=self.HammingScore(test_y.T,train_y,verbose=verbose,xset=xset,yset=yset,separate=separate)
                        constantScore = self.HammingScore(test_y.T,constant_train_y,verbose=False,xset=xset,yset=yset,separate=separate)
                        if verbose:
                            print("Hamming score for " + str(todayCluster) + " is " + str(score))
                            print("Constant score for " + str(todayCluster) + " is " + str(constantScore))
                        scores[todayCluster].append(score)
                        constantScores[todayCluster].append(constantScore)

            if not separate:
                train_x = np.asarray(list(xset.values()))
                test_y = np.asarray(list(yset.values()))

                if dim_red:
                    train_x=pcaModel.transform(train_x.T)
                    train_y = clfs.predict(train_x)
                else:
                    #train_y = bin_encode(clfs.predict(train_x.T),limit=0.05)
                    train_y = clfs.predict(train_x.T)
                constant_train_y = self.MLMFPredict(constantClfs, yset)
                score = self.HammingScore(test_y.T, train_y, verbose=verbose, xset=xset, yset=yset,separate=separate)
                constantScore = self.HammingScore(test_y.T, constant_train_y, verbose=verbose, xset=xset, yset=yset,separate=separate)
                if verbose:
                    print("Hamming score for " + date + " is " + str(score))
                    print("Constant score for " + date + " is " + str(constantScore))
                scores.append(score)
                constantScores.append(constantScore)
            break
        if separate:
            for key in scores.keys():
                print("Mean Hamming Score for cluster "+str(key)+" x"+str(len(scores[key]))+" is "+str(np.mean(scores[key])))
                print("Mean constant score for cluster "+str(key)+" is "+str(np.mean(constantScores[key]))+self.nl)
        else:
            print("Mean Hamming Score: " +str(np.mean(scores)))
            print("Mean constant score: "+ str(np.mean(constantScores)) + self.nl)

    ### Multi-label prediction ######################################################################################################################

    def DatasetByClusterOld(self,labels=None,events=None,keys=None,min_events=1,min_score=50,out=False,window_size=7,verbose=True,separate=True,sliding=True,exclude_entities=False):
        print("Making dataset")
        if min_events>1:
            conceptCount = self.GetConceptCount()
        else:
            conceptCount=None
        if events is None:
            events=self.GetEvents()
            events["cluster"]=labels
        if keys is None:
            keys=set(events.cluster.values)
        if exclude_entities:
            concepts=self.GetConcepts(index_col=0)
        else:
            concepts=None
        if separate:
            dset = {}
        else:
            xset = OrderedDict()
            yset = OrderedDict()
            samples = 0
        for key in keys:
            if verbose and separate:
                print(str(key))
            if separate:
                xset=OrderedDict()
                yset=OrderedDict()
                samples = 0

            cluster = events.loc[events.cluster==key]
            dateMin = str_to_date(cluster.date.min())
            dateMax = str_to_date(cluster.date.max())
            startDate = dateMin
            endDate = (dateMin + timedelta(days=window_size))
            dates = pd.to_datetime(cluster.date)

            cluster.sort_values("date", inplace=True)

            while(endDate<=dateMax):
                xCond = np.logical_and(dates >= startDate, dates < endDate)
                xWindow = cluster.loc[xCond]
                yCond = dates == endDate
                yWindow = cluster.loc[yCond]

                self.Compile(xset, min_score, samples=samples, events=xWindow,min_events=min_events,conceptCount=conceptCount,concept_info=concepts,exclude_entities=exclude_entities)
                self.Compile(yset, min_score, samples=samples, events=yWindow,min_events=min_events,conceptCount=conceptCount,concept_info=concepts,exclude_entities=exclude_entities)

                samples+=1
                if sliding:
                    startDate = (startDate + timedelta(days=1))
                    endDate = (endDate + timedelta(days=1))
                else:
                    startDate = endDate
                    endDate = (endDate + timedelta(days=window_size))
            if separate:
                x = np.asarray(list(xset.values()), dtype=int)
                y = np.asarray(list(yset.values()), dtype=int)
                if verbose:
                    print("n features: " + str(x.shape[0]))
                    print("n classes: " + str(y.shape[0]))
                    print("n samples: " + str(x.shape[1]) if len(x.shape)>1 else 1)
                if out and x.size > 0 and y.size > 0:
                    np.savetxt(os.path.join(self.data_subdir, str(key)) + "_x.txt", x.T, delimiter=self.sep, header=self.sep.join(map(str, list(xset.keys()))), fmt="%d")
                    np.savetxt(os.path.join(self.data_subdir, str(key)) + "_y.txt", y.T, delimiter=self.sep, header=self.sep.join(map(str, list(yset.keys()))), fmt="%d")
                dset[key] = (xset, yset)
        if not separate:
            x = np.asarray(list(xset.values()), dtype=int)
            y = np.asarray(list(yset.values()), dtype=int)
            if verbose:
                print("n features: " + str(x.shape[0]))
                print("n classes: " + str(y.shape[0]))
                print("n samples: " + str(x.shape[1]) if len(x.shape) > 1 else 1)
            if out:
                save_model((xset,yset), out)
                np.savetxt(os.path.join(self.data_subdir,"_x") + ".txt", x.T, delimiter=self.sep,header=self.sep.join(map(str, list(xset.keys()))), fmt="%d")
                np.savetxt(os.path.join(self.data_subdir,"_y") + ".txt", y.T, delimiter=self.sep,header=self.sep.join(map(str, list(yset.keys()))), fmt="%d")
            return xset,yset
        else:
            return dset

    def DatasetByCluster(self, labels=None, events=None, min_events=1, min_score=50, out=None,window_size=7,sliding=True, exclude_entities=True):
        print("Making dataset")
        if min_events > 1:
            conceptCount = self.GetConceptCount()
        else:
            conceptCount = None
        if events is None:
            events = self.GetEvents()
            events["cluster"] = labels
        if exclude_entities:
            concepts = self.GetConcepts(index_col=0)
        else:
            concepts = None
        xset = OrderedDict()
        yset = OrderedDict()
        samples = 0
        for _,cluster in events.groupby("cluster"):
            dateMin = str_to_date(cluster.date.min())
            dateMax = str_to_date(cluster.date.max())
            startDate = dateMin
            endDate = (dateMin + timedelta(days=window_size))
            dates = pd.to_datetime(cluster.date)

            cluster.sort_values("date", inplace=True)

            while (endDate <= dateMax):
                xCond = np.logical_and(dates >= startDate, dates < endDate)
                xWindow = cluster.loc[xCond]
                yCond = dates == endDate
                yWindow = cluster.loc[yCond]

                self.Compile(xset, min_score, samples=samples, events=xWindow, min_events=min_events,
                             conceptCount=conceptCount, concept_info=concepts, exclude_entities=exclude_entities)
                self.Compile(yset, min_score, samples=samples, events=yWindow, min_events=min_events,
                             conceptCount=conceptCount, concept_info=concepts, exclude_entities=exclude_entities)

                samples += 1
                if sliding:
                    startDate = (startDate + timedelta(days=1))
                    endDate = (endDate + timedelta(days=1))
                else:
                    startDate = endDate
                    endDate = (endDate + timedelta(days=window_size))
        x = np.asarray(list(xset.values()), dtype=int)
        y = np.asarray(list(yset.values()), dtype=int)
        print("n features: " + str(x.shape[0]))
        print("n classes: " + str(y.shape[0]))
        print("n samples: " + str(x.shape[1]) if len(x.shape) > 1 else 1)
        if out is not None:
            save_model((xset, yset), out)
            np.savetxt(os.path.join(self.data_subdir, "_x") + ".txt", x.T, delimiter=self.sep,
                       header=self.sep.join(map(str, list(xset.keys()))), fmt="%d")
            np.savetxt(os.path.join(self.data_subdir, "_y") + ".txt", y.T, delimiter=self.sep,
                       header=self.sep.join(map(str, list(yset.keys()))), fmt="%d")
        return xset, yset

    # compile dataset for prediction
    def DatasetByConcept(self,labels=None,events=None,conceptNames=None,conceptIds=None,min_events=10,min_score=50,out=False):
        print("Making dataset")
        if events is None:
            events=self.GetEvents()
            events["cluster"]=labels

        if conceptIds is None:
            conceptIds=self.ConceptNameToId(conceptNames)

        dset={}
        for conceptId in conceptIds:
            print(str(conceptId))
            xset=OrderedDict()
            yset=OrderedDict()
            samples=0
            for _, cluster in events.groupby("cluster"):
                for _,event in cluster.iterrows():
                    concepts = ast.literal_eval(event.concepts)
                    for id, score in concepts:
                        if id == conceptId and score >= min_score:
                            prev=cluster.loc[cluster.date<event.date]
                            fol=cluster.loc[cluster.date>event.date]
                            if len(prev)>=min_events and len(fol)>=min_events:
                                self.Compile(xset,min_score,samples=samples,events=prev)
                                self.Compile(yset,min_score,samples=samples,events=fol)
                                samples+=1
            x=np.asarray(list(xset.values()),dtype=int)
            y=np.asarray(list(yset.values()),dtype=int)
            if out and x.size>0 and y.size>0:
                print("has data")
                np.savetxt(os.path.join(self.data_subdir, str(conceptId)) + "_x.txt",x.T,delimiter=self.sep,header=self.sep.join(map(str,list(xset.keys()))),fmt="%d")
                np.savetxt(os.path.join(self.data_subdir, str(conceptId)) + "_y.txt",y.T,delimiter=self.sep,header=self.sep.join(map(str,list(yset.keys()))),fmt="%d")
            dset[conceptId]=(xset,yset)
        return dset

    def Compile(self,dset,min_score,events=None,concepts=None,samples=None,min_events=1,conceptCount=None,concept_info=None,exclude_entities=True):
        if exclude_entities:
            excluded_types=["loc","person","org"]
        used=set(dset.keys())

        excluded=0
        if events is not None:
            for _,event in events.iterrows():
                conceptList = ast.literal_eval(event.concepts)
                included=False
                for id,score in conceptList:
                    if score >= min_score:
                        if (not exclude_entities or concept_info.loc[id,:].type not in excluded_types) and (min_events<2 or conceptCount[str(id)][1]>=min_events):
                            if id in used:
                                dset[id].append(1)
                                used.remove(id)
                                included=True
                            elif samples is not None:
                                dset[id]=[0 for _ in range(samples)]
                                dset[id].append(1)
                                included = True
                    else:
                        break
                if not included:
                    excluded+=1
        else:
            included=False
            conceptList = ast.literal_eval(concepts)
            for id, score in conceptList:
                if score >= min_score:
                    if (not exclude_entities or concept_info.loc[id,:].type not in excluded_types) and (min_events < 2 or conceptCount[str(id)][1] >= min_events):
                        if id in used:
                            dset[id].append(1)
                            used.remove(id)
                            included = True
                        elif samples is not None:
                            dset[id] = [0 for _ in range(samples)]
                            dset[id].append(1)
                            included = True
                else:
                    break
            if not included:
                excluded+=1

        for id in used:
            dset[id].append(0)

        #print("n excluded: "+str(excluded))

    # random forest prediction
    def RandomForest(self, dset,separate=True,conceptNames=None, keys=None,nar=True, from_file=False, n_estimators=10, criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features="auto", max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None,dim_red=False,dim_red_model=None):
        from sklearn.ensemble import RandomForestClassifier
        print("Training random forest")

        if separate:
            if keys is None:
                keys = self.ConceptNameToId(conceptNames)
            clfs={}
            for id in keys:
                if nar:
                    print(id)
                if from_file and file_exists(os.path.join(self.data_subdir, str(id)) + "_x.txt"):
                    x = np.genfromtxt(os.path.join(self.data_subdir, str(id)) + "_x.txt",delimiter=self.sep,skip_header=1,dtype=int)
                else:
                    x = np.asarray(list(dset[id][0].values()), dtype=int).T
                if x.shape[0]==0 or len(x.shape)<2:
                    clfs[id]=None
                    continue
                if from_file and file_exists(os.path.join(self.data_subdir, str(id)) + "_y.txt"):
                    y = np.genfromtxt(os.path.join(self.data_subdir, str(id)) + "_y.txt", delimiter=self.sep,skip_header=1, dtype=int)
                else:
                    y = np.asarray(list(dset[id][1].values()), dtype=int).T
                if y.shape[0]==0 or len(y.shape)<2:
                    clfs[id]=None
                    continue
                model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start, class_weight=class_weight)

                model.fit(x,y)
                clfs[id]=model
            return clfs
        else:
            if from_file and file_exists(os.path.join(self.data_subdir, "_x") + ".txt"):
                x = np.genfromtxt(os.path.join(self.data_subdir, "_x") + ".txt", delimiter=self.sep, skip_header=1,dtype=int)
            else:
                x = np.asarray(list(dset[0].values()), dtype=int).T
            if from_file and file_exists(os.path.join(self.data_subdir, "_y") + ".txt"):
                y = np.genfromtxt(os.path.join(self.data_subdir, "_y") + ".txt", delimiter=self.sep, skip_header=1,dtype=int)
            else:
                y = np.asarray(list(dset[1].values()), dtype=int).T
            model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth,
                                           min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                           min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
                                           max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease,
                                           min_impurity_split=min_impurity_split, bootstrap=bootstrap,
                                           oob_score=oob_score, n_jobs=n_jobs, random_state=random_state,
                                           verbose=verbose, warm_start=warm_start, class_weight=class_weight)
            if dim_red:
                if dim_red_model is None:
                    reduced,pcaModel = self.PCA(x,n_components=100)
                else:
                    reduced = dim_red_model.transform(x)
                model.fit(reduced, y)
                return model,pcaModel
            else:
                model.fit(x, y)
                return model

    def MultiLabelMostFrequent(self, events, n_labels=10,keys=None,conceptNames=None,verbose=True,from_file=False,min_score=50,separate=True,filter=None):
        if verbose:
            print("Training most frequent classfier")
        if separate:
            if keys is None:
                keys = self.ConceptNameToId(conceptNames)
            clfs={}
            for id in keys:
                cluster=events.loc[events.cluster==id]
                if verbose:
                    print(id)

                freq = self.CountConcepts(events=cluster,desc=True,min_score=min_score,filter=filter)
                model=list(freq.keys())[:n_labels]
                clfs[id]=model
            return clfs
        else:
            freq=self.CountConcepts(events=events,desc=True,min_score=min_score,filter=filter)
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

    def NeuralNetwork(self,dset,separate=False,nar=True):
        from keras.models import Sequential
        from keras.layers import Dense
        print("Training neural network")

        x = np.asarray(list(dset[0].values()), dtype=int).T
        y = np.asarray(list(dset[1].values()), dtype=int).T
        print(x.shape)
        nn = Sequential()
        nn.add(Dense(x.shape[1], activation="relu", input_shape=(x.shape[1],)))
        nn.add(Dense(y.shape[1], activation="sigmoid"))
        nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        nn.fit(x, y, batch_size=16, epochs=5,verbose=0, validation_split=0.1)

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
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(np.where(y_true[i])[0])
            set_pred = set(np.where(y_pred[i])[0])

            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                if verbose:
                    if separate:
                        print("N Predicted: "+str(len(set_pred)))
                        print("N True: "+str(len(set_true)))
                        print("N Correctly predicted: "+str(len(set_true.intersection(set_pred))))
                    else:
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

    #optimize a clustering algorithm using silhouette score
    def Cluster(self, x, min=1, max=15, exp=2, method="Kmeans", seed=None, max_iter=30, n_init=3, n_jobs=1,verbose=1,plot=True):
        bestClusters=0
        bestScore=-1
        bestModel=None
        print("Algorithm: "+method)
        y = []
        n_clusters_x = []
        for i in range(min, max,exp):
            #n_clusters=2**i
            n_clusters=i
            if method.lower() == "KMeans".lower():
                model,labels,score=self.KMeans(x,n_clusters,seed=seed, max_iter=max_iter,n_init=n_init,n_jobs=n_jobs,verbose=verbose)
            elif method.lower() =="MiniBatchKMeans".lower():
                model,labels,score=self.KMeans(x,n_clusters,seed=seed, max_iter=max_iter,n_init=n_init,n_jobs=n_jobs,useMiniBatchKMeans=True,verbose=verbose)
            elif method.lower() == "NMF".lower():
                model,labels=self.NMF(x,n_clusters)
            elif method.lower() == 'Hierarichal'.lower():
                model,labels=self.AgglomerativeClustering(x,n_clusters,caching_dir="hierarichal_caching_dir")
            score=silhuette(x,labels)
            y.append(score)
            n_clusters_x.append(n_clusters)
            print("n-clusters: "+str(n_clusters))
            if score>bestScore:
                bestScore=score
                bestClusters=n_clusters
                bestModel = model

        print("Best score: "+str(bestScore) + " for n-clusters: "+str(bestClusters))
        save_model(bestModel,method)

        if plot:
            plt.plot(y, n_clusters_x)
            plt.ylabel("score")
            plt.xlabel("n_clusters")
            plt.show()

    # write all the clusters to a file in a human readable format
    def ShowRelationships(self,labels,out,events):
        with open(os.path.join(self.results_subdir,out)+".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for _,cluster in events.groupby("cluster"):
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
    def ConceptHeatmap(self,limit=0,out=None,min_events=5):
        events = self.GetEvents()
        eventDates=events["date"]
        dateMin = str_to_date(eventDates.min())
        dateMax = str_to_date(eventDates.max())

        dateRange = pd.date_range(dateMin, dateMax).tolist()
        conceptCount = self.CountConcepts()

        heatmap = {date.strftime('%Y-%m-%d'): defaultdict(int) for date in dateRange}
        for _,event in events.iterrows():
            date = event["date"]
            conceptsList = ast.literal_eval(event["concepts"])
            for id,score in conceptsList:
                if conceptCount[id][1] >= min_events:
                    heatmap[date][id] += score if score>=limit else 0

        df = pd.DataFrame.from_dict(heatmap)
        if out is not None:
            df.to_csv(os.path.join(self.data_subdir,out)+".csv",sep=self.sep,na_rep=0,encoding=self.enc)
        return df

    # compute set of concepts used in the model
    def DatasetConcepts(self,min_events=5):
        with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
            conceptCount = json.load(f)

        events=self.GetEvents()
        concepts=OrderedDict()
        for _,event in events.iterrows():
            conceptList = ast.literal_eval(event.concepts)
            for id,score in conceptList:
                if conceptCount[str(id)][1]>=min_events:
                    concepts[str(id)]="0"

        return concepts

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
        elif type.lower()=="dendrogram":
            dendrogram(vals  #,
                       #leaf_rotation=90.,  # rotates the x axis labels
                       #leaf_font_size=8.,  # font size for the x axis labels
                       )
        plt.show()

    # show concept frequency line graph by date
    def ConceptFrequencyLineGraph(self,conceptNames,method=None):
        heatmap = pd.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0)  # read frequencies into dataframe
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
    def ConceptFrequencyHeatmap(self,conceptNames=None,n_days=100,n_concepts=50):
        heatmap = self.GetHeatmap()
        if n_concepts>0:
            shown = np.empty((n_concepts, n_days))
            with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
                conceptCount = OrderedDict(json.load(f))
            conceptIds=[c[0] for c in islice(reversed(conceptCount.items()),n_concepts)]
            conceptNames=[c[1][2] for c in islice(reversed(conceptCount.items()),n_concepts)]
            for i,id in enumerate(conceptIds):
                #t = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
                t = heatmap.loc[int(id),heatmap.columns[:n_days]].values
                normalizer = np.vectorize(normalize)
                t=normalizer(t,t.min(),t.max())
                shown[i] = t


        elif conceptNames is not None:
            shown= np.empty((len(conceptNames), n_days))
            conceptIds = self.ConceptNameToId(conceptNames)
            for i,id in enumerate(conceptIds):
                t = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
                shown[i] = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
        else: shown = heatmap.values

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
        v=defaultdict(list)
        if include_names:
            idnamed=self.GetConceptIdToNameLookupDict()
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
        if out is not None:
            with open(os.path.join(self.data_subdir, out) + ".json", "w", encoding=self.enc) as f:
                json.dump(v, f, indent=2)

        if n_concepts > 0:
            if values_only:
                v[:n_concepts]
            else:
                v = OrderedDict(islice(v.items(), min(n_concepts, len(v))))

        return v



    ### Debugging ######################################################################################################################

    # find the ids of duplicates
    def FindDuplicates(self):
        concepts = self.GetConcepts()
        uniqueConceptIds = concepts["conceptId"].astype(int)  # get just the concept ids

        check=dict()
        for id in uniqueConceptIds:
            dups = check.get(id,0)+1
            check[id]=dups
            if dups>1:
                print(id)

    # check the dataset for duplicates
    def TestUnique(self):
        events=self.GetConcepts()
        uniqueConceptIds = events["conceptId"].astype(int)

        all=len(uniqueConceptIds)
        unique=len(uniqueConceptIds.unique())

        print("Total items in dataset: "+str(all))
        print("Unique items in dataset: "+str(unique))
        return all==unique