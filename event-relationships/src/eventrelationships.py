from eventregistry import *
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import apriori
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from scipy.sparse import csr_matrix
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import dendrogram, linkage
from src.helpers import *
from datetime import date,timedelta
from collections import defaultdict
from itertools import islice
import json
import pandas
import numpy
import logging
import time
import math
import ast
import gc
import pickle
import os.path
import matplotlib.pyplot as plt

class EventRelationships:
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
        selected = pandas.Series()
        for uri in conceptUris:
            selected = selected.append(concepts.loc[concepts["uri"] == uri], ignore_index=True)
        return selected["conceptId"].astype(int)



    ### Get Data #########################################################################################################################

    def GetEvents(self):
        return pandas.read_csv(os.path.join(self.data_subdir, self.events_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str) # read events into dataframe

    def GetConcepts(self):
        return pandas.read_csv(os.path.join(self.data_subdir, self.concepts_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str) # read concepts into dataframe

    def GetCategories(self):
        return pandas.read_csv(os.path.join(self.data_subdir, self.categories_filename) + ".csv", sep=self.sep, encoding=self.enc, dtype=str)  # read categories into dataframe

    def GetHeatmap(self):
        return pandas.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0)  # read frequencies into dataframe



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
        newEvents=pandas.read_csv(os.path.join("old_data", self.events_filename) + ".csv", sep=self.sep, encoding=self.enc,
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
                        if pandas.isnull(title):
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
        newConcepts = pandas.read_csv(os.path.join("old_data", self.concepts_filename) + ".csv",
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
            bins = numpy.zeros(uniqueConceptIds.size,dtype=int)
            for id, score in conceptsList:
                index = binsearch(uniqueConceptIds.tolist(), int(id))
                if normalized: score = normalize(score, 0, 100)
                bins[index] = score
            matrix.append(bins)
        return matrix

    #compute csr matrix from data
    def CsrMatrix(self,out=None,normalized=False,concept_wgt=1,include_date=False,date_wgt=1,date_max=None,min_events=5,events=None):
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
            indptr.append(len(indices))
            if not included: excluded+=1

        matrix = csr_matrix((data, indices, indptr), dtype=int,shape=(len(eventsConcepts),len(vocabulary)))
        if out is not None:
            save_sparse_csr(out,matrix)
            #save_as_json(vocabulary,os.path.join(self.data_subdir, out))

        print("n dimensions: "+str(len(vocabulary)))
        print("n excluded: "+str(excluded))
        return (matrix,vocabulary)



    ### Clustering Algorithms ######################################################################################################################

    # nmf clustering
    def NMF(self,x,nClusters,init="nndsvd",seed=None,maxiter=100):
        model = NMF(n_components=nClusters, init=init, random_state=seed, max_iter=maxiter)
        W = model.fit_transform(x)
        labels=[]
        for sample in W:
            labels.append(numpy.argmax(sample))
        #H = model.components_
        return (model,labels)

    # dbscan clustering
    def DBSCAN(self,x,max_distance=0.5,min_samples=10):
        db = DBSCAN(eps=max_distance,min_samples=min_samples).fit(x)
        return (db,db.labels_)

    #k-means clustering
    def KMeans(self, x, n_clusters, seed=None, init="k-means++", max_iter=300, n_init=10, n_jobs=1,verbose=0, useMiniBatchKMeans=False,out=None):
        if useMiniBatchKMeans:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed,
                verbose=verbose)
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
    def AffinityPropagation(self, x, damping=0.5,max_iter=200,convergence_iter=15):
        affinity = AffinityPropagation(
            damping=damping,
            max_iter=max_iter,
            convergence_iter=convergence_iter)
        affinity = affinity.fit(x)
        return (affinity, affinity.labels_)

    # agglomerative (hierarichal) clustering
    def AgglomerativeClustering(self,x,n_clusters=2,affinity="euclidean",caching_dir=None,connectivity=None,full_tree="auto", method="ward",imp="sklearn"):
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
            model.fit(x)
            return (model,model.labels_)



    ### Dimensionality Reduction ######################################################################################################################

    def TruncatedSVD(self,x,n_components = 2,algorithm = "randomized",n_iter = 5,seed=None,tol=0.0):
        svd = TruncatedSVD(n_components=n_components,algorithm=algorithm, n_iter=n_iter, random_state=42,tol=tol)
        svd.fit(x)
        print(svd.explained_variance_ratio_)
        print(svd.explained_variance_ratio_.sum())
        print(svd.singular_values_)



    ### Multi-label prediction ######################################################################################################################

    # random forest prediction
    def RandomForest(self,conceptNames=None,conceptIds=None):
        from sklearn.ensemble import RandomForestClassifier
        if conceptIds is None:
            conceptIds = self.ConceptNameToId(conceptNames)

        clfs={}
        for id in conceptIds:
            print(id)
            x = numpy.genfromtxt(os.path.join(self.data_subdir, str(id)) + "_x.txt",delimiter=self.sep,skip_header=1,dtype=int)
            if x.shape[0]<2 or len(x.shape)<2:
                clfs[id]=None
                continue
            y = numpy.genfromtxt(os.path.join(self.data_subdir, str(id)) + "_y.txt",delimiter=self.sep,skip_header=1,dtype=int)
            if y.shape[0]<2 or len(y.shape)<2:
                clfs[id]=None
                continue
            model = RandomForestClassifier()
            model.fit(x,y)
            clfs[id]=model
        return clfs

    # apriori association rule prediction
    def Apriori(self, support=0.01, minlen=1):
        data = pandas.read_csv(os.path.join(self.data_subdir, "AssociationRules") + ".csv", sep=self.sep,
                               encoding=self.enc, index_col=0)
        frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)
        print(frequent_itemsets)



    ### Outlier Detection ######################################################################################################################

    def IsolationForest(self,x,n_estimators=100,max_samples="auto",contamination=0.1,max_features=1.0,bootstrap=False,n_jobs=1,seed=None,verbose=0):
        model = IsolationForest()
        model.fit(x)
        return model.predict(x)



    ### Validation metrics ######################################################################################################################

    def HammingScore(self,y_true, y_pred):
        '''
        Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
        https://stackoverflow.com/q/32239577/395857
        '''
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set(numpy.where(y_true[i])[0])
            set_pred = set(numpy.where(y_pred[i])[0])
            #print('\nset_true: {0}'.format(set_true))
            #print('set_pred: {0}'.format(set_pred))
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
            #print('tmp_a: {0}'.format(tmp_a))
            acc_list.append(tmp_a)
        return numpy.mean(acc_list)

    ### Machine Learning ######################################################################################################################

    # show cluster shapes
    def CountClusterSize(self,labels):
        cs = dict.fromkeys(set(labels),0)
        for label in labels:
            cs[label]+=1
        self.Plot(list(cs.values()))

    # compute top concepts for each cluster
    def GetClusterConcepts(self,labels,n_concepts=20,method="frequency",out=None):
        clusters = self.GetClusters(labels)
        df=pandas.DataFrame(columns=["cluster","concepts"])
        incoming = []
        for label,cluster in clusters.groupby(level=0):
            print(label)
            count = self.CountConcepts(events=cluster,method=method,desc=True)
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
    def TrainTestSplit(self,date,out=True):
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
    def ShowRelationships(self,labels,out):
        clusters = self.GetClusters(labels)

        with open(os.path.join(self.results_subdir,out)+".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for _,cluster in clusters.groupby(level=0):
                f.write("New cluster: "+self.nl)
                for _,event in cluster.iterrows():
                    f.write(str(event["title"])+" ["+event["date"]+"] => ")
                f.write(self.nl)

    # same as above, but with clusters of clusters
    def ShowClusterRelationships(self,labels,clusterLabels,out):
        clusters=self.GetClusters(labels,clusterLabels=clusterLabels,avgDate=True)

        tab ="\t"
        with open(os.path.join(self.results_subdir, out) + ".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for _, group in clusters.groupby(level=0):
                f.write("New group: "+self.nl)
                for _,cluster in group.groupby("avgDate"):
                    f.write(tab+"New cluster: " + self.nl+tab+tab)
                    for _, event in cluster.iterrows():
                        f.write(str(event["title"]) + " [" + event["date"] + "] => ")
                    f.write(self.nl)

    # adjust the cluster centers according to the event with the highest social score in each cluster and recluster
    def AdjustCentroids(self,x,labels,method="KMeans"):
        clusters=self.GetClusters(labels)

        # find event with the highest social score in each cluster
        centers = pandas.DataFrame()
        for _,cluster in clusters.groupby(level=0):
            best=cluster.loc[cluster["socialScore"].idxmax(),:]
            centers=centers.append(best,ignore_index=True)

        centers = numpy.asarray(self.SparseMatrix(events=centers))

        if method.lower() == "KMeans".lower():
            model,labels=self.KMeans(x, len(set(labels)),init=centers,n_init=1)
        return (model,labels)

    # compile the clustered events in a multiindex dataframe
    def GetClusters(self,labels,sorted=True,events=None,clusterLabels=None,avgDate=False):
        if events is None:
            events=self.GetEvents()
            events["cluster"]=labels
        if clusterLabels is not None:
            clusterConcepts = pandas.read_csv(os.path.join(self.data_subdir, "KmeansClusterConcepts") + ".csv",
                                              sep=self.sep, encoding=self.enc)
            clusterConcepts["group"] = clusterLabels

            clusters = {}

            for i,clusterLabel in enumerate(set(clusterLabels)):
                print(i)
                g = clusterConcepts.loc[clusterConcepts["group"] == clusterLabel]
                groupLabels = g["cluster"].values
                e = events.loc[events["cluster"].isin(set(groupLabels))]

                clusters[clusterLabel] = self.GetClusters(events=e,labels=groupLabels,avgDate=avgDate)
            print("done")
        else:
            clusters = {label: pandas.DataFrame() for label in set(labels)}

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

        res=pandas.concat(clusters)

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
        counter={}
        for _,event in events.iterrows():
            conceptsList = ast.literal_eval(event["concepts"])
            for id, score in conceptsList:
                if id==conceptId and score>=min_score:
                    counter[event.eventId]=event
                    break

        return counter

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

        dateRange = pandas.date_range(dateMin,dateMax).tolist()
        conceptCount = self.CountConcepts()

        heatmap = {date.strftime('%Y-%m-%d'): defaultdict(int) for date in dateRange}
        for _,event in events.iterrows():
            date = event["date"]
            conceptsList = ast.literal_eval(event["concepts"])
            for id,score in conceptsList:
                if conceptCount[id][1] >= min_events:
                    heatmap[date][id] += score if score>=limit else 0

        df = pandas.DataFrame.from_dict(heatmap)
        if out is not None:
            df.to_csv(os.path.join(self.data_subdir,out)+".csv",sep=self.sep,na_rep=0,encoding=self.enc)
        return df

    # compute dataset from heatmap for association rule learning
    def AssociationRulesFromHeatmap(self,out=None):
        heatmap = pandas.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0, na_values=0)  # read frequencies into dataframe
        events = self.GetEvents()
        eventDates = events["date"]
        dateMin = str_to_date(eventDates.min())
        dateMax = str_to_date(eventDates.max())
        weeks = pandas.date_range(dateMin, dateMax,freq="W").tolist()

        averages = self.NoZeroMean(heatmap)

        df = pandas.DataFrame(columns=heatmap.index,index=weeks)

        columns=pandas.to_datetime(heatmap.columns)
        for i,week in enumerate(weeks):
            print(i)
            nextWeek = weeks[-1] if i+1==len(weeks) else weeks[i+1]
            cond = numpy.logical_and(columns>week,columns<=nextWeek)
            those = heatmap.loc[:,cond]
            for j,id in enumerate(heatmap.index):
                avg = averages[j]
                df.at[week,id] = 1 if any(those.loc[id,:]>avg) else 0

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

    # compile dataset for prediction
    def RandomForestFromClusters(self,labels,conceptNames=None,conceptIds=None,groupLabels=None,min_events=10,min_score=50,data=None):
        clusters = self.GetClusters(labels, clusterLabels=groupLabels,avgDate=False,events=data)
        if conceptIds is None:
            conceptIds=self.ConceptNameToId(conceptNames)
        usedConcepts=self.DatasetConcepts()
        for id in conceptIds:
            with open(os.path.join(self.data_subdir, str(id)) + "_x.txt", "w", encoding=self.enc,newline=self.nl) as x, open(os.path.join(self.data_subdir, str(id)) + "_y.txt", "w", encoding=self.enc, newline=self.nl) as y:
                x.write(self.sep.join(usedConcepts.keys()) + self.nl)
                y.write(self.sep.join(usedConcepts.keys()) + self.nl)
                for _, cluster in clusters.groupby(level=0):
                    events = self.EventsByConceptScore(cluster,id,min_score=min_score)
                    for eventId,event in events.items():
                        prev=cluster.loc[cluster.date<event.date]
                        fol=cluster.loc[cluster.date>event.date]
                        if len(prev)>=min_events and len(fol)>=min_events:
                            self.Compile(prev,OrderedDict(usedConcepts),f=x,min_score=min_score)
                            self.Compile(fol,OrderedDict(usedConcepts),f=y,min_score=min_score)

        return usedConcepts

    # compile dataset of attributes
    def Compile(self,events,usedConcepts,f=None,min_score=50):
        for _,event in events.iterrows():
            conceptList = ast.literal_eval(event.concepts)
            for id,score in conceptList:
                if str(id) in usedConcepts and score>=min_score:
                    usedConcepts[str(id)]="1"
        if f is not None:
            f.write(self.sep.join(list(usedConcepts.values()))+self.nl)

        return usedConcepts.values()

    # compute mean by dataframe row
    def NoZeroMean(self,d):
        return numpy.nanmean(d.values,axis=1)

    def CrossValidate(self,date_start="2017-09-01",date_end="2017-09-30"):
        dateRange = pandas.date_range(str_to_date(date_start),str_to_date(date_end)).tolist()
        scores = defaultdict(list)
        for date in dateRange:
            date = date.strftime('%Y-%m-%d')
            print(date)
            train,test = self.TrainTestSplit(date,out=False)
            trainMatrix,trainVocab=self.CsrMatrix(normalized=True,concept_wgt=100,include_date=True,date_wgt=5000,events=train,min_events=5,date_max=date)
            trainModel,trainLabels = self.KMeans(trainMatrix,1000,useMiniBatchKMeans=True)
            train["cluster"]=trainLabels

            testMatrix = self.CsrMatrix(normalized=True, concept_wgt=100, include_date=True, date_wgt=5000, events=test,min_events=5)[0]
            testModel, testLabels = self.KMeans(testMatrix, 1000, useMiniBatchKMeans=True)
            test["cluster"]=testLabels

            today=test.loc[test.date==date]
            conceptIds=self.TopConcepts(today)
            #usedConcepts=self.RandomForestFromClusters(trainLabels,conceptIds=conceptIds,min_events=1,data=train)
            usedConcepts = self.DatasetConcepts()
            clfs=self.RandomForest(conceptIds=conceptIds)

            train_x=[]
            test_y=[]

            for _,event in today.iterrows():
                testCluster=test.loc[test.cluster==event.cluster]
                sampleCsr=self.CsrFromVocab(trainVocab,event,concept_wgt=100,date_wgt=5000)
                sampleCluster= trainModel.predict(sampleCsr)
                trainCluster = train.loc[train.cluster==sampleCluster[0]]
                x=list(self.Compile(trainCluster,OrderedDict(usedConcepts)))
                train_x.append(x)
                y=list(self.Compile(testCluster,OrderedDict(usedConcepts)))
                test_y.append(y)
            test_y=numpy.asarray(test_y,dtype=int)
            train_x=numpy.asarray(train_x,dtype=int)
            for id in conceptIds:
                if clfs[id] is not None:
                    train_y=clfs[id].predict(train_x)
                    score=self.HammingScore(test_y,train_y)
                    print("Hamming score for " + str(id) + " is " + str(score))
                    scores[id].append(score)
        #for id,res in scores.items():


    def CsrFromVocab(self,vocab,event,concept_wgt=100,date_wgt=5000):
        indptr = [0]
        indices = []
        data = []
        conceptsList = ast.literal_eval(event.concepts)  # list of tuples like (conceptId,score)
        #build csr vectors
        for id,score in conceptsList:
            if id in vocab:
                index = vocab[id]
                indices.append(index)
                score = normalize(score,0,100)
                data.append(score*concept_wgt)

        #include date dimension
        index = vocab["date"]
        indices.append(index)
        data.append(1*date_wgt)
        indptr.append(len(indices))

        return csr_matrix((data, indices, indptr), dtype=int,shape=(1,len(vocab)))


    def TopConcepts(self,events,min_score=50):
        concepts=set()
        for _,event in events.iterrows():
            conceptList=ast.literal_eval(event.concepts)
            for id,score in conceptList:
                if score>=min_score:
                    concepts.add(id)

        return concepts




    ### Data Visualisation #########################################################################################################################

    # plot data
    def Plot(self,x,type="bar"):
        if type.lower()=="bar":
            fig, ax = plt.subplots()
            ax.bar(range(len(x)),x,align="center")
        elif type.lower()=="line":
            plt.plot(x)
        elif type.lower()=="dendrogram":
            dendrogram(x#,
                #leaf_rotation=90.,  # rotates the x axis labels
                #leaf_font_size=8.,  # font size for the x axis labels
            )
        plt.show()

    # show concept frequency line graph by date
    def ConceptFrequencyLineGraph(self,conceptNames,method=None):
        heatmap = pandas.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, index_col=0)  # read frequencies into dataframe
        conceptIds = self.ConceptNameToId(conceptNames)
        for id,name in zip(conceptIds,conceptNames):
            data = heatmap.loc[id,:]
            if method is not None:
                data = self.SmoothFunction(data,method).values
            plt.plot(data,label = name)
            data[data==0] = numpy.nan
            plt.axhline(numpy.nanmean(data))
        plt.ylabel("concept score frequency")
        plt.xlabel("days")
        plt.legend()

        plt.show()

    # show concept frequency heatmap by date
    def ConceptFrequencyHeatmap(self,conceptNames=None,n_days=100,n_concepts=50):
        heatmap = self.GetHeatmap()
        if n_concepts>0:
            shown = numpy.empty((n_concepts, n_days))
            with open(os.path.join(self.data_subdir, "conceptCount") + ".json", "r", encoding=self.enc) as f:
                conceptCount = OrderedDict(json.load(f))
            conceptIds=[c[0] for c in islice(reversed(conceptCount.items()),n_concepts)]
            conceptNames=[c[1][2] for c in islice(reversed(conceptCount.items()),n_concepts)]
            for i,id in enumerate(conceptIds):
                #t = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
                t = heatmap.loc[int(id),heatmap.columns[:n_days]].values
                normalizer = numpy.vectorize(normalize)
                t=normalizer(t,t.min(),t.max())
                shown[i] = t


        elif conceptNames is not None:
            shown= numpy.empty((len(conceptNames),n_days))
            conceptIds = self.ConceptNameToId(conceptNames)
            for i,id in enumerate(conceptIds):
                t = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
                shown[i] = heatmap.loc[heatmap.index==id,heatmap.columns[:n_days]].values
        else: shown = heatmap.values

        # map elements
        fig, ax = plt.subplots()
        map = ax.pcolor(shown)

        # put the major ticks at the middle of each cell
        ax.set_yticks(numpy.arange(shown.shape[0]) + 0.5, minor=False)
        ax.set_xticks(numpy.arange(shown.shape[1]) + 0.5, minor=False)

        ax.invert_yaxis()

        # labels
        ax.set_yticklabels(conceptNames, minor=False)
        plt.xticks(rotation=90)
        ax.set_xticklabels(heatmap.columns[:n_days],minor=False)

        plt.show()



    ### Data Analysis #########################################################################################################################

    # compute correlations between concept frequencies
    def ComputeCorrelations(self,method=None):
        heatmap = pandas.read_csv("ConceptHeatmap.csv", sep=self.sep, encoding=self.enc, dtype=str,
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
    def CountConcepts(self,out=None,events=None,method="frequency",desc=False):
        if events is None:
            events = self.GetEvents()
        concepts = self.GetConcepts()
        vocab=defaultdict(list)
        for i,event in events.iterrows():
            try:
                conceptsList = ast.literal_eval(event["concepts"])
            except TypeError as e:
                print(str(i))
                print(event["concepts"])
            for id,score in conceptsList:
                if len(vocab[id])==0:
                    vocab[id].append(0)
                    vocab[id].append(0)
                    #vocab[id].append(concepts.loc[concepts["conceptId"]==str(id)]["title"].values[0])
                vocab[id][0]+=score
                vocab[id][1]+=1
        if method.lower()=="score".lower():
            v=OrderedDict(sorted(vocab.items(), key=lambda cpt: cpt[1][0],reverse=desc))
        elif method.lower()=="frequency".lower():
            v=OrderedDict(sorted(vocab.items(), key=lambda cpt: cpt[1][1],reverse=desc))
        elif method.lower()=="average".lower():
            v=OrderedDict(sorted(vocab.items(), key=lambda cpt: cpt[1][0]/cpt[1][1],reverse=desc))
        if out is not None:
            with open(os.path.join(self.data_subdir, out) + ".json", "w", encoding=self.enc) as f:
                json.dump(v, f, indent=2)

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