from eventregistry import *
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from scipy.sparse import csr_matrix
from src.helpers import *
import json
import pandas
import numpy
import logging
import time
import math
import ast
import scipy
import gc
import pickle
import os.path

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
        if connect: self.er = EventRegistry()  # gets key from settings.json file in module root by default
        self.returnInfo=ReturnInfo(eventInfo=EventInfoFlags(socialScore=True))

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
            with open(self.events_filename+".json", "w", encoding=self.enc) as f:
                json.dump(res, f, indent=2)

        return res

    # returns an iterator
    def GetEventsIter(self, categories):
        categoryURIs = [self.er.getCategoryUri(cat) for cat in categories]

        q = QueryEventsIter(categoryUri=QueryItems.OR(categoryURIs),lang="eng")
        res = q.execQuery(self.er,returnInfo=self.returnInfo)

        return res

    def GetEvents(self):
        return pandas.read_csv(self.events_filename + ".csv", sep=self.sep, encoding=self.enc, dtype=str) # read events into dataframe

    def GetConcepts(self):
        return pandas.read_csv(self.concepts_filename + ".csv", sep=self.sep, encoding=self.enc, dtype=str) # read concepts into dataframe

    def GetCategories(self):
        return pandas.read_csv(self.categories_filename + ".csv", sep=self.sep, encoding=self.enc, dtype=str)  # read categories into dataframe



    ### Events CSV #########################################################################################################################

    def GenerateEventsCsv(self, eventIter):
        headers = ["eventId", "uri", "title","summary", "date", "location","socialScore","articleCount", "concepts","categories"]
        if file_exists(self.events_filename+".csv"):
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
        with open(self.events_filename+".csv", "w", encoding=self.enc, newline=self.nl) as f:
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

        with open(self.events_filename+".csv", "a", encoding=self.enc, newline=self.nl) as f:
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
    def ExpandEvents(self): #todo
        existingEvents = pandas.read_csv(self.events_filename + ".csv", sep=self.sep, encoding=self.enc)
        existingUris = existingEvents["uri"].tolist()

        events = self.GetEventsObj(eventUris=existingUris)

        with open(self.events_filename + "2.csv", "w", encoding=self.enc, newline=self.nl) as f:
            for event in events:
                try:
                    dd = self.GenerateEventCsvLine(event)
                    f.write(dd)
                    self.GenerateConceptsCsv(event["concepts"])
                except KeyError as e:
                    print(json.dumps(event))
                    raise



    ### Concepts CSV #########################################################################################################################

    def GenerateConceptsCsv(self, concepts):
        headers = ["conceptId", "uri", "title", "type"]
        if file_exists(self.concepts_filename+".csv"):
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
        with open(self.concepts_filename+".csv", "w", encoding=self.enc, newline=self.nl) as f:
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

        with open(self.concepts_filename+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for concept in concepts:
                try:
                    if binsearch(existingIds,int(concept["id"])) is False:
                        f.write(self.GenerateConceptCsvLine(concept))
                except KeyError as e:
                    print(json.dumps(concept))
                    raise



    ### Categories CSV #########################################################################################################################

    def GenerateCategoriesCsv(self, categories):
        headers = ["categoryId", "uri"]
        if file_exists(self.categories_filename + ".csv"):
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
        with open(self.categories_filename + ".csv", "w", encoding=self.enc, newline=self.nl) as f:
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

        with open(self.categories_filename + ".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for category in categories:
                try:
                    if binsearch(existingIds, int(category["id"])) is False:
                        f.write(self.GenerateCategoryCsvLine(category))
                except KeyError as e:
                    print(json.dumps(category))
                    raise



    ### Sparse Matrix #########################################################################################################################

    #compute spare matrix from data
    def SparseMatrix(self,events=None):
        if events is None:
            events = self.GetEvents()
        eventsConcepts = events["concepts"]  # get just the event concepts

        concepts = self.GetConcepts()
        uniqueConceptIds = concepts["conceptId"].astype(int)  # get just the concept ids
        uniqueConceptIds.sort_values(inplace=True)  # sort the list for bisect search

        matrix = []
        for eventConcepts in eventsConcepts:
            conceptsList = ast.literal_eval(eventConcepts)  # list of tuples like (conceptId,score)
            bins = numpy.zeros(uniqueConceptIds.size,dtype=int) #for some reasons, .size returns 1 less than the actual size of the dataset todo or does it
            for id, score in conceptsList:
                index = binsearch(uniqueConceptIds.tolist(), int(id))
                bins[index] = score
            matrix.append(bins)
        return matrix

    #compute csr matrix from data
    def CsrMatrix(self,out):
        events = self.GetEvents()
        eventsConcepts = events["concepts"]  # get just the event concepts

        indptr=[0]
        indices = []
        data=[]
        vocabulary = {}
        for i,eventConcepts in enumerate(eventsConcepts):
            try:
                conceptsList = ast.literal_eval(eventConcepts)  # list of tuples like (conceptId,score)
            except ValueError:
                print(eventConcepts)
                print(str(i))
                raise
            for id,score in conceptsList:
                index = vocabulary.setdefault(id,len(vocabulary))
                indices.append(index)
                data.append(score)
            indptr.append(len(indices))
        matrix = csr_matrix((data, indices, indptr), dtype=int,shape=(len(eventsConcepts),len(vocabulary)))
        save_sparse_csr(out,matrix)
        return matrix



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
    def KMeans(self, x, n_clusters, seed=None, init="k-means++", max_iter=300, n_init=10, n_jobs=1, useMiniBatchKMeans=False):
        if useMiniBatchKMeans:
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                random_state=seed)
            #max_iter=max_iter,
            #n_init=n_init,
            #n_jobs=n_jobs)
        else:
            kmeans = KMeans(
                n_clusters=n_clusters,
                init=init,
                random_state=seed,
                max_iter=max_iter,
                n_init=n_init,
                n_jobs=n_jobs)
        kmeans=kmeans.fit(x)
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
    def AgglomerativeClustering(self,x,n_clusters,affinity="euclidean",caching_dir=None,connectivity=None,full_tree="auto", linkage="ward"):
        model = AgglomerativeClustering(
            n_clusters=2,
            affinity=affinity,
            memory=caching_dir,
            connectivity=connectivity,
            compute_full_tree=full_tree,
            linkage=linkage)
        model=model.fit(x)
        return (model,model.labels_)

    ### Machine Learning ######################################################################################################################

    #optimize a clustering algorithm using silhouette score
    def Cluster(self, x, minClusters, maxClusters,step, method,seed=None,max_iter=300,n_init=10,n_jobs=1):
        bestClusters=0
        bestScore=-1
        bestModel=None
        print("Algorithm: "+method)
        for i in range(minClusters, maxClusters,step):
            if method.lower() == "KMeans".lower():
                model,labels=self.KMeans(x,i,seed=seed, max_iter=max_iter,n_init=n_init,n_jobs=n_jobs)
            elif method.lower() == "NMF".lower():
                model,labels=self.NMF(x,i)
            silhouette=silhuette(x,labels)
            print("n-clusters: "+str(i))
            if silhouette>bestScore:
                bestScore=silhouette
                bestClusters=i
                bestModel = model

        print("Best score: "+str(bestScore) + " for n-clusters: "+str(bestClusters))
        pickle.dump(bestModel,open(method+".obj","w",encoding=self.enc))

    #write all the clusters to a file in a human readable format
    def ShowRelationships(self,labels,out):
        clusters = self.GetClusters(labels)

        mkdir(self.results_subdir)
        with open(os.path.join(self.results_subdir,out)+".txt", "w", encoding=self.enc, newline=self.nl) as f:
            for _,cluster in clusters.groupby(level=0):
                f.write("New cluster: "+self.nl)
                for _,event in cluster.iterrows():
                    f.write(str(event["title"])+" ["+event["date"]+"] => ")
                f.write(self.nl)

    #adjust the cluster centers according to the event with the highest social score in each cluster
    def AdjustCentroids(self,x,labels,method):
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

    #compile the clustered events in a multiindex dataframe
    def GetClusters(self,labels,sorted=True):
        events=self.GetEvents()
        events["cluster"]=labels
        # get clusters
        clusters = {label: pandas.DataFrame() for label in set(labels)}

        for label in set(labels):
            e = events.loc[events["cluster"]==label]
            clusters[label] = clusters[label].append(e, ignore_index=True)
            if sorted: clusters[label].sort_values("date",inplace=True)

        res=pandas.concat(clusters)

        return res

    #search events by list of concepts and return the surrounding clustered events
    def FindEventRelationships(self,labels,conceptNames, before=10,after=10,n_events=1,n_clusters=10):
        # get selected concepts
        conceptUris = [self.er.getConceptUri(con) for con in conceptNames]
        concepts = self.GetConcepts()
        selected = concepts.loc[concepts["uri"].isin(conceptUris)]
        selectedIds = selected["conceptId"].astype(int)

        events = self.GetEvents()

        clusters = self.GetClusters(labels)

        #get matching events
        matchingEvents= pandas.DataFrame()
        events["score"]=0
        events["cluster"]=labels
        for i,event in events.iterrows():
            conceptsList = ast.literal_eval(event["concepts"])
            for id,score in conceptsList:
                if id in selectedIds.values:
                    event["score"]+=score
            if event["score"]>0:
                matchingEvents = matchingEvents.append(event,ignore_index=True)

        #get highest matching event in each cluster
        bestMatches = pandas.DataFrame()
        for _,cluster in matchingEvents.groupby("cluster"):
            best=cluster.loc[cluster["score"].idxmax(),:]
            bestMatches = bestMatches.append(best,ignore_index=True)

        #find cluster with more than 1 sample corresponding to the best matching event
        csize=0
        while csize<2 and len(bestMatches)>0:
            bestMatchingEvent = bestMatches.loc[bestMatches["score"].idxmax(),:]
            bestMatchingCluster = clusters.loc[bestMatchingEvent["cluster"],:]
            csize=len(bestMatchingCluster)
            if csize<2:
                bestMatches = bestMatches.loc[bestMatches.eventId!=bestMatchingEvent.eventId]

        #write to file
        mkdir(self.results_subdir)
        out = self.sep.join(conceptNames)
        with open(os.path.join(self.results_subdir,out) + ".txt", "w", encoding=self.enc, newline=self.nl) as f:
                for _, event in bestMatchingCluster.iterrows():
                    f.write(str(event["title"]) + " [" + event["date"] + "] => ")
                f.write(self.nl)





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