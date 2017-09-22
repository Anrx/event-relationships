from eventregistry import *
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn import metrics
from scipy.sparse import csr_matrix
import json
import pandas
import numpy
import logging
import time
import math
import ast
import scipy
import bisect
import gc


class EventRelationships:
    nl = "\n"
    sep = ";"
    enc = "utf-8"

    def __init__(self, key=None, settings=None):
        self.er = EventRegistry()  # gets key from settings.json file in module root by default

    def GetEventClusters(self):  # supposed to return events clustered by concepts
        q = RequestEventsEventClusters(
            returnInfo=ReturnInfo(eventInfo=EventInfoFlags(), conceptInfo=ConceptInfoFlags()))
        print(q)

    # returns a dictionary-like object and writes the json to file
    def GetEventsObj(self, categories, toFile=True, filename="events.json"):
        categoryUris = [self.er.getCategoryUri(cat) for cat in categories]

        q = QueryEvents(categoryUri=QueryItems.OR(categoryUris))
        res = self.er.execQuery(q)

        if (toFile):
            with open(filename, "w", encoding=self.enc) as f:
                json.dump(res, f, indent=2)

        return res

    # returns an iterator
    def GetEventsIter(self, categories):
        categoryURIs = [self.er.getCategoryUri(cat) for cat in categories]

        q = QueryEventsIter(categoryUri=QueryItems.OR(categoryURIs))
        res = q.execQuery(self.er)

        return res

    ### Events CSV ###

    def GenerateEventsCsv(self, eventIter, filename="events.csv", new=True):
        headers = ["eventId", "uri", "title", "date", "location", "concepts"]
        if new:
            self.CreateEvents(eventIter, filename, headers)
        else:
            self.UpdateEvents(eventIter, filename)

    def GenerateEventCsvLine(self, event):
        title = event["title"]
        title = title["eng"].replace(self.sep, ",") if "eng" in title else title[list(title.keys())[0]].replace(
            self.sep, ",")
        # print("Doing event "+title+".")
        concepts = ",".join(map(lambda c: "(" + c["id"] + "," + str(c["score"]) + ")", event["concepts"])) if event[
            "concepts"] else "null"
        return (str(event["id"]) + self.sep +
                event["uri"] + self.sep +
                title + self.sep +
                event["eventDate"] + self.sep +
                json.dumps(event["location"]) + self.sep +
                concepts + self.nl)

    def CreateEvents(self, eventIter, filename, headers):
        first = True
        with open(filename, "w", encoding=self.enc, newline=self.nl) as f:
            f.write(self.sep.join(headers) + self.nl)
            for event in eventIter:
                try:
                    if "warning" not in event and event["concepts"]:
                        f.write(self.GenerateEventCsvLine(event))
                        self.GenerateConceptsCsv(event["concepts"], new=first)
                        first = False
                except KeyError as e:
                    print(json.dumps(event))
                    raise

    def UpdateEvents(self, eventIter, filename):
        existingEvents = pandas.read_csv(filename, sep=self.sep, encoding=self.enc)  # dataframe
        existingIds = existingEvents["eventId"].values

        with open(filename, "a", encoding=self.enc, newline=self.nl) as f:
            for event in eventIter:
                try:
                    if 'warning' not in event and event["concepts"] and event["id"] not in existingIds:
                        dd = self.GenerateEventCsvLine(event)
                        f.write(dd)
                        self.GenerateConceptsCsv(event["concepts"], new=False)
                except KeyError as e:
                    print(json.dumps(event))
                    raise

    ### Concepts CSV ###

    def GenerateConceptsCsv(self, concepts, filename="concepts.csv", new=True):
        headers = ["conceptId", "uri", "title", "type"]
        if new:
            # start=time.time()
            self.CreateConcepts(concepts, filename, headers)
            # end=time.time()
            # print("Create concepts execution time: "+str(end-start))
        else:
            # start = time.time()
            self.UpdateConcepts(concepts, filename)
            # end = time.time()
            # print("Update concepts execution time: " + str(end - start))

    def GenerateConceptCsvLine(self, concept):
        label = concept["label"]
        label = label["eng"] if "eng" in label else label[list(label.keys())[0]]
        return (str(concept["id"]) + self.sep +
                concept["uri"] + self.sep +
                label + self.sep +
                concept["type"] + self.nl)

    def CreateConcepts(self, concepts, filename, headers):
        with open(filename, "w", encoding=self.enc, newline=self.nl) as f:
            f.write(self.sep.join(headers) + self.nl)
            for concept in concepts:
                try:
                    f.write(self.GenerateConceptCsvLine(concept))
                except KeyError as e:
                    print(json.dumps(concept))
                    raise

    def UpdateConcepts(self, concepts, filename):
        existingConcepts = pandas.read_csv(filename, sep=self.sep, encoding=self.enc, dtype=str)
        existingIds = existingConcepts["conceptId"].values

        with open(filename, "a", encoding=self.enc, newline=self.nl) as f:
            for concept in concepts:
                try:
                    if concept["id"] not in existingIds:
                        f.write(self.GenerateConceptCsvLine(concept))
                except KeyError as e:
                    print(json.dumps(concept))
                    raise

    #compute spare matrix from data
    def SparseMatrix(self, eventsFilename,conceptsFilename):
        events = pandas.read_csv(eventsFilename, sep=self.sep, encoding=self.enc, dtype=str)  # read events into dataframe
        eventConcepts = events["concepts"]  # get just the event concepts
        eventIds = events["eventId"]  # get just the event ids

        concepts = pandas.read_csv(conceptsFilename, sep=self.sep, encoding=self.enc,
                                   dtype=str)  # read concepts into dataframe
        uniqueConceptIds = concepts["conceptId"].astype(int)  # get just the concept ids
        uniqueConceptIds.sort_values(inplace=True)  # sort the list for bisect

        data = []
        for eventConcept in eventConcepts:
            conceptsList = ast.literal_eval(eventConcept)  # list of tuples like (conceptId,score)
            conceptIds = [id for id, score in conceptsList]  # get just the concept ids
            conceptScores = [score for id, score in conceptsList]  # get just the concept score

            bins = numpy.zeros(uniqueConceptIds.size+1,dtype=int)
            for id, score in zip(conceptIds, conceptScores):
                bins[bisect.bisect_left(uniqueConceptIds, id)] = score
            data.append(bins)

        #compute csr matrix
        data = csr_matrix(data)

        self.save_sparse_csr("eventsConceptsMatrix",data)


        #numpy.save("eventsConceptsMatrix",data)
        #numpy.savetxt("eventIds.csv",eventIds,fmt="%s")
        #numpy.savetxt("conceptIds.csv",uniqueConceptIds,fmt="%s")

        #free some ram
        #del events, eventConcepts, eventIds,concepts, uniqueConceptIds,conceptsList,conceptIds,conceptScores,bins
        #gc.collect()

        self.KMeans(data,100)

    def save_sparse_csr(self,filename, x):
        numpy.savez(filename, data=x.data, indices=x.indices,
                 indptr=x.indptr, shape=x.shape)

    def load_sparse_csr(self,filename):
        csr = numpy.load(filename)
        return csr_matrix((csr['data'], csr['indices'], csr['indptr']),
                          shape=csr['shape'])

    # nmf clustering
    def NMF(self,x,nClusters,seed=None):
        model = NMF(n_components=nClusters, init='random', random_state=seed)
        W = model.fit_transform(x)
        H = model.components_

    # dbscan clustering
    def DBSCAN(self,x,maxDistance,minSamples=10):
        db = DBSCAN(eps=maxDistance, min_samples=minSamples).fit(x)
        return db
        #labels = db.labels_
        #n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        #print('Estimated number of clusters: %d' % n_clusters_ )
        #print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))

    #k-means clustering
    def KMeans(self,x,nClusters,seed=None):
        kmeans = KMeans(n_clusters=nClusters, random_state=seed).fit(x)
        #return kmeans
        labels=kmeans.labels_
        print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(x, labels))

    #optimize a clustering algorithm using silhouette score
    def Cluster(self,x,method):
        return 0

