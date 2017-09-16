from eventregistry import *
import json
import pandas
import numpy
import logging
import time
import math
import ast
import scipy
import bisect


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

    def SparseMatrix(self, filename):
        events = pandas.read_csv(filename, sep=self.sep, encoding=self.enc, dtype=str)  # read events into dataframe
        eventConcepts = events["concepts"]  # get just the event concepts
        eventIds = events["eventId"]  # get just the event ids

        concepts = pandas.read_csv("concepts.csv", sep=self.sep, encoding=self.enc,
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

        numpy.save("eventsConceptsMatrix.csv",data)
        numpy.savetxt("eventIds.csv",eventIds)
        numpy.savetxt("conceptIds.csv",uniqueConceptIds)