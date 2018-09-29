from eventrelationships import EventRelationships
from helpers import *
from eventregistry import *
import pandas as pd
import csv

class ErInterface(EventRelationships):

    def __init__(self,connect=True):
        super()
        if connect:
            self.er = EventRegistry()  # gets key from settings.json file in module root by default
            self.returnInfo = ReturnInfo(eventInfo=EventInfoFlags(socialScore=True))  # set return info



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
            save_as_json(res,os.path.join(self.data_subdir, self.events_filename))

        return res

    # returns an iterator
    def GetEventsIter(self, categories,dateStart="2017-01-01",dateEnd="2017-03-01"):
    #def GetEventsIter(self, categories,dateStart="2017-03-01",dateEnd="2017-04-30"):
        categoryURIs = [self.er.getCategoryUri(cat) for cat in categories]

        q = QueryEventsIter(categoryUri=QueryItems.OR(categoryURIs),lang="eng",dateStart=dateStart,dateEnd=dateEnd,minArticlesInEvent=10)
        res = q.execQuery(self.er,sortBy="date",sortByAsc=True,count=50,returnInfo=self.returnInfo)

        return res

    # returns ids of given concept names
    def ConceptNameToId(self,conceptNames):
        conceptUris = [self.er.getConceptUri(con) for con in conceptNames]
        concepts = self.GetConcepts()
        selected = pd.Series()
        for uri in conceptUris:
            selected = selected.append(concepts.loc[concepts["uri"] == uri], ignore_index=True)
        return selected["conceptId"].astype(int)



    ### Events CSV #########################################################################################################################

    def GenerateEventsCsv(self, eventIter):
        headers = ["eventId", "uri", "title","summary", "date", "location","socialScore","articleCount","concepts","categories"]

        if file_exists(os.path.join(self.data_subdir, self.events_filename)+".csv"):
            self.UpdateEvents(eventIter)
        else:
            self.CreateEvents(eventIter, headers)

    def GenerateEventCsvLine(self, event):
        eventId=str(event["id"]) if "id" in event else event["uri"]
        uri = event["uri"]
        title = event["title"]
        title = title["eng"].replace(self.sep, ",") if "eng" in title else title[list(title.keys())[0]].replace(
            self.sep, ",")
        summary = event["summary"]
        summary= summary["eng"].replace(self.sep, ",") if "eng" in summary else summary[list(summary.keys())[0]].replace(
            self.sep, ",")
        summary = summary.replace(self.nl, " ")
        summary = summary.replace("\"","")
        summary = summary.replace("'","")
        date = event["eventDate"]
        location = json.dumps(event["location"])
        socialScore = str(event["socialScore"])
        articleCount = str(event["totalArticleCount"])
        concepts = ",".join(map(lambda c: "(\"" + str(c["uri"]) + "\"," + str(c["score"]) + ")", event["concepts"])) if event[
            "concepts"] else "null"
        categories = ",".join(map(lambda c: "(\"" + str(c["uri"]) + "\"," + str(c["wgt"]) + ")", event["categories"])) if event[
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
        existingUris = set(existingEvents.uri)

        with open(os.path.join(self.data_subdir, self.events_filename)+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for event in eventIter:
                try:
                    if "warning" not in event and event["concepts"] and event["uri"] not in existingUris:
                        f.write(self.GenerateEventCsvLine(event))
                        self.GenerateConceptsCsv(event["concepts"])
                        self.GenerateCategoriesCsv(event["categories"])
                except KeyError as e:
                    print(json.dumps(event))
                    raise

    #todo updates dataset with new columns
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
        conceptId = str(concept["id"]) if "id" in concept else concept["uri"]
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
        existingUris = set(existingConcepts.uri)

        with open(os.path.join(self.data_subdir, self.concepts_filename)+".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for concept in concepts:
                try:
                    if concept["uri"] not in existingUris:
                        x = self.GenerateConceptCsvLine(concept)
                        f.write(x)
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
        categoryId = str(category["id"]) if "id" in category else category["uri"]
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
        existingUris = set(existingCategories.uri)

        with open(os.path.join(self.data_subdir, self.categories_filename) + ".csv", "a", encoding=self.enc, newline=self.nl) as f:
            for category in categories:
                try:
                    if category["uri"] not in existingUris:
                        f.write(self.GenerateCategoryCsvLine(category))
                except KeyError as e:
                    print(json.dumps(category))
                    raise

    def RemoveDuplicates(self,dataframe,out):
        dataframe.drop_duplicates(inplace=True)
        dataframe.to_csv(out,sep=self.sep,encoding=self.enc,na_rep=None,header=True,index=False,mode="w")

