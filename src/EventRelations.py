import eventregistry as er
import pprint as pp

class EventRelations:
    def __init__(self,key):
        self.key=key
        self.ereg=er.EventRegistry(apiKey=key)

    def GetEventInfo(self,eventURI):
        query=er.QueryEvent(eventURI)
        query.addRequestedResult(er.RequestEventInfo())
        return self.ereg.execQuery(query)
