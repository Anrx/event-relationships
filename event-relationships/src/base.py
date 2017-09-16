from src.eventrelationships import *

er=EventRelationships()
#er.GetEventsObj(["society issues"])
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]),new=False)
#er.GetEventClusters()
er.SparseMatrix("events.csv")