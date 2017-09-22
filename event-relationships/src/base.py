from src.eventrelationships import *
import numpy

er=EventRelationships()
#er.GetEventsObj(["society issues"])
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]),new=False)
#er.GetEventClusters()
#er.SparseMatrix("events.csv","concepts.csv")

er.KMeans(er.load_sparse_csr("eventsConceptsMatrix.npz"),100)