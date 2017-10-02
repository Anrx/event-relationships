from src.eventrelationships import *
from sklearn import metrics
import numpy

er=EventRelationships(connect=False)
#er.GetEventsObj(["society issues"])
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))
#er.GenerateEventsCsv(er.GetEventsIter(["society issues"]),new=False)
#er.GetEventClusters()
#er.SparseMatrix("events.csv","concepts.csv")

#er.KMeans(er.load_sparse_csr("eventsConceptsMatrix.npz"),100)

#er.Cluster(er.load_sparse_csr("eventsConceptsMatrix.npz"),100,16000,100,"KMeans")
#er.Cluster(er.load_sparse_csr("eventsConceptsMatrix.npz"),100,16000,100,"NMF")

matrix = er.load_sparse_csr("eventsConceptsMatrix.npz")
#model,labels=er.KMeans(matrix,1000)
model,labels=er.NMF(matrix,4000)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(matrix, labels))
er.ShowRelationships(model,labels,4000,"events.csv","results2.txt")
