from src.eventrelationships import *
from sklearn import metrics
from src.helpers import *
from eventregistry import *
import copy

if __name__ == '__main__':
    #generate dataset
    #er=EventRelationships("events","concepts","categories")
    #er.GetEventsObj(["society issues"])
    #er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))
    #er.ExpandEvents()

    #generate sparse mtrix
    #er=EventRelationships("events.csv","concepts.csv",connect=False)
    #er.SparseMatrix()

    #generate csr matrix
    #er=EventRelationships("events","concepts","categories",connect=False)
    #er.CsrMatrix("csr_matrix")

    #clustering with silhuette optimization
    #er=EventRelationships(connect=False)
    #er.Cluster(load_sparse_csr("csr_matrix.npz"),200,16000,200,"KMeans",max_iter=100,n_init=5,n_jobs=-1)
    #er.Cluster(load_sparse_csr("eventsConceptsMatrix.npz"),100,16000,100,"NMF")

    #clustering without silhuette optimization
    #er=EventRelationships("events","concepts","categories",connect=False)
    #matrix =load_sparse_csr("csr_matrix.npz")
    #model,labels=er.KMeans(matrix,1000,n_init=5,max_iter=100)
    #model,labels=er.NMF(matrix,1000,seed=1,maxiter=30)
    #model,labels = er.DBSCAN(matrix,maxDistance=0.5,minSamples=5)
    #model,labels = er.AffinityPropagation(matrix)
    #model,labels = er.AgglomerativeClustering(matrix.toarray(),n_clusters=1000,affinity="l1",linkage="average")
    #silhuette(matrix,labels)
    #er.ShowRelationships(model, labels, "Agglomerative-1000-l1-average")
    #save_model(model,"Agglomerative-1000-l1-average")

    #adjust centroids
    #er=EventRelationships("events","concepts","categories",connect=False)
    #model = load_model("KMeans-1000",er.enc)
    #matrix =load_sparse_csr("csr_matrix.npz")
    #model, labels = er.AdjustCentroids(matrix,model.labels_,"KMeans")
    #silhuette(matrix, labels)
    #er.ShowRelationships(labels, "KMeans-1000-ajusted2")
    #save_model(model, "KMeans-1000-ajdusted")

    #fix dataset
    #er=EventRelationships("events","concepts","categories",connect=False)
    #er.FindDuplicates()
    #print(str(er.TestUnique()))

    #get relationships
    er=EventRelationships("events","concepts","categories",connect=True)
    model = load_model("KMeans-1000",er.enc)
    er.FindEventRelationships(model.labels_,["catalonia"])

