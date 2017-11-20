from src.eventrelationships import *
from src.Kmeans import *
from src.helpers import *
from sklearn import metrics
#from eventregistry import *
import copy

if __name__ == '__main__':
### generate dataset ######################################################################################################################
    #er=EventRelationships("events","concepts","categories")
    #er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))

    #concat events
    #er=EventRelationships("events","concepts","categories",connect=False)
    #er.ConcatEvents()
    #er.ConcatConcepts()



### generate csr matrix ######################################################################################################################
    #er=EventRelationships("events_train","concepts","categories",connect=False)
    #er.CsrMatrix("csr_matrix_train",min_events=0)
    #er.CsrMatrix("csr_matrix_min2_train",min_events=2)
    #er.CsrMatrix("csr_matrix_normalized_train",min_events=0,normalized=True)



### clustering with silhuette optimization ######################################################################################################################
    #er=EventRelationships("events","concepts","categories",connect=False)
    #matrix =load_sparse_csr("csr_matrix_date_wgt5000_min5_train.npz")
    #er.Cluster(load_sparse_csr("csr_matrix.npz"),200,16000,200,"KMeans",max_iter=100,n_init=5,n_jobs=-1)
    #er.Cluster(matrix,min=1000,max=14000,exp=1000,verbose=0,method="minibatchkmeans")
    #er.Cluster(matrix,max=13,n_jobs=-1)



### clustering without silhuette optimization ######################################################################################################################
    #er=EventRelationships("events_train","concepts","categories",connect=False)
    #matrix =load_sparse_csr("csr_matrix_normalized_train.npz")
    #model = er.AgglomerativeClustering(matrix,imp="scipy")
    #model,labels = er.KMeans(matrix,1000,verbose=1,useMiniBatchKMeans=True,out="KMeans_1000_normalized_train_miniBatch")

    #model=load_model("KMeans_2000_date_wgt5000_min5_train",er.enc)
    #er.CountClusterSize(labels)

    #CustomKmeans(matrix,100)
    #model,labels,score=er.KMeans(matrix,100,verbose=1,useMiniBatchKMeans=False)
    #model,labels=er.NMF(matrix,1000,seed=1,maxiter=30)
    #model,labels = er.DBSCAN(matrix,maxDistance=0.5,minSamples=5)
    #model,labels = er.AffinityPropagation(matrix)
    #model,labels = er.AgglomerativeClustering(matrix.toarray(),n_clusters=1000,affinity="l1",linkage="average")
    #silhuette(matrix,labels)
    #save_model(model,"KMeans_100_min5_train")
    #er.ShowRelationships(labels, "KMeans_100_min5_train")

    #find specific clusters
    #er=EventRelationships("events","concepts","categories",connect=True)
    #model = load_model("KMeans-100-date-wgt5000",er.enc)
    #er.FindEventRelationships(model.labels_,["war","immigration"])



### visualization ######################################################################################################################
    #prepare dataset for heatmap
    #er = EventRelationships("events", "concepts", "categories", connect=False)
    #er.ConceptHeatmap(out="ConceptHeatmap")

    #concept line graph
    #er = EventRelationships("events", "concepts", "categories", connect=True)
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"])
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"],method="mean")
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"],method="median")

    #concept frequency heat map
    #er = EventRelationships("events", "concepts", "categories", connect=True)
    #er.ConceptFrequencyHeatmap(n_days=90)



### cluster concepts ######################################################################################################################
    #er = EventRelationships("events_train", "concepts", "categories", connect=False)
    #model = load_model("KMeans_2000_date_wgt5000_min5_train",er.enc)
    #clusterConcepts=er.GetClusterConcepts(model.labels_,out="KmeansClusterConcepts",method="average")
    #clusterConcepts = pandas.read_csv(os.path.join(er.data_subdir, "KmeansClusterConcepts") + ".csv", sep=er.sep, encoding=er.enc)
    #matrix=er.CsrMatrix(out="csr_matrix_clusters",events=clusterConcepts,min_events=0).toarray()
    #matrix =load_sparse_csr("csr_matrix_clusters.npz")
    #groupModel,labels = er.AgglomerativeClustering(matrix,n_clusters=200,affinity="euclidean",linkage="ward")
    #groupModel,labels = er.AffinityPropagation(matrix)
    #model = er.AgglomerativeClustering(matrix,imp="scipy")
    #er.Plot(model,type="dendrogram")
    #er.CountClusterSize(labels)
    #silhuette(matrix,labels)
    #save_model(groupModel,"Hierarichal_200_clusters")
    #er.ShowRelationships(labels, "Affinity_min5_train")

    #show cluster relationships
    er = EventRelationships("events_train", "concepts", "categories", connect=False)
    model = load_model("KMeans_2000_date_wgt5000_min5_train", er.enc)
    groupModel = load_model("Hierarichal_200_clusters", er.enc)
    er.ShowClusterRelationships(model.labels_,groupModel.labels_,"Hierarichal_200_clusters")