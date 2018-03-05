from src.eventrelationships import *
from src.Kmeans import *
from src.helpers import *
from sklearn import metrics
import pandas
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
    #er.CsrMatrix("csr_matrix_min5_date_wgt5000_train",min_events=5,normalized=True,concept_wgt=100,date_wgt=5000)



### clustering with silhuette optimization ######################################################################################################################
    #er=EventRelationships("events","concepts","categories",connect=False)
    #matrix =load_sparse_csr("csr_matrix_date_wgt5000_min5_train.npz")
    #er.Cluster(load_sparse_csr("csr_matrix.npz"),200,16000,200,"KMeans",max_iter=100,n_init=5,n_jobs=-1)
    #er.Cluster(matrix,min=1000,max=14000,exp=1000,verbose=0,method="minibatchkmeans")
    #er.Cluster(matrix,max=13,n_jobs=-1)



### clustering without silhuette optimization ######################################################################################################################
    #er=EventRelationships("events_train","concepts","categories",connect=False)
    #matrix,v=er.CsrMatrix(min_events=100,verbose=True)
    #matrix = load_sparse_csr("csr_matrix_min5_train")

    #model,labels = er.KMeans(matrix,1000,verbose=1,useMiniBatchKMeans=True,out="MiniBatchKMeans_1000_min2_date_wgt10000_train")
    #model,labels=er.KMeans(matrix,1000,useMiniBatchKMeans=True,out="MiniBatchKMeans_1000_min5_train")
    #CustomKmeans(matrix,100)
    #model,labels=er.KMeans(matrix,1000,verbose=1,useMiniBatchKMeans=False)
    #model,labels=er.NMF(matrix,1000,out="NMF_1000_min5_train")
    #model,labels = er.DBSCAN(matrix,max_distance=50,min_samples=5,metric="cosine",leaf_size=100)
    #model,labels=er.Birch(matrix,copy=False)
    #model,labels = er.MeanShift(matrix.toarray())
    #model,labels = er.SpectralClustering(matrix)
    #model,labels = er.AffinityPropagation(matrix)
    #model,labels = er.AgglomerativeClustering(matrix.toarray(),n_clusters=1000,affinity="l1",method="average")
    #model = er.AgglomerativeClustering(matrix,imp="scipy")

    #model=load_model("MiniBatchKMeans_1000_min5_train",er.enc)

    #W=model.transform(matrix)
    #labels=[]
    #for sample in W:
    #    labels.append(np.argmax(sample))
    #er.CountClusterSize(model.labels_)
    #events=er.GetEvents()
    #events["cluster"]=labels
    #er.ShowRelationships(labels, "NMF_1000_min5_train",events)


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
    #matrix,vocab = er.CsrMatrix(out="csr_min50_date1000", min_events=50,date_wgt=1000,concept_wgt=100,normalized=True,include_date=True)
    #model,labels = er.AgglomerativeClustering(matrix,n_clusters=5000,out="Agglomerative2000min50date1000")
    #model,labels = er.KMeans(matrix,n_clusters=5000,out="KMeans5000min50date1000",useMiniBatchKMeans=False)
    #model = load_model("KMeans5000min50date1000",er.enc)
    #labels= model.labels_
    #er.CountClusterSize(labels)
    #clusters=er.GetClusters(labels)
    #er.ShowRelationships(labels, "KMeans5000min50date1000",clusters)
    #clusterConcepts=er.GetClusterConcepts(clusters,out="AgglomerativeGroup",method="average")
    #clusterConcepts = pandas.read_csv(os.path.join(er.data_subdir, "AgglomerativeGroup") + ".csv", sep=er.sep, encoding=er.enc)
    #groupMatrix,vocab=er.CsrMatrix(events=clusterConcepts,min_events=0)

    #groupModel,groupLabels = er.AgglomerativeClustering(groupMatrix,n_clusters=500,out="AgglomerativeGroup500")
    #groupModel = load_model("AgglomerativeGroup500",er.enc)
    #groupLabels= groupModel.labels_
    #er.CountClusterSize(groupLabels)

    #er.ShowClusterRelationships(model.labels_,groupModel.labels_,"AgglomerativeRelationships500")

### random forest prediction ######################################################################################################################
    er = EventRelationships("events", "concepts", "categories", connect=False)
    er.CrossValidateByCluster()

### testing ######################################################################################################################
    #er = EventRelationships("events_train", "concepts", "categories", connect=False)
    #er.TrainTestSplit("2017-09-01")

    #er = EventRelationships("events_train", "concepts", "categories", connect=True)
    #events = er.GetEvents()
    #model = load_model("MiniBatchKMeans_1000_min5_train", er.enc)
    #events["cluster"] = model.labels_
    #train = events.loc[events.cluster == 538]
    #v,n=er.CountConcepts(events=train,desc=True, min_score=0,values_only=True,include_names=True)
    #er.Plot(v[:50],x_labels=n[:50])

    #trainMatrix, trainVocab = er.CsrMatrix(min_events=50, verbose=True)
    #trainModel, trainLabels = er.KMeans(trainMatrix, 1000, useMiniBatchKMeans=True, nar=True,
    #                                    out="MiniBatchKMeans_1000_min5_train_cluster538")
    #train["cluster"] = trainLabels
    #er.CountClusterSize(model.labels_)
    #er.ShowRelationships(model.labels_, "MiniBatchKMeans_1000_min5_train_cluster538", train)

### Sequential clustering ######################################################################################################################
    #er = EventRelationships("events", "concepts", "categories", connect=False)
    #er.ClusterByTimeWindow()
    #er.PredictByCluster()