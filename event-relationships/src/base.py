from src.eventrelationships import *
from src.Kmeans import *
from src.helpers import *
from src.erinterface import *

if __name__ == '__main__':
### generate dataset ######################################################################################################################
    #er=ErInterface()
    #er.GenerateEventsCsv(er.GetEventsIter(["society issues"]))
    #er.GetEventsObj(categories=["society issues"],dateStart="2016-01-01",dateEnd="2016-02-02")


    #er=ErInterface(False)
    #er.RemoveDuplicates(er.GetEvents(),"events.csv")

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
    #er = EventRelationships()
    #er.ConceptHeatmap(out="conceptHeatmap")
    #er.CountConcepts(out="conceptCount",include_names=True)
    #er.ConceptFrequencyHeatmap(n_days=366,n_concepts=300)

    #concept line graph
    #er = EventRelationships("events", "concepts", "categories", connect=True)
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"])
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"],method="mean")
    #er.ConceptFrequencyLineGraph(["gun control","mass shooting"],method="median")



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
    er = EventRelationships()
    #date = "2016-04-25"
    #train, test = er.TrainTestSplit(date)
    #trainModel = load_model("scipy" + date, enc=er.enc)
    #trainLabels = fcluster(trainModel, 1000, criterion="distance")
    #clusters = er.GetClusters(trainLabels,train)
    #er.AverageChronologicalLinkage(clusters)
    #er.CrossValidateByCluster()
    #er.ConceptHeatmap(out="conceptHeatmapFreq")
    er.ConceptFrequencyHeatmap()
    #er.PlotConceptTypes()
    #er.PlotBinomialDistribution()

    #train,test = er.TrainTestSplit("2016-04-25",out=True)
    #er.Cluster()
    #model =load_model("scipy2016-04-25",er.enc)
    #er.ShowDendrogram(model,p=50000)
    #er.FindDuplicates()
    #model=load_model("bestCluster",er.enc)
    #events, test = er.TrainTestSplit("2016-04-25")
    #matrix,vocab=er.CsrMatrix(events=events,min_events=100,verbose=True)
    #model, labels = er.KMeans(matrix, 500, useMiniBatchKMeans=False, seed=er.seed, out="500Kmeans2016-04-25")
    #model,labels = er.CosineKmeans(matrix,500)
    #save_model(model,"500CosineKmeans2016-04-25")
    #er.ShowRelationships(model.labels_,"200DBSCAN2016-04-25rel",events)


### double clustering ######################################################################################################################
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

### other ######################################################################################################################
    #er = EventRelationships("events", "concepts", "categories", connect=False)
    #events=er.GetEvents()
    #cat=events.loc[events.categories!="null"]
    #cat.to_csv(os.path.join(er.data_subdir, er.events_filename) + "_cat.csv", sep=er.sep, index=False,encoding=er.enc)

    #er = EventRelationships()
    #cc = er.GetConceptCount()
    #cccounts=[item[1] for key,item in cc.items()]
    #er.PlotDistribution(cccounts)

    #er.PlotEventDateRange()