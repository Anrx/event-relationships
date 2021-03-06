from eventrelationships import *
from Kmeans import *
from helpers import *
from erinterface import *
from dset import *

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
    #er = EventRelationships()
    #conceptCount = er.GetConceptCount()
    #bins = [0,0,0,0,0]
    #labels = ["0-5","5-10","10-100","100-1000","1000-30000"]
    #for id,stuff in conceptCount.items():
    #    count = stuff[1]
    #    if count<=5:
    #        bins[0]+=1
    #    elif count<=10:
    #        bins[1] += 1
    #    elif count<=100:
    #        bins[2] += 1
    #    elif count<=1000:
    #        bins[3] += 1#
    #    elif count<=30000:
    #        bins[4] += 1
    #er.Plot(bins,x_labels=labels,rotate=False,out="conceptDistributionByNumEvents.png")

    #er.ConceptHeatmap(out="conceptHeatmap")
    #er.CountConcepts(out="conceptCount",include_names=False)
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
    #er = EventRelationships()
    #date = "2016-04-25"
    #train, test = er.TrainTestSplit(date)
    #trainModel = load_model("scipy" + date, enc=er.enc)
    #trainLabels = fcluster(trainModel, 1000, criterion="distance")
    #clusters = er.GetClusters(trainLabels,train)
    #er.AverageChronologicalLinkage(clusters)
    #er.CrossValidateByCluster()
    #er.Cluster()
    #er.PlotEventDateRange()
    #er.PlotConceptTypes()
    #er.ConceptHeatmap(out="conceptHeatmapFreq")
    #er.ConceptFrequencyHeatmap()
    #er.PlotConceptTypes()
    #er.PlotBinomialDistribution()

    #train,test = er.TrainTestSplit("2016-04-25",out=True)
    #er.Cluster()
    #model =load_model("scipyAggAvg2016-08-28",er.enc)
    #er.ShowDendrogram(model,p=500)
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
    #trainModel, trainLabels = er.KMeans(trainMatrix, 1000, useMiniBatchKMeans=True, nar=True,out="MiniBatchKMeans_1000_min5_train_cluster538")
    #train["cluster"] = trainLabels
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


### work

    er = EventRelationships()
    #er.Cluster()
    er.CrossValidateByCluster()
    #results = []
    #trees = [10,50,100,200]
    #for i in trees:
    #    res = er.CrossValidateByCluster(n_trees=i,one=False,window_size=30)
    #    results.append(res[0])
    #resPlot=np.vstack(results)
    #np.savetxt("30dRandomForest10-300Trees100cGini2016-8-28.csv",resPlot,delimiter=er.sep,encoding=er.enc)
    ##resPlot=np.loadtxt("14dRandomForest10-300Trees100cGini2016-8-28.csv",delimiter=er.sep,encodi#ng=er.enc)
    #plt.style.use("ggplot")
    #xvals=trees
    #plt.plot(xvals,resPlot[:,0],"g",label="Točnost")
    ##plt.plot(xvals,resPlot[:1],"r",label="Hamming loss")
    #plt.plot(xvals,resPlot[:,2],"y",label="Preciznost")
    #plt.plot(xvals,resPlot[:,3],"b",label="Priklic")
    #plt.plot(xvals,resPlot[:,4],"k",label="F-mera")

    #plt.ylabel("Ocena")
    #plt.xlabel("Število dreves")
    #plt.legend()
    #plt.show()

    #er.PlotSVCCoefficients(window_size=30,cindex=54,top_features=20)

    #er = EventRelationships()
    #results = []
    #epochs = [10,50,100]
    #for i in epochs:
    #    res = er.CrossValidateByCluster(n_epochs=i,one=False,window_size=30)
    #    results.append(res[0])
    #resPlot=np.vstack(results)
    #np.savetxt("30dNeuralNetworkBinRel2016-8-28.csv",resPlot,delimiter=er.sep,encoding=er.enc)
    ##resPlot=np.loadtxt("14dRandomForest10-300Trees100cGini2016-8-28.csv",delimiter=er.sep,encodi#ng=er.enc)
    #plt.style.use("ggplot")
    #xvals=epochs
    #plt.plot(xvals,resPlot[:,0],"g",label="Točnost")
    ##plt.plot(xvals,resPlot[:1],"r",label="Hamming loss")
    #plt.plot(xvals,resPlot[:,2],"y",label="Preciznost")
    #plt.plot(xvals,resPlot[:,3],"b",label="Priklic")
    #plt.plot(xvals,resPlot[:,4],"k",label="F-mera")

    #plt.ylabel("Ocena")
    #plt.xlabel("Število dreves")
    #plt.legend()
    #plt.show()

    #er = EventRelationships()
    #resultsPred = []
    #resultsConstant = []
    #clusterModel = load_model("scipyAggWard2016-08-28",enc=er.enc)
    #predModel = load_model("30crossValPred100cOneVsRest2016-08-28",er.enc)
    #train_set= load_model("30dset100c2016-08-28",er.enc)
    #back=er.Validate(predModel,clusterModel,train_set=train_set)
    #resultsPred.append(back[0])
    #resultsConstant.append(back[1])
    #back=er.Validate(predModel,clusterModel,train_set=train_set,date_start="2017-10-01",date_end="2017-10-31")
    #resultsPred.append(back[0])
    #resultsConstant.append(back[1])

    #er.CrossValidateByCluster()

### plots #######################
    #er = EventRelationships()

    #er.PlotEventDateRangePerWeek()
    #er.PlotEventDateRangeErrorBars()
    #er.PlotConceptTypes()
    #er.ConceptFrequencyHeatmap(out="heatmap100d50cFreq.png")
    #er.PlotSVCCoefficients(cindex=54)
    #er.PlotSVCCoefficients(cindex=0)
    #model = load_model("scipyAggWard2016-08-28", er.enc)
    #model = load_model("scipyAggAvg2016-08-28", er.enc)
    #labels = fcluster(model, 100, criterion="maxclust")
    #er.CountClusterSize(labels,frequency=False)
    #er.ShowDendrogram(model)

    #np.savetxt("chronAggClustScores2016-8-28.csv", scores, delimiter=self.sep, encoding=self.enc)
    #np.savetxt("chronAggClustClusters2016-8-28.csv", clusters, delimiter=self.sep, encoding=self.enc)
    #scores=np.loadtxt("silhouetteAggClustScores2016-8-28.csv",delimiter=er.sep,encoding=er.enc)
    #clusters=np.loadtxt("silhouetteAggClustClusters2016-8-28.csv",delimiter=er.sep,encoding=er.enc)
    #plt.plot(clusters, scores)
    #plt.xticks(size=12)
    #plt.yticks(size=12)
    #plt.savefig("silhouetteAggClust2016-08-28", bbox_inches="tight", dpi=300)

    #tsvd = load_model("tsvd2016-08-28", er.enc)
    #er.Plot(np.cumsum(tsvd.explained_variance_ratio_),out="cumSum2000tsvd.png",type="line")

### plot cumsum ###
    #er=EventRelationships()
    #model = load_model("tsvd2016-08-28", enc=er.enc)
    #er.Plot(np.cumsum(model.explained_variance_ratio_),type="line",out="cumSum2000tsvd.png")

    #er = EventRelationships()
    #er.Cluster()
    #er.CrossValidateByCluster()