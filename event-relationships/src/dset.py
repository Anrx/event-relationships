from collections import OrderedDict
import numpy as np
import os.path
from src.helpers import *
from src.eventrelationships import EventRelationships
from enum import Enum
import pandas as pd

class Dset(EventRelationships):
    excluded_types = {"loc", "person", "org"}
    min_score=0
    min_events=100
    predict="concepts"
    window_size=14

    def __init__(self, train_set=None):
        if train_set is None:
            self.xset = OrderedDict()
            self.yset = OrderedDict()
            self.concepts = self.GetConcepts(index_col=0)
            self.conceptCount = self.GetConceptCount()
            self.add_keys=True
        else:
            self.xset = OrderedDict({key: [] for key in list(train_set.xset.keys())})
            self.yset = OrderedDict({key: [] for key in list(train_set.yset.keys())})
            self.concepts = train_set.concepts
            self.conceptCount = train_set.conceptCount
            self.add_keys=False

        self.dsets={DsetAxis.X : self.xset, DsetAxis.Y: self.yset}
        self.dset=(self.xset, self.yset)
        self.nsamples=0

    def Compile(self,events,out=None):
        print("Making dataset: ")
        for _,cluster in events.groupby("cluster"):
            dateMin = str_to_date(cluster.date.min())
            dateMax = str_to_date(cluster.date.max())
            startDate = dateMin
            endDate = (dateMin + timedelta(days=self.window_size))
            dates = pd.to_datetime(cluster.date)

            cluster.sort_values("date", inplace=True)

            while (endDate <= dateMax):
                xCond = np.logical_and(dates >= startDate, dates < endDate)
                xWindow = cluster.loc[xCond]
                yCond = dates == endDate
                yWindow = cluster.loc[yCond]

                self.Concat(xWindow,yWindow)

                startDate = (startDate + timedelta(days=1))
                endDate = (endDate + timedelta(days=1))

        self.Analyze()

        if out is not None:
            save_model(self, out)

    def Concat(self,xevents,yevents):
        if (xevents.size == 0 or yevents.size == 0): return

        self.nsamples+=1

        for id in self.xset.keys(): self.xset[id].append(0);
        for id in self.yset.keys(): self.yset[id].append(0);

        for i, xevent in xevents.iterrows(): self.Append(xevent,DsetAxis.X)
        for i, yevent in yevents.iterrows(): self.Append(yevent,DsetAxis.Y)

    def Append(self,event,axis):
        conceptList = string_to_object(event.concepts)
        for id, score in conceptList:
            if score >= self.min_score and self.concepts.loc[id, :].type not in self.excluded_types and self.conceptCount[str(id)][1] >= self.min_events:
                if id in self.dsets[axis]:
                    self.dsets[axis][id][self.nsamples-1]=1
                elif self.add_keys:
                    self.dsets[axis][id] = [0 for i in range(self.nsamples - 1)]
                    self.dsets[axis][id].append(1)

    def Merge(self,existing):
        for k, v in existing.xset.items():
            self.xset[k] += v
        for k, v in existing.yset.items():
            self.yset[k] += v

        self.nsamples+=existing.nsamples

    def Analyze(self):
        print("dset analysis: ")
        x,y = self.ToArray()
        print("n features: " + str(x.shape[1]))
        print("n classes: " + str(y.shape[1]))
        print("n samples: " + str(self.nsamples))
        print("x % full: " + str((np.count_nonzero(x) / x.size) * 100))
        print("y % full: " + str((np.count_nonzero(y) / y.size) * 100))

    def SaveArray(self):
        x,y=self.ToArray()
        np.savetxt(os.path.join(self.temp_subdir, "_x") + ".txt", x, delimiter=self.sep,header=self.sep.join(map(str, list(self.xset.keys()))), fmt="%d")
        np.savetxt(os.path.join(self.temp_subdir, "_y") + ".txt", y, delimiter=self.sep,header=self.sep.join(map(str, list(self.yset.keys()))), fmt="%d")

    def ToArray(self):
        x = np.asarray(list(self.xset.values()), dtype=int).T
        y = np.asarray(list(self.yset.values()), dtype=int).T
        return x,y

    def GetKeys(self,axis):
        return self.dsets[axis].keys();

class DsetAxis(Enum):
    X=0
    Y=1