import pandas as pd
import numpy as np
from carla import Data
from carla.data.catalog import OnlineCatalog
from sklearn.model_selection import train_test_split


class InnDataSet(Data):
    def __init__(self, name, test_size=0.2, carla_dataset=None):
        if carla_dataset is not None:
            self.online_dataset = carla_dataset
        else:
            self.online_dataset = OnlineCatalog(name)
        self.pdf = self.online_dataset.df
        target = self.online_dataset.target

        # put target column to the last
        self.dfx, self.dfy = self.pdf.drop(columns=[target]), pd.DataFrame(self.pdf[target])
        self.pdf = pd.concat([self.dfx, self.dfy], axis=1)
        self.pX = self.dfx
        self.py = self.dfy
        # divide to df1 and df2 for following retraining
        self.pfeat_var_map = {}
        for i in range(len(self.dfx.columns)):
            self.pfeat_var_map[i] = [i]
        size = self.pdf.shape[0]
        np.random.seed(1)
        idx_1 = np.sort(np.random.choice(size, int(size / 2) - 1, replace=False))
        idx_2 = np.array([i for i in np.arange(size) if i not in idx_1])
        self.pX1 = pd.DataFrame(data=self.dfx.values[idx_1], columns=self.dfx.columns)
        self.py1 = pd.DataFrame(data=self.dfy.values[idx_1], columns=self.dfy.columns)
        self.pX2 = pd.DataFrame(data=self.dfx.values[idx_2], columns=self.dfx.columns)
        self.py2 = pd.DataFrame(data=self.dfy.values[idx_2], columns=self.dfy.columns)
        self.pX1_train, self.pX1_test, self.py1_train, self.py1_test = train_test_split(self.pX1, self.py1, stratify=self.py1, test_size=test_size, shuffle=True, random_state=0)

    @property
    def columns(self):
        return list(self.pdf.columns)

    @property
    def features(self):
        return [i for i in list(self.pdf.columns) if i != self.online_dataset.target]

    @property
    def categorical(self):
        return [item for item in self.pX.columns if item not in self.online_dataset.continuous]

    @property
    def continuous(self):
        return self.online_dataset.continuous

    @property
    def X(self):
        return self.dfx

    @property
    def y(self):
        return self.dfy

    @property
    def immutables(self):
        imm = []
        for item in self.online_dataset.immutables:
            for name in self.pdf.columns:
                if item in name:
                    imm.append(name)
        if len(self.online_dataset.immutables) == 0:
            imm.append(self.continuous[-1]) # ensure we can run FACE... Couldn't resolve the bugs in carla
        return imm

    @property
    def target(self):
        return self.online_dataset.target

    @property
    def df(self):
        return self.online_dataset.df

    @property
    def df_train(self):
        return self.pX1_train

    @property
    def df_test(self):
        return self.pX1_test

    @property
    def X1(self):
        return self.pX1

    @property
    def y1(self):
        return self.py1

    @property
    def X2(self):
        return self.pX2

    @property
    def y2(self):
        return self.py2

    @property
    def X1_train(self):
        return self.pX1_train

    @property
    def X1_test(self):
        return self.pX1_test

    @property
    def y1_test(self):
        return self.py1_test

    @property
    def y1_train(self):
        return self.py1_train

    @property
    def ordinal_features(self):
        return {}

    @property
    def continuous_features(self):
        return self.online_dataset.continuous

    @property
    def discrete_features(self):
        lst = [item for item in self.pX.columns if item not in self.online_dataset.continuous]
        # squash binary OHEs to 1
        disc = {}
        for item in lst:
            disc[item] = 1
        return disc

    @property
    def feat_var_map(self):
        return self.pfeat_var_map

    def transform(self, df):
        return df

    def inverse_transform(self, df):
        return df


def load_dataset_utils(d):
    return d.X, d.y, d.df, d.columns, d.ordinal_features, d.discrete_features, d.continuous_features, d.feat_var_map
