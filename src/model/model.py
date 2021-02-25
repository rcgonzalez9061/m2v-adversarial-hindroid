import os, pickle, json
from scipy.sparse import csr_matrix, load_npz
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import pandas as pd

import dask.dataframe as dd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from stellargraph import StellarGraph, IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk

from gensim.models import Word2Vec

class M2VDroid():
    def __init__(self, source_folder, classifier=RandomForestClassifier, classifier_args={}, name=None):
        self.name = name if name is not None else os.path.basename(source_folder.rstrip(os.path.sep))
        self._folder = source_folder
        self.edges_path = os.path.join(source_folder, 'edges.csv')
        self.classifier = classifier
        self.classifier_args = classifier_args
        
        self.app_map = pd.read_csv(os.path.join(source_folder, 'app_map.csv'), index_col='app', squeeze=True)
        self.inverse_app_map = pd.read_csv(os.path.join(source_folder, 'app_map.csv'), index_col='uid', squeeze=True)
        self.api_map = pd.read_csv(os.path.join(source_folder, 'api_map.csv'), index_col='api', squeeze=True)
        with open(os.path.join(source_folder, 'metapath_walk.json')) as file:
            self.metapath_walks = json.load(file)
        with open(os.path.join(source_folder, 'nodes.json'), 'rb') as file:
            self.nodes = pickle.load(file)
    
    def fit_predict(self, path, walk_args, w2v_args):
        outpath = os.path.join(path, f'm2v-{self.name}')
        os.makedirs(outpath, exist_ok=True)
        # get app data, compute unique apis
        apps = pd.read_csv(os.path.join(path, 'app_list.csv'), usecols=['app'], squeeze=True, dtype=str)
        app_data_list = os.path.join('data', 'out', 'all-apps', 'app-data/') + apps + '.csv'
        
        print('Computing new edges')
        data = dd.read_csv(list(app_data_list), dtype=str, usecols=['app', 'api']).drop_duplicates().compute()
        data.api = data.api.map(self.api_map)
        data.columns = ['source', 'target']
        data = data.dropna()
        
        nodes = self.nodes.copy()
        nodes['app'] = IndexedArray(index=np.array(list(nodes['app'].index) + list(apps)))
        edges = pd.concat([pd.read_csv(self.edges_path, dtype=str), data], ignore_index=True).reset_index(drop=True)
        g = StellarGraph(nodes=nodes, edges=edges)
        print(g)
        
        print('Running random walk')
        rw = UniformRandomMetaPathWalk(g)
        new_walks = rw.run(list(apps), n=walk_args['n'], length=walk_args['length'], metapaths=walk_args['metapaths'])
        metapath_walks = (
            self.metapath_walks 
            + new_walks
        )
        
        print('Running Word2Vec')
        # make features with word2vec
        w2v = Word2Vec(metapath_walks, **w2v_args)
        
        print('Fitting model')
        features = pd.DataFrame(w2v.wv.vectors)
        features['uid'] = w2v.wv.index2word
#         map_func = lambda uid: uid if uid not in self.app_map else self.app_map[uid]
#         features['app'] = features['uid'].map(map_func)
        X_train = features[features.uid.str.match('app\d+')]
        X_train.uid = X_train.uid.map(self.inverse_app_map)
        X_train = X_train.set_index('uid', drop=True)
        X_test = features.set_index('uid').loc[apps]
        
        # train model and predict new apps
        labels = pd.read_csv('data/out/all-apps/app_list.csv', usecols=['app', 'malware'], index_col='app', squeeze=True)
        y_test = labels[X_test.index]
        y_train = labels[X_train.index]
        
        mdl = self.classifier(**self.classifier_args)
        mdl.fit(X_train, y_train)
        
        pred = mdl.predict(X_test)
        
        print(classification_report(y_test, pred))
        
        results = pd.DataFrame(index=apps).assign(
            m2vDroid=pred,
            true=y_test
        )
        results.to_csv(os.path.join(outpath, 'predictions.csv'))
        return results
            
            
def create_model(outfolder, app_path):
    model_path = os.path.join(outfolder, 'model.pkl')
    
    all_apps = pd.read_csv("data/out/all-apps/all_apps.csv", index_col='app')
    test_apps = pd.read_csv(app_path, index_col = 'app')
    test_apps_mal = test_apps.join(all_apps, how = 'left')

    all_apps_features =  pd.read_csv('data/out/all-apps/features.csv', index_col='uid')
    all_apps_features['app'] = all_apps_features.index.map(
        pd.read_csv('data/out/all-apps/app_map.csv', index_col='uid').app
    )
    all_apps_features['malware'] = (all_apps_features['app'].map(all_apps.category)=='malware').astype(int)
    all_apps_features['category'] = all_apps_features.app.map(all_apps.category)

    train = pd.read_csv('data/out/training-sample/app_map.csv', usecols=['app'])
    train = all_apps_features.set_index('app').loc[train.app]

    test_sample = all_apps_features[np.logical_not(
        all_apps_features.app.apply(lambda x: x in train.index)
    )]
    test_sample['category'] = test_sample.app.map(all_apps.category)
    test_sample = test_sample[test_sample.category!='random-apps']



    X_train, y_train = train.drop(columns=['malware', 'category']), train.malware
    X_test, y_test = test_sample.drop(columns=['app', 'malware', 'category']), test_sample.malware


    model = RandomForestClassifier(max_depth=3, n_jobs=-1)  # probably overfit
    model.fit(X_train, y_train)

    class_train = classification_report(model.predict(X_train), y_train)
    class_test = classification_report(model.predict(X_test), y_test)
    output = test_sample[['app','malware','category']].join(y_test, how = 'left')
    m_score = f1_score(model.predict(X_test), y_test)
    output.to_csv('output.csv')
    with open(model_path, 'wb') as file:
            pickle.dump(class_train, file)
            pickle.dump(class_test,file)
            pickle.dump(m_score,file)