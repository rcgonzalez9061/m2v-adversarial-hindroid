from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score, classification_report

import os
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
from p_tqdm import p_map, p_imap
from scipy import sparse
from itertools import combinations, product
from functools import partial
import csv
from sparse_dot_mkl import dot_product_mkl

class Hindroid():
    def __init__(self, source_folder, name=None):
        # load matrices
        self.name = name if name is not None else os.path.split(source_folder)[1]
        self.A = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'A_mat.npz')).astype('float32')
        self.B = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'B_mat.npz')).astype('float32').tocsr()
        self.P = sparse.load_npz(os.path.join(source_folder, 'hindroid', 'P_mat.npz')).astype('float32').tocsr()
        
        # load models
        with open(os.path.join(source_folder, 'hindroid', 'AAT.mdl'), 'rb') as file:
            self.AAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'ABAT.mdl'), 'rb') as file:
            self.ABAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'APAT.mdl'), 'rb') as file:
            self.APAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'ABPBTAT.mdl'), 'rb') as file:
            self.ABPBTAT = pickle.load(file)
        with open(os.path.join(source_folder, 'hindroid', 'APBPTAT.mdl'), 'rb') as file:
            self.APBPTAT = pickle.load(file)
            
        self.app_map = pd.read_csv(os.path.join(source_folder, 'app_map.csv'), index_col='app', squeeze=True)
        self.api_map = pd.read_csv(os.path.join(source_folder, 'api_map.csv'), index_col='api', squeeze=True)
        self.api_map = self.api_map.str.replace('api', '').astype(int)
    
    def fit_predict(self, path):
        '''
        Predicts all apps listed in the folder defined by `path` in `app_list.csv`.
        
        Outputs predictions to a csv in 
        '''
        # get app data, compute unique apis
        apps = pd.read_csv(os.path.join(path, 'app_list.csv'), usecols=['app'], squeeze=True, dtype=str)
        app_data_list = (
            os.path.join('data', 'out', 'all-apps', 'app-data/') +
            apps
            + '.csv'
        )
        data = dd.read_csv(list(app_data_list), dtype=str, usecols=['app', 'api'])
        data['api'] = data['api'].map(self.api_map)
        data = data.dropna()
        data['api'] = data.api.astype(int)
        print('Computing unique APIs per app')
        apis_by_app = data.groupby('app').api.unique().compute()
        
        def make_feature(api_list):
            app_features = np.zeros(self.api_map.size)
            app_features[list(api_list)] = 1
            return app_features
        
        print("Building A test matrix")
        features = []
        for api_list in tqdm(apis_by_app):
            features.append(make_feature(api_list))
        
        print("Making predictions")
        results = self.batch_predict(features)
        
        true_labels = pd.read_csv('data/out/all-apps/app_list.csv', usecols=['app', 'malware'], index_col='app', squeeze=True)
        true_labels = true_labels[apps]
        
        for col, pred in results.iteritems():
            print(f'{col}:')
            print(classification_report(true_labels, pred))
        results['true'] = true_labels
        results.to_csv(os.path.join(path, f'{self.name}_HD_predictions.csv'))
        
        
        return results
        
    
    def predict(self, x):
        '''
        Predict feature vector(s) of apps with all available kernels.
        
        Parameters:
        -----------------
        x: np.array, feature vectors with width same as the A matrix width, number of unique apis, or self.A.shape[0].
        
        Returns:
        A series with all predictions indexed by their metapath. 
        '''
        metapath_map = {
            'AAT': 'dot_product_mkl(x, self.A.T)',
            'ABAT': 'dot_product_mkl(dot_product_mkl(x, self.B), self.A.T)',
            'APAT': 'dot_product_mkl(dot_product_mkl(x, self.P), self.A.T)',
            'ABPBTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.B), self.P), self.B.T), self.A.T)',
            'APBPTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.P), self.B), self.P.T), self.A.T)',
        }
        
        predictions = {}
        for metapath, formula in metapath_map.items():
            features = eval(formula)
            pred = eval(f'self.{metapath}.predict(features)')
            predictions[metapath]= pred
        
        return pd.Series(predictions)
    
    def predict_with_kernel(self, x, kernel):
        '''
        Predict a feature vector(s) of apps with a specified kernel.
        
        Parameters:
        -----------------
        x: np.array, vector with size the same as the A matrix width, number of unique apis, or self.A.shape[0].
        kernel: str, A member of {'AAT', 'ABAT', 'APAT', 'ABPBTAT', 'APBPTAT'}
        '''
        formula_map = {
            'AAT': 'dot_product_mkl(x, self.A.T)',
            'ABAT': 'dot_product_mkl(dot_product_mkl(x, self.B), self.A.T)',
            'APAT': 'dot_product_mkl(dot_product_mkl(x, self.P), self.A.T)',
            'ABPBTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.B), self.P), self.B.T), self.A.T)',
            'APBPTAT': 'dot_product_mkl(dot_product_mkl(dot_product_mkl(dot_product_mkl(x, self.P), self.B), self.P.T), self.A.T)',
        }
        
        features = eval(formula_map[kernel])
        prediction = eval(f'self.{metapath}.predict(features)')
        
        return predictions
    
    def batch_predict(self, X):
        '''
        Predict a batch of feature vectors of apps with all available kernels.
        
        Parameters:
        -----------------
        X: np.array, vector with size the same as the A matrix width, number of unique apis, or self.A.shape[0].
        
        Returns: DataFrame with predictions using Apps as rows and kernels as columns with the true labels appended.  
        '''
        formula_map = {
            'AAT': 'self.A.T',
            'ABAT': 'dot_product_mkl(self.B, self.A.T)',
            'APAT': 'dot_product_mkl(self.P, self.A.T)',
            'ABPBTAT': 'dot_product_mkl(self.B, dot_product_mkl(self.P, dot_product_mkl(self.B.T, self.A.T)))',
            'APBPTAT': 'dot_product_mkl(self.P, dot_product_mkl(self.B, dot_product_mkl(self.P.T, self.A.T)))',
        }
        
        results = pd.DataFrame(
            columns=['AAT', 'ABAT', 'APAT', 'ABPBTAT', 'APBPTAT'],
        )
        
        # predict by model
        for col in results.columns:
            print(f'Predicting {col}')
            fit_matrix = eval(formula_map[col])
            batch_size = 300 # split features into batches of this size (avoids OOM errors)
            fit_features = sparse.vstack([
                dot_product_mkl(sparse.csr_matrix(X[i:i+batch_size], dtype='float32'), fit_matrix)
                for i in range(0, len(X), batch_size)
            ]).todense()
            preds = eval(f'self.{col}.predict(fit_features)')
            results[col] = preds
        
        return results
        
        
        
        
        
        
        