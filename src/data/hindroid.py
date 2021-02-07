from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, recall_score

import os
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
from p_tqdm import p_umap, p_imap
from scipy import sparse
from itertools import combinations, product
from functools import partial
import csv

def dask_prep(edges_path, client):
    print(f"Dask Cluster: {client.cluster}")
    print(f"Dashboard port: {client.scheduler_info()['services']['dashboard']}")

    edges = dd.read_csv(edges_path, dtype=str)
    edges['target'] = edges.target.str.replace('api', '').astype(int)


    # A matrix prep
    app_api_edges = edges[edges.source.str.startswith('app')]
    app_api_edges['source'] = app_api_edges.source.str.replace('app', '').astype(int)
    app_api_edges.groupby('source').target.unique().compute().sort_index().to_pickle('data/temp/app_api_sets.pkl')

    # B matrix prep
    api_method_edges = edges[edges.source.str.startswith('method')]
    api_method_edges = api_method_edges.groupby('source').target.unique().compute()
    api_method_edges[api_method_edges.apply(len)>1].to_pickle('data/temp/method_api_sets.pkl')

    # P matrix prep
    api_package_edges = edges[edges.source.str.startswith('package')]
    api_package_edges = api_package_edges.groupby('source').target.unique().compute()
    api_package_edges[api_package_edges.apply(len)>1].to_pickle('data/temp/package_api_sets.pkl')

def build_matrices(outfolder, base_data=None):
    os.makedirs(os.path.join(outfolder, 'hindroid'), exist_ok=True)

    edges_path = os.path.join(outfolder, 'edges.csv')
    app_map_path = os.path.join(outfolder, 'app_map.csv')
    api_map_path = os.path.join(outfolder, 'api_map.csv')

    apis = pd.read_csv(api_map_path, index_col='api').uid.str.replace('api', '').astype(int).values
    apps = pd.read_csv(app_map_path, index_col='app').uid.str.replace('app', '').astype(int).values
    
    num_apps = apps.size
    num_apis = apis.size
    
    mlb = MultiLabelBinarizer(sparse_output=True)
    mlb.fit([(api,) for api in apis])
        
    with Client() as client:
        dask_prep(edges_path, client)

    # A matrix
    print("Constructing A matrix...")

    A_mat = mlb.transform(pd.read_pickle('data/temp/app_api_sets.pkl'))
    f'Constructed: {repr(A_mat)}'
    sparse.save_npz(os.path.join(outfolder, 'hindroid', 'A_mat.npz'), A_mat)

    # B Matrix
    print("Constructing B matrix...")
    B_mat = build_BP_mat(pd.read_pickle('data/temp/method_api_sets.pkl'), num_apis)
    print(f'Constructed: {repr(B_mat)}')
    sparse.save_npz(os.path.join(outfolder, 'hindroid', 'B_mat.npz'), B_mat.astype('int'))
    
    # P Matrix
    print("Constructing P matrix...") 
    P_mat = build_BP_mat(pd.read_pickle('data/temp/package_api_sets.pkl'), num_apis)
    print(f'Constructed: {repr(P_mat)}')
    sparse.save_npz(os.path.join(outfolder, 'hindroid', 'P_mat.npz'), P_mat.astype('int'))
    
    return (A_mat, B_mat, P_mat)

def build_BP_mat(api_sets, num_apis):
    comb_func = lambda api_list: np.array(list(combinations(api_list, r=2)))
    row = []
    col = []
    for combos in p_imap(comb_func, api_sets):
        row.extend(combos[:,0])
        col.extend(combos[:,1])
    mat = sparse.csr_matrix(([True]*len(row), (row, col)), shape=(num_apis, num_apis), dtype=bool)
    del row, col
    mat.setdiag(True)
    mat += mat.T
    return mat

def make_models(source_folder, svm_args={}):
    apps = load_apps(source_folder)
    
    source_folder = os.path.join(source_folder, 'hindroid')
    metapath_map = {
        'AAT': 'A * A.T',
        'ABAT': 'A * B * A.T',
        'APAT': 'A * P * A.T',
        'ABPBTAT': 'A * B * P * B.T *  A.T',
        'APBPTAT': 'A * P * B * P.T *  A.T',
    }
    
    A = sparse.load_npz(os.path.join(source_folder, 'A_mat.npz'))
    B = sparse.load_npz(os.path.join(source_folder, 'B_mat.npz'))
    P = sparse.load_npz(os.path.join(source_folder, 'P_mat.npz'))
    
    metrics = pd.DataFrame(columns = ['kernel', 'acc', 'recall', 'f1'])
    
    for metapath, formula in metapath_map.items():
        print(f'Fitting {metapath} model...')
        commuting_matrix = eval(formula)
        sparse.save_npz(os.path.join(source_folder, f'{metapath}.npz'), commuting_matrix)
        
        mdl = SVC(**svm_args)
        mdl.fit(commuting_matrix.todense(), apps.label)
        
        # collect metrics
        accuracy = accuracy_score(apps.label, mdl.predict(commuting_matrix.todense()))
        recall = recall_score(apps.label, mdl.predict(commuting_matrix.todense()))
        f1 = f1_score(apps.label, mdl.predict(commuting_matrix.todense()))
        metrics = metrics.append(pd.Series({
            'kernel': metapath, 
            'acc': accuracy, 
            'recall': recall, 
            'f1': f1
        }), ignore_index=True)
        
        with open(os.path.join(source_folder, f'{metapath}.mdl'), 'wb') as file:
            pickle.dump(mdl, file)
    print(metrics.set_index('kernel'))

    
def predict():
    metapath_map = {
        'AAT': 'A_test * A.T',
        'ABAT': 'A_test * B * A.T',
        'APAT': 'A_test * P * A.T',
        'ABPBTAT': 'A_test * B * P * B.T *  A.T',
        'APBPTAT': 'A_test * P * B * P.T *  A.T',
    }
    
    
    
    
def load_apps(source_folder):
    apps = (
        pd.read_csv(
            os.path.join(source_folder, 'app_map.csv'),
            dtype=str,
            index_col='app'
        ).join(
            pd.read_csv(
                os.path.join(source_folder, 'app_list.csv'),
                dtype=str,
                index_col='app'
            )
        )
    )
    apps = apps.reset_index().set_index(apps.uid.str.replace('app', '').astype(int)).sort_index()
    apps['label'] = (apps.category=='malware').astype(int)
    return apps

def set_row_csr(A, row_idx, new_row):
    '''
    SOURCE: https://stackoverflow.com/questions/28427236/set-row-of-csr-matrix
    Replace a row in a CSR sparse matrix A.

    Parameters
    ----------
    A: csr_matrix
        Matrix to change
    row_idx: int
        index of the row to be changed
    new_row: np.array
        list of new values for the row of A

    Returns
    -------
    None (the matrix A is changed in place)

    Prerequisites
    -------------
    The row index shall be smaller than the number of rows in A
    The number of elements in new row must be equal to the number of columns in matrix A
    '''
    assert sparse.isspmatrix_csr(A), 'A shall be a csr_matrix'
    assert row_idx < A.shape[0], \
            'The row index ({0}) shall be smaller than the number of rows in A ({1})' \
            .format(row_idx, A.shape[0])
    try:
        N_elements_new_row = len(new_row)
    except TypeError:
        msg = 'Argument new_row shall be a list or numpy array, is now a {0}'\
        .format(type(new_row))
        raise AssertionError(msg)
    N_cols = A.shape[1]
    assert N_cols == N_elements_new_row, \
            'The number of elements in new row ({0}) must be equal to ' \
            'the number of columns in matrix A ({1})' \
            .format(N_elements_new_row, N_cols)

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]
    additional_nnz = N_cols - (idx_end_row - idx_start_row)

    A.data = np.r_[A.data[:idx_start_row], new_row, A.data[idx_end_row:]]
    A.indices = np.r_[A.indices[:idx_start_row], np.arange(N_cols), A.indices[idx_end_row:]]
    A.indptr = np.r_[A.indptr[:row_idx + 1], A.indptr[(row_idx + 1):] + additional_nnz]
    