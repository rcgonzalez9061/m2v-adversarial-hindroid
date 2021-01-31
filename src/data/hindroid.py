from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.svm import SVC

import os
import pandas as pd
import numpy as np
import pickle
import json
from tqdm import tqdm
from p_tqdm import p_imap
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
    api_method_edges.groupby('source').target.unique().compute().to_pickle('data/temp/method_api_sets.pkl')

    # P matrix prep
    api_package_edges = edges[edges.source.str.startswith('package')]
    api_package_edges.groupby('source').target.unique().compute().to_pickle('data/temp/package_api_sets.pkl')

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
    
#     del apis
    
    with Client() as client:
        dask_prep(edges_path, client)

    # A matrix
    print("Constructing A matrix...")

    A_mat = mlb.transform(pd.read_pickle('data/temp/app_api_sets.pkl'))
    sparse.save_npz(os.path.join(outfolder, 'hindroid', 'A_mat.npz'), A_mat)

    def prep_edges(api_list):
        return set(product(api_list, api_list))

    # B Matrix
    print("Constructing B matrix...")
    temp_edges = list(pd.Series(apis).apply(lambda x: (x,x)))
#     for combos in p_imap(prep_edges, pd.read_pickle('data/temp/method_api_sets.pkl')):
#         temp_edges.extend(combos)
    for api_list in tqdm(pd.read_pickle('data/temp/method_api_sets.pkl')):
        combos = combinations(api_list, r=2)
        temp_edges.extend(combinations(api_list, r=2))
    B_mat = mlb.transform(pd.DataFrame(temp_edges).groupby(0)[1].unique().sort_index())
    sparse.save_npz(os.path.join(outfolder, 'hindroid', 'B_mat.npz'), B_mat)


    # P Matrix
#     print("Constructing P matrix...")
#     temp_edges = list(pd.Series(apis).apply(lambda x: (x,x)))
# #     for combos in p_imap(prep_edges, pd.read_pickle('data/temp/package_api_sets.pkl')):
# #         temp_edges.extend(combos)
#     for package, api_list in tqdm(pd.read_pickle('data/temp/package_api_sets.pkl').items()):
#         if len(api_list) >= 15000: # if combinations too large for memory
#             print(package, len(api_list))
#             for idx1 in api_list:
#                 for idx2 in api_list:
#                     temp_edges.append((idx1, idx2))
#         for i in :
#             temp_edges.extend(combinations(api_list, r=2))
#     P_mat = mlb.transform(pd.DataFrame(temp_edges).groupby(0)[1].unique().sort_index())
#     sparse.save_npz(os.path.join(outfolder, 'hindroid', 'P_mat.npz'), P_mat)


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
    
    for metapath, formula in metapath_map.items():
        print(f'Fitting {metapath} model...')
        commuting_matrix = eval(formula)
        sparse.save_npz(os.path.join(source_folder, f'{metapath}.npz'), commuting_matrix)
        
        mdl = SVC(**svm_args)
        mdl.fit(commuting_matrix.todense(), apps.label)
        
        with open(os.path.join(source_folder, f'{metapath}.mdl'), 'wb') as file:
            pickle.dump(mdl, file)

    
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
    
    