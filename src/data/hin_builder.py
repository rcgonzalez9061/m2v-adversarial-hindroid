from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd

from stellargraph import StellarGraph, IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk

from gensim.models import Word2Vec

import os
import pandas as pd
import numpy as np
import pickle
import json
from p_tqdm import p_umap
from shutil import copyfile


def get_features(outfolder, walk_args=None, w2v_args=None, base_data=None):
    '''
    Implements metapath2vec by:
    1. Building a graph
    2. Performing a random metapath walk then 
    3. Applying word2vec on the walks generated.
    ---------
    Parameters:
    outfolder:      Path to directory where output will be saved, should contain app_list.csv
    walk_args:      Arguments for stellargraph.data.UniformRandomMetaPathWalk
    w2v_args:       Arguments for gensim.models.Word2Vec
    '''   
    app_list_path = os.path.join(outfolder, 'app_list.csv')
    nodes_path = os.path.join(outfolder, 'nodes.json')
    edge_path = os.path.join(outfolder, 'edges.csv')
    graph_path = os.path.join(outfolder, 'graph.pkl')
    feature_path = os.path.join(outfolder, 'features.csv')
    app_heap_path = os.path.join('data', 'out', 'all-apps', 'app-data/')
    metapath_walk_outpath = os.path.join(outfolder, 'metapath_walk.json')
    
    # build from preparsed data
    if base_data is not None:
        base_walks = os.path.join(base_data, 'metapath_walk.json')
    
    # generate app list
    apps_df = pd.read_csv(app_list_path)
    app_data_list = app_heap_path + apps_df.app + '.csv'
    
    if os.path.exists(graph_path) and base_data is None:  # load graph from file if present
        with open(graph_path, 'rb') as file:
            g = pickle.load(file)
    else:  # otherwise build graph from data
        with Client() as client, performance_report(os.path.join(outfolder, "performance_report.html")):
            print(f"Dask Cluster: {client.cluster}")
            print(f"Dashboard port: {client.scheduler_info()['services']['dashboard']}")
            
            data = dd.read_csv(list(app_data_list), dtype=str)
            client.persist(data)
            
            nodes = {}
            api_map = None
            
            # setup edges.csv
            if base_data is not None:
                print(f"Copying {os.path.join(base_data, 'edges.csv')}")
                copyfile(
                    os.path.join(base_data, 'edges.csv'),
                    os.path.join(outfolder, 'edges.csv')
                )
            else:
                pd.DataFrame(columns=['source', 'target']).to_csv(edge_path, index=False)

            for label in ['api', 'app', 'method', 'package']:
                print(f'Indexing {label}s')
                uid_map = data[label].unique().compute()
                uid_map = uid_map.to_frame()
                
                if base_data is not None: # load base items
                    base_items = pd.read_csv(
                        os.path.join(base_data, label+'_map.csv'),
                        usecols=[label]
                    )
                    uid_map = pd.concat([base_items, uid_map], ignore_index=True).drop_duplicates().reset_index(drop=True)
                
                uid_map['uid'] = label + pd.Series(uid_map.index).astype(str)
                uid_map = uid_map.set_index(label)
                uid_map.to_csv(os.path.join(outfolder, label+'_map.csv'))
                nodes[label] = IndexedArray(index=uid_map.uid.values)

                # get edges if not api
                if label == 'api':
                    api_map = uid_map.uid  # create api map
                else:
                    print(f'Finding {label}-api edges')
                    edges = data[[label, 'api']].drop_duplicates().compute()
                    edges[label] = edges[label].map(uid_map.uid)
                    edges['api'] = edges['api'].map(api_map)
                    edges.to_csv(edge_path, mode='a', index=False, header=False)

            # save nodes to file
            with open(nodes_path, 'wb') as file:
                pickle.dump(nodes, file)
            
            g = StellarGraph(nodes = nodes,
                             edges = pd.read_csv(edge_path))

    # save graph to file
    with open(graph_path, 'wb') as file:
        pickle.dump(g, file)
    
    # random walk on all apps, save to metapath_walk.json
    print('Performing random walks')
    rw = UniformRandomMetaPathWalk(g)
#     app_nodes = list(g.nodes()[g.nodes().str.contains('app')])
    app_nodes = list(
        apps_df.app.map(
            pd.read_csv(os.path.join(outfolder, 'app_map.csv'), index_col='app').uid
        )
    )
    
#     def run_walks(app):
#         return rw.run([app], n=walk_args['n'], length=walk_args['length'], metapaths=walk_args['metapaths'])
#     metapath_walks = np.concatenate(p_umap(run_walks, app_nodes, num_cpus=walk_args['nprocs'])).tolist()
    metapath_walks = rw.run(app_nodes, n=walk_args['n'], length=walk_args['length'], metapaths=walk_args['metapaths'])
    
    if base_data is not None:  # if building from other data, append to walks
        with open(base_walks, 'r') as mp_file:
            metapath_walks.extend(json.load(mp_file))
    
    with open(metapath_walk_outpath, 'w') as file:
        json.dump(metapath_walks, file)
    
    print('Running Word2vec')
    w2v = Word2Vec(metapath_walks, **w2v_args)
    
    features = pd.DataFrame(w2v.wv.vectors)
    features['uid'] = w2v.wv.index2word
    features['app'] = features['uid'].map(
        pd.read_csv(os.path.join(outfolder, 'app_map.csv'), index_col='uid').app
    )
    features = features[features.uid.str.contains('app')].set_index('uid')
    features.to_csv(feature_path)
