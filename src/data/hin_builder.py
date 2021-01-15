from dask.distributed import Client, LocalCluster, performance_report
import dask.dataframe as dd
import os
import pandas as pd
import numpy as np
import pickle
import json
from stellargraph import StellarGraph, IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec
from p_tqdm import p_umap


def get_features(outfolder, walk_args=None, w2v_args=None):
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
    edge_path = os.path.join(outfolder, 'edges.csv')
    graph_path = os.path.join(outfolder, 'graph.pkl')
    feature_path = os.path.join(outfolder, 'features.csv')
    app_heap_path = os.path.join('data', 'out', 'all-apps', 'app-data/')
    metapath_walk_outpath = os.path.join(outfolder, 'metapath_walk.json')
    
    # generate app list
    apps_df = pd.read_csv(app_list_path)
    app_data_list = app_heap_path + apps_df.app + '.csv'
    
    if os.path.exists(graph_path):  # load graph from file if present
        with open(graph_path, 'rb') as file:
            g = pickle.load(file)
    else:  # otherwise build graph from data
        with Client() as client, performance_report(os.path.join(outfolder, "performance_report.html")):
            print(f"Dask Cluster: {client.cluster}")
            print(f"Daskboard port: {client.scheduler_info()['services']['dashboard']}")
            
            data = dd.read_csv(list(app_data_list), dtype=str)
            client.persist(data)
            
            nodes = {}
            api_map = None

            pd.DataFrame(columns=['source', 'target']).to_csv(edge_path, index=False)
            for label in ['api', 'app', 'method', 'package']:
                print(f'Indexing {label}s')
                uid_map = data[label].unique().compute()
                uid_map = uid_map.to_frame()
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

            g = StellarGraph(nodes = nodes,
                             edges = pd.read_csv(edge_path))

    # save graph to file
    with open(graph_path, 'wb') as file:
        pickle.dump(g, file)
    
    # random walk on all apps, save to metapath_walk.json
    print('Performing random walks')
    rw = UniformRandomMetaPathWalk(g)
    app_nodes = list(g.nodes()[g.nodes().str.contains('app')])
    
    def run_walks(metapath):
        return rw.run(app_nodes, n=1, length=walk_args['length'], metapaths=[metapath])
    
    metapaths = list(np.array(walk_args['metapaths']*walk_args['n'], dtype=object).flatten())
    metapath_walks = np.concatenate(p_umap(run_walks, metapaths, num_cpus=walk_args['nprocs'])).tolist()
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
