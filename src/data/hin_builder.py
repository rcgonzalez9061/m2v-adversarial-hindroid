from dask.distributed import Client
import dask.dataframe as dd
import os
import pandas as pd
import numpy as np
import pickle
from stellargraph import StellarGraph, IndexedArray
from stellargraph.data import UniformRandomMetaPathWalk
from gensim.models import Word2Vec


def get_features(outfolder, walk_args=None, w2v_args=None):
    '''
    ---------
    Paramters:
    app_data_list: list of csv paths to apps that will be considered
    outfolder: path to directory where output will be saved
    walk_args: arguments for stellargraph.data.UniformRandomMetaPathWalk
    w2v_args: arguments for gensim.models.Word2Vec
    '''
    app_list_path = os.path.join(outfolder, 'app_list.csv')
    edge_path = os.path.join(outfolder, 'edges.csv')
    graph_path = os.path.join(outfolder, 'graph.pkl')
    feature_path = os.path.join(outfolder, 'features.csv')
    app_heap_path = os.path.join('data', 'out', 'all-apps', 'app-data/')
    
    # generate app list
    apps_df = pd.read_csv(app_list_path)
    app_data_list = app_heap_path + apps_df.app + '.csv'
    
    data = dd.read_csv(list(app_data_list))
    
    if os.path.exists(graph_path):  # load graph from file
        with open(graph_path, 'rb') as file:
            g = pickle.load(file)
    else:  # build graph
        client = Client()
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
                api_map = uid_map.uid
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
    
    # random walk
    print('Performing random walks')
    rw = UniformRandomMetaPathWalk(g)
    walks = rw.run(nodes=list(g.nodes()), **walk_args)
    
    print('Running Word2vec')
    w2v = Word2Vec(walks, **w2v_args)
    
    features = pd.DataFrame(w2v.wv.vectors)
    features['uid'] = w2v.wv.index2word
    features = features[features.uid.str.contains('app')].set_index('uid')
    features.to_csv(feature_path)
