from dask.distributed import Client
import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from IPython.display import display, HTML
from sklearn.cluster import KMeans
import plotly
import plotly.graph_objs as go
import plotly.io as pio


def make_groundtruth_figures(data_folder):
    vectors = pd.read_csv(os.path.join(data_folder, 'features.csv'))
    vectors = vectors.set_index('app')
    
    all_apps = pd.read_csv("data/out/all-apps/all_apps.csv", index_col='app')
    all_apps['label'] = all_apps[all_apps.category=='malware'].app_dir.str.split('/').apply(lambda list: list[5])
    other_mal_map = {key: "Other malware" for key, value in all_apps.label.value_counts().items() if value <= 200}
    all_apps.label = all_apps.label.map(other_mal_map).fillna(all_apps.label)
    all_apps.label.fillna(all_apps.category, inplace=True)
    
    vectors = vectors.assign(
        label=all_apps.label,
        category=all_apps.category
    )
    labels = vectors.label
    
    # Retrieve node embeddings and corresponding subjects
    node_ids = list(vectors.uid)  # list of node IDs
    node_embeddings = vectors.drop(columns=['uid', 'category', 'label'])
    node_targets = labels

    transform = TSNE  # PCA

    # 2D plot -- matplotlib
    trans = transform(n_components=2)
    node_embeddings_2d = trans.fit_transform(node_embeddings)
    label_map = {l: i for i, l in enumerate(np.unique(node_targets))}
    node_colours = [label_map[target] for target in node_targets]

    plt.figure(figsize=(20, 16))
    plt.axes().set(aspect="equal")
    scatter = plt.scatter(
        node_embeddings_2d[:, 0],
        node_embeddings_2d[:, 1],
        c=node_colours,
        cmap='tab10',
        alpha=0.3
    )
    plt.title("{} visualization of node embeddings".format(transform.__name__))
    legend1 = plt.legend(scatter.legend_elements()[0], label_map.keys(),
                        loc="upper right", title="Classes")
    plt.savefig(os.path.join(data_folder, '2D-plot.png'))
    
    # 3D plot - using plotly
    trans3d = transform(n_components=3)
    node_embeddings_3d = trans3d.fit_transform(node_embeddings)
    data_3d = pd.DataFrame(node_embeddings_3d, index=vectors.index)
    data_3d['malware'] = vectors['category']=='malware'
    data_3d['type'] = vectors.label
    type_chart = data_3d[['malware', 'type']].drop_duplicates()
    type_chart['num'] = type_chart.type.map(label_map)

    layout = go.Layout(
        title="3D representation of node embeddings",
        margin={'l': 0, 'r': 0, 'b': 0, 't': 30},
        legend=dict(y=0.5, itemsizing='constant')
    )

    fig = go.Figure(layout=layout)

    for index, row in type_chart.sort_values('num', ascending=False).iterrows():
        if row['malware']:
            symbol = 'circle'
            name = f"Malware, {row['type']}"
            size = 2
        else:
            symbol = 'x'
            name = f"Non-malware, {row['type']}"
            size = 1.5
        
        if row['type']=='Other malware':
            name=row['type']
        
        df = data_3d[data_3d.type==row['type']]
        rbg = tuple([255*val for val in cm.tab10(row['num'])[:3]])
        color = f"rgb{rbg}"
        trace  = go.Scatter3d(
            name=name,
            x=df[0],
            y=df[1],
            z=df[2],
            customdata=list(df.index),
            hovertemplate=
            "<b>%{customdata}</b><br>" +
            f"{name}" +
            "<extra></extra>",
            mode='markers',
            marker={
                'size': size,
                'opacity': 1,
                'color': color,
                'symbol': symbol
            }
        )

        fig.add_trace(trace)

    # Save the plot.
    pio.write_html(fig, file=os.path.join(data_folder, '3D-plot.html'), auto_open=True)
    
    
def generate_analysis(data_path, job_list=[], **kwargs):
    "Generates plots, aggregates, and statistical analysis on app data located in `data_path`"

    #  load data
#     app_data_path = os.path.join(data_path, 'app_data.csv')
#     app_data = dd.read_csv(app_data_path)
#     os.makedirs(out_folder, exist_ok=True)

    
    if "plots" in job_list:
        make_groundtruth_figures(data_path)
