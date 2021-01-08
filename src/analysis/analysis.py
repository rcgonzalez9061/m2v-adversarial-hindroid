from dask.distributed import Client
import dask.dataframe as dd
import numpy as np
import os

def compute_summary(app_data, out_folder):
    #  compute statistics
    stats = app_data.describe().compute()
    
    #  write to file
    file_path = os.path.join(out_folder, "summary.json")
    stats.to_json(file_path)


def compute_comparison(app_data, out_folder):
    #  compute statistics
    gb = app_data.groupby("malware").describe().compute()
    
    #  make presentable
    gb.columns = [' '.join(col).strip() for col in gb.columns.values]
    stats = gb.T
    stats.columns = ["Benign","Malware"]
    count = stats.iloc[[0]]
    count.index = ['count']
    stats = count.append(  # rm redundant counts data
        stats[np.logical_not(stats.index.str.contains("count"))]
    )
    
    #  write to file
    file_path = os.path.join(out_folder, "compare.json")
    stats.to_json(file_path)

    
def generate_analysis(data_path, out_folder, job_list=[], **kwargs):
    "Generates aggregates and statistical analysis on app data located in `data_path`"
    client = Client()
    
    #  load data
    app_data_path = os.path.join(data_path, 'app_data.csv')
    app_data = dd.read_csv(app_data_path)
    
    os.makedirs(out_folder, exist_ok=True)

    #  generate labels
    app_data['malware'] = app_data.app.str.contains("\.") == False
    
    if "summary" in job_list:
        compute_summary(app_data, out_folder)
        
    if "compare" in job_list:
        compute_comparison(app_data, out_folder)