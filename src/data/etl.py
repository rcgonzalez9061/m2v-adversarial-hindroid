import pandas as pd
import numpy as np
import os
from pathlib import Path
import concurrent.futures
from scipy import sparse
from itertools import combinations
import json
from traceback import TracebackException
import re

nprocs=9
MAX_APP_DATA_SIZE = 5
APP_DATA_PARSING_CHUNKSIZE = 1000
API_DATA_COLUMNS =  ["app", "api", "invoke_type", "class", "method", "package", "context"]
PACKAGE_CLEANING_PATTERN = r"\/[^/]+;->.*"

class Application():
    """
    Defines a application/APK.
    """
    
    smali_class_pattern = r"L[\w/]*;"
    API_call_pattern = r"invoke-.*"
    
    API_code_block_edges = set()
    all_apis = set()
    apps = set()
    
    def extract_app_ID(self):
        """
        Returns app identifier, currently based on its directory name. 

        May a unique ID later on.
        """
    
        return os.path.basename(self.app_dir)
    
    
    def __init__(self, app_dir):
        self.app_dir = app_dir
        self.ID = self.extract_app_ID()
        self.API_data = None
        self.apis = set()
        self.smali_list = []
        self.num_methods = 0
        
        self.apps.add(self.ID)
        
        
    def find_smali_filepaths(self):
        """
        Retrieves a list of paths to all smali files in the given directory. 
        
        Records paths in self.smali_list and returns them.
        """
        # reset current list in case
        self.smali_list = []
        
        for result in os.walk(self.app_dir):
            current_dir = result[0]
            files = result[2]
            for filename in files:
                smali_ext = '.smali'
                if filename[-len(smali_ext):] == smali_ext:
                    self.smali_list.append(os.path.join(current_dir, filename))

        return self.smali_list
    
    def parse_smali(self, filepath):
        """Parses a singluar smali file
        
        filepath: str, path to smali file"""
        with open(filepath, 'r') as file:
            lines = file.readlines()

        if lines:
            # get class name
            current_class = lines.pop(0).split()[-1]

            # scan for code blocks and API calls
            line_iter = iter(lines)
            current_method = ""
            apis_in_method = set()
            for line in line_iter:
                if ".method" in line:
                    current_method = current_class + "->" + line.split()[-1]
                    self.num_methods += 1
                    self.prepare_B_edges(apis_in_method)
                    apis_in_method = set()
                elif "invoke-" in line:
                    split = line.split()
                    invoke_type = (
                        split[0]
                        .split("-")[-1] # remove invoke
                        .split("/")[0] # remove "/range"
                    )
                    api_call = split[-1]
                    self.all_apis.add(api_call)
                    self.apis.add(api_call)
                    apis_in_method.add(api_call)
                    package = re.sub(PACKAGE_CLEANING_PATTERN, "", api_call)

                    self.API_data.append([self.ID, api_call, invoke_type, current_class, current_method, package, line])
            self.prepare_B_edges(apis_in_method)
            
    def prepare_B_edges(self, apis):
        """
        Adds edges to self.API_code_block_edges.
        
        Params
        -----------
        apis: list, set, or iterable of apis  within the same method.
        """
        edges = [list(i) for i in combinations(apis, r=2)]
        
        for edge in edges:
            edge.sort()
            self.API_code_block_edges.add(tuple(edge))
    
    def parse(self):
        """
        Parses all smali files within the app.
        """
        self.API_data = []
        
        for file_path in self.find_smali_filepaths():
            self.parse_smali(file_path)
            
        api_data = pd.DataFrame(self.API_data, columns=API_DATA_COLUMNS)
            
        return api_data
    
    def get_stats(self):
        """
        Compute basic statistics about the app. 
        
        Returns Series.
        """
        if self.API_data is None:
            data = self.parse()
        else:
            data = pd.DataFrame(self.API_data, 
                                columns=API_DATA_COLUMNS
                               )
            
        stats = pd.Series({
            'app': self.ID,
            'apis': self.apis,
            'app_dir': self.app_dir,
            'num_apis': data.shape[0],
            'num_unique_apis': data.api.unique().size
        })
        stats = stats.append(data.invoke_type.value_counts())
        
        return stats
        
    
    
def find_apps(directory, labeled_by_folder=False):
    """
    Locates the unzipped apk folders of all apps 
    """
    #print(f"Locating apps in {directory}...")
    apps = []
    app_directories = []
    if labeled_by_folder:
        labels = []
        
    for parent_path, subfolders, files in os.walk(directory):
            for subfolder in subfolders:
                if "smali" in subfolder:
                    app_name = os.path.basename(parent_path)
                    app_path = parent_path
                    
                    apps.append(app_name)
                    app_directories.append(parent_path)
                    if labeled_by_folder:
                        label_folder_name = os.path.basename(Path(parent_path).parent)
                        labels.append(label_folder_name)
                    break
    
    df = pd.DataFrame({
        'app': apps, 
        "app_dir": app_directories
    })
    
    if labeled_by_folder:
        df['label'] = labels
        
    return df.set_index('app')

def parse_app(app_dir, api_data_path, app_data_path, context):
    app = Application(app_dir)
    
    try:
        app.parse().to_csv(api_data_path,
                           header=False,
                           index=False,
                           mode='a'
                          )
        context['app_counter'] += 1
        context['parsed_apps'].add(app.ID)
        print(f"\rParsed {context['app_counter']} of {context['num_apps']} apps...", end='')
        stats = app.get_stats().to_frame().T
        stats.to_csv(app_data_path,
                     header=False,
                     index=False,
                     mode='a')
    except Exception as e:
        log(f"\r{type(e).__name__} WHILE PARSING {app.ID} at {app.app_dir}", context['outfolder'])
        log_error(e, context['outfolder'])

        
def get_all(source, project_root, app_list, outfolder=os.path.join('data', 'temp'), index=None, step=0, nprocs=4):
    """
    Parses all apps in the given source directory.
    
    source: str, path to location of all apps.
    project_root: str, path to the project's root directory
    """
    os.makedirs(outfolder, exist_ok=True)
    apps_path = os.path.join(outfolder, 'apps.csv')
    app_data_path = os.path.join(outfolder, 'app_data.csv')
    api_data_path = os.path.join(outfolder, 'api_data.csv')
    
    
    if app_list:  # parse a predetermined app list
        apps_df = pd.read_csv(app_list)
    else:
        # parse folders for app directories, save to file
        apps_df = find_apps(source)
        apps_df.to_csv(apps_path)
    
    context = {
        'app_counter': 0,
        'num_apps': apps_df.shape[0],
        'parsed_apps': set(),
        'outfolder': outfolder,
        'pkgs_parsed': 0
    }
    
    if step <= 1:
        # setup api_data.csv
        pd.DataFrame(columns=API_DATA_COLUMNS).to_csv(api_data_path, index=False)
        
        # setup app_data.csv
        app_data = pd.DataFrame(columns=['app', 'apis', 'app_dir', 
                                         'num_apis', 'num_unique_apis', 
                                         'virtual', 'static', 'direct',
                                         'interface', 'super']).set_index('app')
        app_data.to_csv(app_data_path)

        print(f"RUNNING ETL\nData will be available at {Path(outfolder)}")
        print("STEP 1 - PARSING APPS")
        with open(api_data_path, 'a') as api_data_file, open(app_data_path, 'a') as app_data_file:
            # concurrent execution of smali parsing
            print(f"\rParsed {context['app_counter']} of {context['num_apps']} apps...", end='')
            with concurrent.futures.ThreadPoolExecutor(max_workers=nprocs) as executor:
                parsing_futures = {executor.submit(
                    parse_app, app_dir,         
                    api_data_file,
                    app_data_file,
                    context
                ): app_dir for app_dir in apps_df.app_dir}


        # matrix computation setup
        apps = list(Application.apps)
        apps.sort()  # sort for consistency
        num_apps = len(apps)
        apps_factorized = pd.factorize(apps)
        apps_index = {key: int(index) for index, key in zip(apps_factorized[0], apps_factorized[1])}

        # get/make index
        if index is not None:
            api_index = index['api_index']
            all_apis = api_index
            index['app_index'] = apps_index
        else: 
            all_apis = list(Application.all_apis)
            all_apis.sort()  # sort for consistency
            apis_factorized = pd.factorize(all_apis)
            api_index = {key: int(index) for index, key in zip(apis_factorized[0], apis_factorized[1])}

        # write index to file
        with open(os.path.join(outfolder, 'index.json'), 'w') as outfile: json.dump({
            "app_index": apps_index,
            "api_index": api_index
        }, outfile)
        
    with open(os.path.join(outfolder, 'index.json'), 'r') as file: 
        file = json.load(file)
        apps_index = file['app_index']
        api_index = file['api_index']
        num_apps = len(apps_index)
        num_apis = len(api_index)
    
    if step <= 2:
        # compute A matrix
        print("\nSTEP 2 - BUILDING A MATRIX")
        A_mat = sparse.lil_matrix((num_apps, num_apis), dtype=np.int8)

        app_data = pd.read_csv(app_data_path, index_col='app', iterator=True, chunksize=APP_DATA_PARSING_CHUNKSIZE)
        apps_processed = 0
        for chunk in app_data:
            for app, api_list in chunk.apis.str.slice(1,-1).str.split(', ').items():
                print(f"\rApp {apps_processed} of {num_apps} processed...", end="")
                for api in api_list:
                    app_idx = apps_index[app]
                    app_idx = apps_index[app]
                    try:
                        api_idx = api_index[api.strip("'")]
                        A_mat[app_idx, api_idx] = 1
                    except KeyError:
                        pass
                apps_processed += 1
        print(f"\rApp {apps_processed} of {num_apps} processed...done.", end="")

        # write A
        sparse.save_npz(os.path.join(outfolder, "A_mat.npz"), A_mat.tocsr())
    
#     if index:  # return if simply evaluating
#         print("")
#         return
    
    if step <= 3:
        # compute B matrix
        print("\nSTEP 3 - BUILDING B MATRIX")
        B_mat = sparse.lil_matrix((num_apis, num_apis), dtype=np.int8)

        edge_count = 0
        for api1, api2 in Application.API_code_block_edges:
            if edge_count % 1000 == 0:
                print(f"\rEdge {edge_count} of {len(Application.API_code_block_edges)} processed...", end="")
            try:
                api1_idx = api_index[api1]
                api2_idx = api_index[api2]
                B_mat[api1_idx, api2_idx] = 1
                B_mat[api2_idx, api1_idx] = 1
            except KeyError:
                pass                          

            edge_count += 1
        print(f"\rEdge {edge_count} of {len(Application.API_code_block_edges)} processed...done.", end="")

        # write B
        sparse.save_npz(os.path.join(outfolder, "B_mat.npz"), B_mat.tocsr())
    
    if step <= 4:
        # compute P matrix
        print("\nSTEP 4 - BUILDING P MATRIX")    
        pkg_tracking_folder = 'data/temp/pkgs'
        package_tracking_dict = {}
        api_data_iter = pd.read_csv(api_data_path, usecols=['package', 'api'], iterator=True, chunksize=APP_DATA_PARSING_CHUNKSIZE)
        for chunk in api_data_iter:
            for idx, row in chunk.iterrows():
                if row.package not in package_tracking_dict:
                    package_tracking_dict[row.package] = set()
                api = api_index[row.api]
                package_tracking_dict[row.package].add(api)
        print('Parsed api_data')

        P_mat = sparse.lil_matrix((num_apis, num_apis), dtype=np.int8)

        num_packages = len(package_tracking_dict)
        print(f"\rPackage {context['pkgs_parsed']} of {num_packages} processed...", end="")
        pkg_futures = set()
        while package_tracking_dict:
            key, api_set = package_tracking_dict.popitem()
            with concurrent.futures.ThreadPoolExecutor(max_workers=nprocs) as executor:
                pkg_futures.add(executor.submit(
                    insert_P_edges, 
                    api_set,         
                    P_mat,
                    context,
                    num_packages
                ))
        

        sparse.save_npz(os.path.join(outfolder, "P_mat.npz"), P_mat.tocsr())
        print('Done.')
    print('ETL COMPLETED.')
    return

def insert_P_edges(api_list, P_mat, context, num_packages):
    try:
        for api1 in api_list:
            for api2 in api_list:
                P_mat[api1, api2] = 1
                P_mat[api2, api1] = 1
        print(f"\rPackage {context['pkgs_parsed']} of {num_packages} processed...", end="")
    except Exception as e:
        print(print("".join(TracebackException.from_exception(e).format())))
        log(f"\r{type(e).__name__} WHILE PARSING {app.ID} at {app.app_dir}", context['outfolder'])
        log_error(e, context['outfolder'])

def log(message, outfolder):
    """
    Logs a message to a log file and prints it to the CLI.
    """
    log_path = os.path.join(outfolder, "log.log")
    with open(log_path, mode='a') as log:
        log.write(message)
    print(message)
    
def log_error(exception, outfolder):
    """
    Logs a message to a log file and prints it to the CLI.
    """
    log_path = os.path.join(outfolder, "log.log")
    with open(log_path, mode='a') as log:
        print("".join(TracebackException.from_exception(exception).format()))
        log.write(format_exc(exception))


                           
            
def get_data(project_root, source, recompute_all=False, app_list=None, outfolder=None, index=None, step=0, nprocs=4):
    '''
    Retrieve data for year/location/group from the internet
    and return data (or write data to file, if `outfolder` is
    not `None`).
    '''
    if index:
        with open(index) as index_file:
            index = json.load(index_file)
    
    data = get_all(source, project_root, app_list, outfolder, index, step, nprocs)

    return data
