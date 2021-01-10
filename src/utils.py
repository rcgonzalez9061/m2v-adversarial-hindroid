
import os
import nbformat
from nbconvert import HTMLExporter, PDFExporter
import json
# from etl import get_data
from scipy.sparse import load_npz
import os
import pandas as pd

class UIDMapper():
    '''
    Maps keys to a unique identifer with a given prefix.
    '''
    def __init__(self, prefix, initial_dict=None):
        self.prefix = prefix
        if initial_dict is not None:
            self.dict = initial_dict
        else:
            self.dict = dict()
        
    def __next_uid(self):
        return self.prefix + str(len(self.dict))
    
    def add(self, k):
        if k not in self.dict:
            self.dict[k] = self.__next_uid()

    def __getitem__(self, k):
        if k not in self.dict:
            self.dict[k] = self.__next_uid()
        return self.dict[k]

    def __len__(self):
        return len(self.dict)
    
    def __repr__(self):
        return repr(self.dict)
    
    def __str__(self):
        return str(self.dict)

    
def save_dict(d, path):
    '''
    Save dictionary to JSON.
    '''
    UID_to_dit = {key: item.dict for key, item in d.items() if type(item) == type(UIDMapper(''))}
    d = d.copy()
    d.update(UID_to_dit)
    print(d)
    with open(path, 'w') as outfile: 
        json.dump(d, outfile)
        
def read_dict_from_json(path):
    '''
    Read dictionary from JSON.
    '''
    with open(path, 'r') as file: 
        return json.read(file)
       
    
def find_apps(directory, labeled_by_folder=False):
    """
    Locates the unzipped apk folders of all apps 
    """
    #print(f"Locating apps in {directory}...")
    apps = []
    app_directories = []
        
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

    
def convert_notebook(report_in_path, report_out_path, **kwargs):

    curdir = os.path.abspath(os.getcwd())
    indir, _ = os.path.split(report_in_path)
    outdir, _ = os.path.split(report_out_path)
    os.makedirs(outdir, exist_ok=True)

    config = {
        "ExecutePreprocessor": {"enabled": True, "timeout": -1},
        "TemplateExporter": {"exclude_output_prompt": True, 
                             "exclude_input": True, 
                             "exclude_input_prompt": True
                            },
    }

    nb = nbformat.read(open(report_in_path), as_version=4)
    html_exporter = HTMLExporter(config=config)
    
    # no exectute for PDFs
    config["ExecutePreprocessor"]["enabled"] = False
    pdf_exporter = PDFExporter(config=config)

    # change dir to notebook dir, to execute notebook
    os.chdir(indir)

    body, resources = (
        html_exporter
        .from_notebook_node(nb)
    )
    
    pdf_body, pdf_resources = (
        pdf_exporter
        .from_notebook_node(nb)
    )

    # change back to original directory
    os.chdir(curdir)

    with open(report_out_path.replace(".pdf", ".html"), 'w') as fh:
        fh.write(body)
    
    
    with open(report_out_path.replace(".html", ".pdf"), 'wb') as fh:
        fh.write(pdf_body)
        
def match_matrices(A_path, B_path):
    return (
        load_npz(A_path) != load_npz(B_path)
    ).toarray().flatten().any()

# def run_tests():
#     passed = True

#     with open('config/test-data-params.json') as fh:
#         data_cfg = json.load(fh)

#     get_data(**data_cfg)

#     with open('config/test-analysis-params.json') as fh:
#         analysis_cfg = json.load(fh)
    
#     A_mat_path = os.path.join(data_cfg['outfolder'], "A_mat.npz")
#     B_mat_path = os.path.join(data_cfg['outfolder'], "B_mat.npz")
#     P_mat_path = os.path.join(data_cfg['outfolder'], "P_mat.npz")
    
#     print("RUNNING TESTS...")
    
#     for matrix_name in ["A", "B", "P"]:
#         expected_mat_path = f"test/expected/{matrix_name}_mat.npz"
#         results_mat_path = os.path.join(data_cfg['outfolder'], f"{matrix_name}_mat.npz")
#         matches = match_matrices(expected_mat_path, results_mat_path)
        
#         if matches:
#             print(f"ERROR: Matrix {matrix_name} did not match!")
#             passed = False
            

#     if passed:
#         print("TESTS PASSED.")
#     else:
#         print("TEST(S) FAILED!")
