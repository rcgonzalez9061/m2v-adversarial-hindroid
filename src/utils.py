
import os
import nbformat
from nbconvert import HTMLExporter, PDFExporter
import json
from etl import get_data
from scipy.sparse import load_npz


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

def run_tests():
    passed = True

    with open('config/test-data-params.json') as fh:
        data_cfg = json.load(fh)

    get_data(**data_cfg)

    with open('config/test-analysis-params.json') as fh:
        analysis_cfg = json.load(fh)
    
    A_mat_path = os.path.join(data_cfg['outfolder'], "A_mat.npz")
    B_mat_path = os.path.join(data_cfg['outfolder'], "B_mat.npz")
    P_mat_path = os.path.join(data_cfg['outfolder'], "P_mat.npz")
    
    print("RUNNING TESTS...")
    
    for matrix_name in ["A", "B", "P"]:
        expected_mat_path = f"test/expected/{matrix_name}_mat.npz"
        results_mat_path = os.path.join(data_cfg['outfolder'], f"{matrix_name}_mat.npz")
        matches = match_matrices(expected_mat_path, results_mat_path)
        
        if matches:
            print(f"ERROR: Matrix {matrix_name} did not match!")
            passed = False
            

    if passed:
        print("TESTS PASSED.")
    else:
        print("TEST(S) FAILED!")
