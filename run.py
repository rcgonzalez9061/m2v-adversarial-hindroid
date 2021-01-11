#!/usr/bin/env python

import sys
import json

sys.path.insert(0, 'src/data')
sys.path.insert(0, 'src/analysis')
sys.path.insert(0, 'src/model')
sys.path.insert(0, 'src')

from etl import run_etl
# from app_parser import get_data
# from hin_builder import get_features
from analysis import generate_analysis
from model import train
from utils import convert_notebook#, run_tests

def main(targets):
    '''
    Runs the main project pipeline logic, given the targets.
    targets must contain: 'data', 'analysis', 'model'. 
    
    `main` runs the targets in order of data=>analysis=>model.
    '''

#     if 'test' in targets:
#         run_tests()
    
#     if 'parse-smali' in targets:
#         with open('config/parse-params/parse-params.json') as fh:
#             parse_cfg = json.load(fh)

#         get_data(**parse_cfg)
    
#     if 'features' in targets:
#         with open('config/feature-params/feature-params.json') as fh:
#             feature_cfg = json.load(fh)

#         get_features(**feature_cfg)

    if 'data' in targets:
        with open('config/etl-params/etl-params.json') as fh:
            etl_cfg = json.load(fh)
        
        run_etl(**etl_cfg)

    if 'analysis' in targets:
        with open('config/analysis-params/analysis-params.json') as fh:
            analysis_cfg = json.load(fh)
        
        generate_analysis(**analysis_cfg)

    if 'report' in targets:
        with open('config/report-params.json') as fh:
            analysis_cfg = json.load(fh)
        convert_notebook(**analysis_cfg)
            
    if 'model' in targets:
        with open('config/model-params/model-params.json') as fh:
            model_cfg = json.load(fh)
        # make the data target
        train(**model_cfg)
    return


if __name__ == '__main__':
    # run via:
    # python main.py data model
    targets = sys.argv[1:]
    main(targets)
