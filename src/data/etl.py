from app_parser import get_data
from hin_builder import get_features

def run_etl(outfolder, parse_params, feature_params):
    get_data(outfolder, **parse_params)
    get_features(outfolder, **feature_params)
