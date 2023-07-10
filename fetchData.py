import pandas as pd
from config import *
def read_data(src_file_path):

    df=pd.read_csv(src_file_path)
    return df