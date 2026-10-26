import pandas as pd
from exceptions.DatasetError import DatasetError

file_path = "data/student_dataset.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise DatasetError(f"Dataset not found on : {file_path}")
