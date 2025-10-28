import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

file_path = "retail_store_inventory.csv"
df = pd.read_csv(file_path)

print("First five rows of the dataset:")
print(df.head())

print("\nColoumns")
print(df.columns.tolist())

print("\nInfor")
print(df.info())

print("\nSummary statistics:")
print(df.describe())

for col in df.columns:
    print(f"\nColoumn: {col}")
    print(df[col].unique() [:10])






