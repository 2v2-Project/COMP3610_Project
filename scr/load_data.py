import pandas as pd

files = [
    "data/raw/20231002.csv",
    "data/raw/20231003.csv",
    "data/raw/20231004.csv",
    "data/raw/20231005.csv",
    "data/raw/20231006.csv",
    "data/raw/20231007.csv",
    "data/raw/20231008.csv",
    "data/raw/20231009.csv",
    "data/raw/20231010.csv",
    "data/raw/20231011.csv",
]

df = [pd.read_csv(file, header=None) for file in files]
df = pd.concat(df, ignore_index=True)

print("Data loaded successfully!")

print("\nDataFrame Shape: ")
print(df.shape)

print("\nDataFrame Head: ")
print(df.head())

print("\nDataFrame Columns: ")
print(df.columns)

print("\nDataFrame Info: ")
print(df.info())

# print("DataFrame Description: ")
# print(df.describe())