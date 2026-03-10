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

column_names = [
    "datetime", "gamemode",
    "player1.tag", "player1.trophies", "player1.crowns",
    "player1.card1", "player1.card2", "player1.card3", "player1.card4",
    "player1.card5", "player1.card6", "player1.card7", "player1.card8",
    "player2.tag", "player2.trophies", "player2.crowns",
    "player2.card1", "player2.card2", "player2.card3", "player2.card4",
    "player2.card5", "player2.card6", "player2.card7", "player2.card8",
]

df = [pd.read_csv(file, header=None, names=column_names) for file in files]
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