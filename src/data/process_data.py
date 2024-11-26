import pandas as pd

df = pd.read_csv("../../data/raw/Housing.csv")
df = pd.DataFrame(df)
# ================================================================
# 1. First look to dataset and checking the na or duplicated line
# ================================================================
df.head()
df.info()
df.describe()
df.isna().sum()
df.loc[df.duplicated()]
# ======================================================
# 2. Dummy Variables
# ======================================================
object_columns = df.select_dtypes(include=["object"]).columns


def get_dummy_varibles(df, name):

    if name == "furnishingstatus":
        temp = pd.get_dummies(df[name], drop_first=True, prefix=name).astype(int)
    else:
        temp = pd.get_dummies(df[name], drop_first=True).astype(int)

        temp.columns = [name] * temp.shape[1]

    df.drop([name], axis=1, inplace=True)

    df = pd.concat([df, temp], axis=1)
    return df


for column in object_columns:
    df = get_dummy_varibles(df, column)


df.to_pickle(path="../../data/interim/housing.pkl")
