# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from feature_engine import encoding
from sklearn.preprocessing import OneHotEncoder
import numpy as np

# %%
df = pd.read_csv("train_house.csv")
df_test = pd.read_csv("test.csv")

# %%
df.head(10)

# %%
df.isnull().sum().sort_values(ascending=False).head(20)

# %%
df_test.isnull().sum().sort_values(ascending=False).head(34)

# %%
df_test1 = df_test

# %%
df.shape

# %%
df_test.shape

# %%
df.info()

# %%
df_test.info()

# %%
df.duplicated().sum()

# %%
df["LotFrontage"].describe()

# %%
df_test.describe()

# %%
df.isnull().sum().sort_values(ascending=False).head(23)

# %%
df['GarageQual'].value_counts()

# 1300 +
# Street, LandContour, Utilities, LandSlope, Condition2, RoofMatl,
# BsmtCond, Heating, CentralAir, KitchenAbvGr, Functional, GarageQual,
# GarageCond, PavedDrive, "Utilities", "LandSlope"
# "ExterCond",

# "SaleType", "BldgType", "RoofStyle",
# "BsmtFinType2", 'EnclosedPorch', 'SaleCondition'

# %%
df = df.drop(columns=["PoolQC", "MiscFeature", "Alley", "Fence", "MasVnrType",
                 "FireplaceQu", "Id"], axis=1)

# %%
for col in df:
    print(df[[col]].value_counts())

# %%
df = df.drop(columns=["Street",
                 "LandContour", "Utilities",
                 "LandSlope", "Condition2", "Condition1",
                 "RoofMatl",   
                 "Heating", 
                 'Functional',
                 "SaleType",
                 'EnclosedPorch', 'SaleCondition',
                 "Electrical",
                 "SaleType",
                 "RoofStyle", 'PavedDrive',
                 "BsmtCond", 'GarageCond', "CentralAir",
                 "KitchenAbvGr", "BsmtFinType2"
                 ], axis=1)

# 'GarageQual', "BldgType", "ExterCond", "BsmtFinType2", 


# %%
df_test = df_test.drop(columns=["PoolQC", "MiscFeature", 
                                "Alley", "Fence", "MasVnrType",
                                "FireplaceQu", "Id"], axis=1)

df_test = df_test.drop(columns=["Street",
                 "LandContour", "Utilities",
                 "LandSlope", "Condition2", "Condition1",
                 "RoofMatl", 
                 "Heating",
                 'Functional', 
                 "SaleType",
                 'EnclosedPorch', 'SaleCondition',
                 "Electrical",
                 "SaleType",
                 "RoofStyle", 'PavedDrive',
                 "BsmtCond", 'GarageCond', "CentralAir",
                 "KitchenAbvGr", "BsmtFinType2"
                 ], axis=1)

# %%
df_test.isnull().sum().sort_values(ascending=False).head(20)

# %%
lista_df = df.columns.to_list()
df_num = df.select_dtypes(include=["int64", "float"])
df_cat = df.select_dtypes(include=["object"])

# %%
df_cat.describe()

# %%
df["HouseStyle"].value_counts()

# %%
df["LotFrontage"].mean()

# %%
df.isnull().sum().sort_values(ascending=False).head(10)

# %%
df_test.isnull().sum().sort_values(ascending=False).head(22)

# %%
num_missing = df['LotFrontage'].isnull().sum()
random_values = np.random.randint(60, 80, size=num_missing)
df.loc[df['LotFrontage'].isnull(), 'LotFrontage'] = random_values

# %%
num_missing = df_test['LotFrontage'].isnull().sum()
random_values = np.random.randint(60, 80, size=num_missing)
df_test.loc[df_test['LotFrontage'].isnull(), 'LotFrontage'] = random_values

# %%
df["MasVnrArea"] = df["MasVnrArea"].fillna(0)

# %%
df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)

# %%
df['GarageQual'] = df["GarageQual"].fillna("TA")

# %%
df_test['GarageQual'] = df_test["GarageQual"].fillna("TA")

# %%
df["BsmtQual"] = df["BsmtQual"].fillna("TA")

# %%
df_test["BsmtQual"] = df_test["BsmtQual"].fillna("TA")

# %%
df["BsmtExposure"] = df["BsmtExposure"].fillna("No")

# %%
df_test["BsmtExposure"] = df_test["BsmtExposure"].fillna("No")

# %%
df["BsmtFinType1"] = df["BsmtFinType1"].fillna("Unf")

# %%
df_test["BsmtFinType1"] = df_test["BsmtFinType1"].fillna("Unf")

# %%
df["GarageType"] = df["GarageType"].fillna("Attchd")

# %%
df_test["GarageType"] = df_test["GarageType"].fillna("Attchd")

# %%
df["GarageFinish"] = df["GarageFinish"].fillna("Unf")

# %%
df_test["GarageFinish"] = df_test["GarageFinish"].fillna("Unf")

# %%
df.isnull().sum().sort_values(ascending=False).head(13)

# %%
df_test.isnull().sum().sort_values(ascending=False).head(20)

# %%
df_num.corr(method='spearman').SalePrice.sort_values(ascending=False)

# %%
df_num.corr()

# %%
df = df.drop(columns=["GarageArea", "GarageYrBlt",
                      "BsmtHalfBath",
                      "MSSubClass", "YrSold", "BsmtFinSF2"
                      ])

# %%
df_test = df_test.drop(columns=["GarageArea", "GarageYrBlt",
                      "BsmtHalfBath",                    
                      "MSSubClass", "YrSold", "BsmtFinSF2"
                      ])


# %%
df_num = df.select_dtypes(include=["int64", "float"])
lista_df_num = df_num.columns.to_list()
print(lista_df_num)

# %%
df_num.corr(method='spearman').SalePrice.sort_values(ascending=False)

# %%
df.isnull().sum()
df_test.isnull().sum()

# %%
# histograma saleprice
plt.figure(figsize=(10, 6))
sns.histplot(df['SalePrice'], bins=30, kde=True)
plt.title('Histograma de SalePrice')
plt.xlabel('Preço de Venda (SalePrice)')
plt.ylabel('Frequência')
plt.show()

# %%
# scatterplot saleprice x grlivarea
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('Gráfico de Dispersão: SalePrice vs. GrLivArea')
plt.xlabel('Área de Estar (GrLivArea)')
plt.ylabel('Preço de Venda (SalePrice)')
plt.show()

# %%
# boxplot saleprice x overall
sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette='Set2')
plt.title('Boxplot do Preço de Venda por Qualidade Geral')
plt.xlabel('Qualidade Geral (OverallQual)')
plt.ylabel('Preço de Venda (SalePrice)')
plt.show()

# %%
# yearbuilt x saleprice
bins = [1871, 1953, 1972, 1999, 2010]
labels = ['Antes de 1954', '1954-1972', 
'1973-1999', '2000-2010']
df['YearBuilt_Category'] = pd.cut(df['YearBuilt'], bins=bins, labels=labels, right=False)

# %%
plt.figure(figsize=(12, 6))
sns.boxplot(x='YearBuilt_Category', y='SalePrice', data=df, palette='Set2')
plt.title('Boxplot de SalePrice por Faixas de Ano de Construção')
plt.xlabel('Faixas de Ano de Construção')
plt.ylabel('Preço de Venda (SalePrice)')
plt.xticks(rotation=45) 
plt.show()

# %%
df = df.drop(columns=["YearBuilt_Category"])

# %%
df["TotRmsAbvGrd"].value_counts().sort_values(ascending=False)

# %%
sns.boxplot(data=df_num, x="TotRmsAbvGrd")

# %%
df_test.isnull().sum().sort_values(ascending=False)

# %%
df_test["BsmtFinSF1"].value_counts()

# %%
df_test["MSZoning"] = df_test["MSZoning"].fillna("RL")

# %%
df_test["BsmtFullBath"] = df_test["BsmtFullBath"].fillna(0)

# %%
df_test["Exterior1st"] = df_test["Exterior1st"].fillna("VinylSd")

# %%
df_test["GarageCars"] = df_test["GarageCars"].fillna(2)

# %%
df_test["KitchenQual"] = df_test["KitchenQual"].fillna("TA")

# %%
df_test["TotalBsmtSF"] = df_test["TotalBsmtSF"].fillna(0)

# %%
df_test["BsmtUnfSF"] = df_test["BsmtUnfSF"].fillna(0)

# %%
df_test["Exterior2nd"] = df_test["Exterior2nd"].fillna("VinylSd")

# %%
df_test["BsmtFinSF1"] = df_test["BsmtFinSF1"].fillna(0)

# %%
for col in df_num:
    plt.figure(figsize=(8, 4)) 
    sns.boxplot(data=df_num, x=col) 
    plt.title(f'Boxplot de {col}') 
    plt.show()

# %%
correlation_matrix = df_num.corr()
correlation_matrix

# %%
saleprice_cond = df["SalePrice"] < 500000
df = df[saleprice_cond]

# %%
grliv_cond = df["GrLivArea"] < 4000
df = df[grliv_cond]

# %%
lotfr_cond = df["LotFrontage"] < 200
lotfr_cond2 = df["LotFrontage"] > 25
df = df[lotfr_cond]
df = df[lotfr_cond2]

# %%
lot_ar = df["LotArea"] < 50000
df = df[lot_ar]

# %%
sns.boxplot(data=df, x="LotArea")

# %%
masvnr_cond = df["MasVnrArea"] < 800
df = df[masvnr_cond]

# %%
bsmt_cond = df["BsmtFinSF1"] < 3000
df = df[bsmt_cond]

# %%
totalbsmt_cond = df["TotalBsmtSF"] < 3000
df = df[totalbsmt_cond]

# %%
sf_cond = df["1stFlrSF"] < 3000
df = df[sf_cond]

# %%
wood_cond = df["WoodDeckSF"] < 600
df = df[wood_cond]

# %%
porch_cond = df["OpenPorchSF"] < 300
df = df[porch_cond]

# %%
df_lista_colunas = df.columns.to_list()
print(df_lista_colunas)

# %%
y = df["SalePrice"]

# %%
features = df.columns.to_list()
cat_features = df.select_dtypes(include=["object"])
list_cat_features = cat_features.columns.to_list()
X = df[features]
X = X.drop(columns=["SalePrice"])
print(features)

# %%
features_df_test = df_test.columns.to_list()
cat_features_test = df_test.select_dtypes(include=["object"])
list_cat_features_test = cat_features_test.columns.to_list()
X_df_test = df_test[features_df_test]

# %%
onehot = encoding.OneHotEncoder(variables=list_cat_features)
onehot.fit(X)
X = onehot.transform(X)

# %%
X

# %%
onehot_test = encoding.OneHotEncoder(variables=list_cat_features_test)
onehot_test.fit(X_df_test)
X_df_test = onehot.transform(X_df_test)

# %%
X_df_test

# %%
from sklearn.model_selection import train_test_split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42)

# %%
from sklearn.metrics import mean_absolute_error, r2_score

# %%
from sklearn.linear_model import LinearRegression

# %%
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# %%
y_pred = model_lr.predict(X_test)

# %%
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, R2SCORE: {r2}')

# %%
import matplotlib.pyplot as plt

# %%
fig, ax = plt.subplots(ncols=1,figsize=(10,5))

ax.scatter(y_test/100000,y_pred/100000)
ax.plot([0,700000],[0,700000],'--r')
ax.set(xlim=(0, 7),ylim=(0, 7))
ax.set_xlabel('Real')
ax.set_ylabel('Previsão')

plt.show()

# %%
from sklearn.tree import DecisionTreeRegressor

# %%
model_dtr = DecisionTreeRegressor(random_state=42)
model_dtr.fit(X_train,y_train)

# %%
y_pred = model_dtr.predict(X_test)

# %%
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, R2Score: {r2}')

# %%
pd.DataFrame({'y_test': y_test,
            'y_pred': y_pred,
             'abs_err': abs(y_pred-y_test)}).describe()

# %%
fig, ax = plt.subplots(ncols=1,figsize=(10,5))

ax.scatter(y_test/100000,y_pred/100000)
ax.plot([0,700000],[0,700000],'--r')
ax.set(xlim=(0, 7),ylim=(0, 7))
ax.set_xlabel('Real')
ax.set_ylabel('Previsão')

plt.show()

# %%
from sklearn.linear_model import Ridge

# %%
model_rdg = Ridge(alpha=1.0)
model_rdg.fit(X_train, y_train)

# %%
y_pred = model_rdg.predict(X_test)

# %%
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, R2Score: {r2}')

# %%
fig, ax = plt.subplots(ncols=1,figsize=(10,5))

ax.scatter(y_test/100000,y_pred/100000)
ax.plot([0,700000],[0,700000],'--r')
ax.set(xlim=(0, 7),ylim=(0, 7))
ax.set_xlabel('Real')
ax.set_ylabel('Previsão')

plt.show()

# %%
from sklearn.linear_model import Lasso

# %%
model_ls = Lasso(alpha=0.1)
model_ls.fit(X_train, y_train)

# %%
y_pred = model_ls.predict(X_test)

# %%
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MAE: {mae}, R2Score: {r2}')

# %%
fig, ax = plt.subplots(ncols=1,figsize=(10,5))

ax.scatter(y_test/100000,y_pred/100000)
ax.plot([0,700000],[0,700000],'--r')
ax.set(xlim=(0, 7),ylim=(0, 7))
ax.set_xlabel('Real')
ax.set_ylabel('Previsão')

plt.show()

# %%
test_predictions = model_rdg.predict(X_df_test)

# %%
output = pd.DataFrame({
    "Id": df_test1["Id"],
    "SalePrice": test_predictions
})

output.set_index('Id', inplace=True)
output.to_csv('sample_submission.csv')
