import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

#from forestlayer_master_bis.forestlayer_ter.datasets import uci_letter


def str_to_int(dataf):
    """
        Turns categorical str data to int data.
        -----------------------------------
        input : pd.dataframe
        -----------------------------------
        output : pd.dataframe
        """

    df = dataf.copy()
    categorical = list()

    for column in df.columns:  # Loop over df columns
        if isinstance(df[column][0], str):  # If the first element of the column in a str
            
            categorical.append(True)

            switch_dict = {}
            for i, string in enumerate(df[column].unique()):  # Builds a dictionary {column_name_i : i}
                switch_dict[string] = i

            df[column] = df[column].apply(lambda x: switch_dict[x])  # name_i -> i for name_i in column

        else:
            categorical.append(False)

    return df, categorical


def load_protein(data_path=""):
    #load data
    data = pd.read_csv(data_path + 'CASP.csv')

    # panda to np
    data = data.values

    # seperate X and y
    X, y = data[:, 1:], data[:, 1]

    # dtype of X from int to float
    X = X.astype(np.float32)

    # all features are numerical
    is_categorical = [False]*9

    return X, y, is_categorical


#def load_letter():
#    """
#    load letter dataset
#    --------------------
#    returns (x_train, y_train, x_test, y_test)
#    """
#    return uci_letter.load_data()


#def download_airbnb(user_path, unzip=True):
#    """
#    Download kaggle berlin-airbnb dataset at specified path.
#    
#    This function requires to have a Kaggle account and an active API token located in a kaggle folder.
#    """
#
#    kaggle.api.authenticate()
#    kaggle.api.dataset_download_files('brittabettendorf/berlin-airbnb-data',
#                                      path=user_path, unzip=unzip)
#
#    print("Files saved at ", user_path)
#
#    return None


def load_airbnb(data_path, test_size=0.3, seed=15):
    """
    Loads the airbnb dataset
    """

    calendar = pd.read_csv(data_path + 'calendar.csv.zip')
    listings = pd.read_csv(data_path + 'listings.csv')

    listings = listings.drop(['name', 'host_name', 'last_review', 'license'], axis=1)  # Drop several features

    calendar = calendar.rename(columns={"listing_id": "id"})
    calendar = calendar.drop_duplicates(subset=['id', 'price'], keep='first')

    # Merge the two csv files on the 'id' column
    data = pd.merge(listings.drop(["price"], axis=1), calendar.drop(["available", "date", "adjusted_price"], axis=1),
                    on='id').drop("host_id", axis=1)

    # Retrieve float price from string like "$,250.00"
    data['price'] = data['price'].apply(lambda x: float(re.findall(r'[\d]*[.][\d]+', x)[0]))
    

    # drop na value on two columns, only a few samples
    #data = data.dropna(axis=0, how='any', subset=["minimum_nights_y", 'maximum_nights'], inplace=False)
    data = data.fillna(0)  # Fill other with 0
    
    # retrieve price and delete it in X
    Y = data["price"].values
    data = data.drop(["price", 'id'], axis=1)

    # Turns str features (i.e neighborhood, ...) to categorical features
    data, is_categorical = str_to_int(data)  

    X = data.values

    return X, Y, is_categorical


def load_adult(data_path):
    """
    Loads the adult dataset
    """
    # load data
    data = pd.read_csv(data_path + 'income_evaluation.csv')

    # transform categorical variables to int
    data, is_categorical = str_to_int(data)

    # panda to np
    data = data.values

    # seperate X and y
    X, y = data[:, 0:14], data[:, 14]
    is_categorical = is_categorical[:-1]

    # dtype of X from int to float
    X = X.astype(np.float32)

    return X, y, is_categorical


def load_housing(data_path, test_size=0.3, seed=11):
    """
    Loads the housing dataset
    """
    data = pd.read_csv(data_path + "train.csv")
    data = data.dropna(axis=1)

    data = str_to_int(data)

    X = data.drop(["SalePrice"], axis=1).values
    Y = data["SalePrice"].values

    Y = Y

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return x_train, y_train, x_test, y_test


def load_higgs(path):
    """
    Loads Higgs dataset
    """
    df = pd.read_csv(path+'higgs_vsmall.csv.zip', header=None)

    X = df.values
    y = X[:,-1].astype(np.int64)
    X = X[:, :-1]

    is_categorical = [False]*28
    is_categorical[20] = True

    return X, y, is_categorical


    df = pd.read_csv('../datasets/higgs/HIGGS.csv.gz', header=None)
    df.columns = ['x' + str(i) for i in range(df.shape[1])]
    num_col = list(df.drop(['x0', 'x21'], axis=1).columns)
    cat_col = ['x21']
    is_categorical = [True if name in cat_col else False for name in df.drop(['x0'], axis=1).columns]
    label_col = 'x0'

    def fe(x):
        if x > 2:
            return 1
        elif x > 1:
            return 0
        else:
            return 2

    df.x21 = df.x21.apply(fe)

    # Fill NaN with something better?
    df.fillna(0, inplace=True)

    X = df[num_col + cat_col].to_numpy()
    y = df[label_col].to_numpy().astype(np.int64)

    X, _, y, _ = train_test_split(X, y, test_size=0.95, stratify=y)

    print(X.shape)
    print(y.shape)
    print(np.append(X, np.expand_dims(y, 1), 1).shape)

    np.savetxt("../datasets/higgs/higgs_vsmall.csv", np.append(X, np.expand_dims(y, 1), 1), delimiter=",")
    print('done.')
    exit()
    
    return X, y, is_categorical

    # Importing data (only 200 000 rows)
    data = pd.read_csv(data_path + "higgs_trunc.csv", nrows=nrows)

    # csv to numpy
    array = data.values
    X, Y = array[:, 1:], array[:, 0].astype(int)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return x_train, y_train, x_test, y_test


def load_yeast(data_path="", test_size=0.3, seed=7):
    """
    Loads yeast dataset
    """
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
    names = ['Sequence Name', 'mcg', 'gvh', 'alm', 'mit', 'erl', 'pox', 'vac', 'nuc', 'class']

    if data_path == "":
        dataset = pd.read_csv(url, names=names, delim_whitespace=True)
    else:
        dataset = pd.read_csv(data_path + "yeast.data", names=names, delim_whitespace=True)
    # csv to numpy
    array = dataset.values
    X = array[:, 1:9]
    Y = array[:, 9]

    # Label from str to int
    classes = dataset["class"].unique()
    dic = {classes[i]: i for i in range(len(classes))}
    for i in range(len(Y)):
        Y[i] = dic[Y[i]]

    Y = np.array(Y, dtype=int)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return x_train, y_train, x_test, y_test

def load_covertype(data_path):
    """
    Loads the adult dataset
    """
    # load data
    data = pd.read_csv(data_path + 'covtype.data.zip', header=None)

    # transform categorical variables to int
    #data, is_categorical = str_to_int(data)

    # panda to np
    data = data.values

    # seperate X and y
    X, y = data[:, 0:-1], data[:, -1]
    # 10 numerical varaibles + 4 area binary variables + 40 soil type binary varaibles
    is_categorical = [False]*10 + [True]*44

    # dtype of X from int to float
    X = X.astype(np.float32)

    # substract 1 to y: move forest cover type encofing from [1..7] -> [0..6]
    y = y - 1 

    return X, y, is_categorical

def load_bank(data_path):
    """
    Loads the adult dataset
    """
    # load data
    data = pd.read_csv(data_path + 'bank-full.csv', sep=';')

    # transform categorical variables to int
    data, is_categorical = str_to_int(data)

    # panda to np
    data = data.values

    # seperate X and y
    X, y = data[:, 0:-1], data[:, -1]
    is_categorical = is_categorical[:-1]

    # dtype of X from int to float
    X = X.astype(np.float32)

    return X, y, is_categorical

def load_volkert(data_path):
    # load data
    data_X = pd.read_csv(data_path + 'volkert_train.data', sep=' ', header=None)
    data_y = pd.read_csv(data_path + 'volkert_train.solution', sep=' ', header=None)

    # there is a NaN column at the end of the data frame as every line of the file ends with a space and sep=' '
    # we remove it
    data_X = data_X.dropna(axis=1, how='all')
    data_y = data_y.dropna(axis=1, how='all')


    # some features contain the same value (0) for all instances 
    # we drop them as we want to compare our results to the paper
    # "Gowthami Somepalli et al. SAINT: Improved Neural Networks for Tabular Data via Row Attention and Contrastive Pre-Training."
    # and we believe that the authors did the same
    nunique = data_X.nunique()
    cols_to_drop = nunique[nunique == 1].index
    data_X = data_X.drop(cols_to_drop, axis=1)

    # panda to np
    X = data_X.values
    y = data_y.values

    # transform y from one hot vectors to class labels
    y = np.argmax(y, axis=1)

    
    # all features are numerical
    is_categorical = [False] * 147

    return X, y, is_categorical

def load_heloc(data_path):
    # load data
    data = pd.read_csv(data_path + 'heloc_preprocessed.csv')

    # transform categorical variables to int
    #data, is_categorical = str_to_int(data)
    # remove empty columns
    #data = data[data.iloc[:,2]!=-9]
    #data = data.replace(-7, 0)
    #data = data.replace(-8, 0)
    #data = data.replace(-9, 0)

    # properly encode categorical featues
    le = LabelEncoder()
    data['MaxDelq2PublicRecLast12M'] = le.fit_transform(data['MaxDelq2PublicRecLast12M'].values)
    data['MaxDelqEver'] = le.fit_transform(data['MaxDelqEver'].values)

    # panda to np
    X = data.drop('RiskPerformance', axis=1).values
    y = data['RiskPerformance'].values.astype(np.int64)
    is_categorical = [False]*23
    is_categorical[data.columns.get_loc('MaxDelq2PublicRecLast12M')] = True
    is_categorical[data.columns.get_loc('MaxDelqEver')] = True
    return X, y, is_categorical

def load_blastchar(path):
    # load data
    data = pd.read_csv(path + 'WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # transform categorical variables to int
    data, is_categorical = str_to_int(data)

    # panda to np
    X = data.values
    y = X[:,-1]
    X = X[:,:-1]
    is_categorical = is_categorical[:-1]
    return X.astype(np.float32), y.astype(np.int64), is_categorical

def load_diamonds(path):
    # load data
    df = pd.read_csv(path + 'diamonds.csv', header=None)
    df.columns = ['x' + str(i) for i in range(df.shape[1])]
    
    # panda to np
    X = df.drop('x9', axis=1).values
    y = df['x9'].values
    is_categorical = [False, True, True, True, False, False, False, False, False]

    return X, y, is_categorical
