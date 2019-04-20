import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def format_data(df):
    temp = []
    idx = df.index
    for i in idx:
        temp.append([[item] for item in df.iloc[i]])
    temp = np.array(temp, dtype=float)
    return temp


def load_clean_data(csv_file):
    # Load all data
    data = pd.read_csv(csv_file)
    # Remove Unnecessary columns
    clean_data = data.drop(labels=['Unnamed: 0', 'Unnamed: 0.1','Unnamed: 0.1.1',
                                   'date', 'readings', 'sensors', 'time'],
                           axis=1)
    return clean_data


def shuffle_split_dataset(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42, shuffle=True)
    return X_train, y_train, X_test, y_test


def extract_categories(y):
    cats = []
    for element in y:
        if element not in cats:
            cats.append(element)
    return cats



def load_preprocess_data():
    csv_file1 = 'Dataset/Aruba/BDL_test2.csv'
    csv_file2 = 'Dataset/Aruba/BDL_train2.csv'
    # Create a clean dataset
    clean_data_1 = load_clean_data(csv_file1)
    clean_data_2 = load_clean_data(csv_file2)
    clean_data = pd.concat([clean_data_1, clean_data_2])
    # Shuffle Data
    clean_data_sampled = clean_data.sample(frac=1).reset_index(drop=True)
    return clean_data
