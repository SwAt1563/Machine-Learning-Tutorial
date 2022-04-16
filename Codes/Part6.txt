// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563


from sklearn.feature_selection import SelectKBest, mutual_info_classif  # FOR GET THE INFORMATION GAIN
from sklearn.metrics import accuracy_score, recall_score, \
    precision_score, f1_score, confusion_matrix   # FOR PRINT THE SCORES OF THE MODEL
from sklearn.model_selection import train_test_split  # FOR SPLIT THE DATA TO TRAINING AND TESTING DATA
from sklearn.naive_bayes import GaussianNB  # FOR CREATE NAIVE BIAS MODEL
from sklearn.neural_network import MLPClassifier  # FOR CREATE NEURAL NETWORK MODEL
from sklearn import tree  # FOR CREATE THE DECISION TREE MODEL
import pandas as pa  # FOR READ THE CSV FILE
import numpy as np  # FOR MATH
from matplotlib import pyplot as plt  # FOR PLOT THE INFORMATION GAIN

# FEATURES LABELS
features_labels = ["following", "followers", "actions", "is_retweet", "location"]


# FOR RETURN THE FEATURES VECTORS AND THE CLASSIFIERS
def get_x_y(csv_file):
    x = csv_file[features_labels]
    y = csv_file.Type
    return x, y



# FOR CREATE DECISION TREE MODEL
def tree_model(x_train, y_train):
    tree_model = tree.DecisionTreeClassifier()
    tree_model.fit(x_train, y_train)
    return tree_model


# FOR CREATE NEURAL NETWORK MODEL
def network_model(x_train, y_train):
    network_model = MLPClassifier()
    network_model.fit(x_train, y_train)
    return network_model


# FOR CREATE NAIVE BIAS MODEL
def naive_bias_model(x_train, y_train):
    naive_bias_model = GaussianNB()
    naive_bias_model.fit(x_train, y_train)
    return naive_bias_model






# FOR READ THE DATA CSV FILE
# THEN MAKE TRAINING AND TESTING FILE
def read_csv(file_path):
    # FOR READ THE CSV FILE
    df = pa.read_csv(file_path)

    # FOR SELECT THE NEEDED COLUMNS
    df_GF = df[["following", "followers", "actions", "is_retweet", "location", "Type"]]

    # FOR MAKE COPY OF THE CSV FILE
    df_update = pa.DataFrame.copy(df_GF)


    # FOR SET FORMAT AS WE LIKE
    df_update['Type'] = df_update['Type'].apply(lambda x: 1 if x == "Quality" else 0)



    # FOR REPLACE THE LOCATION WITH 1 IF EXIST AND 0 IF NULL
    df_update['location'] = df_update['location'].apply(lambda x: 1 if not pa.isnull(x) else 0)

    # FOR REPLACE THE NULLS WITH ZEROS
    df_update['actions'] = df_update['actions'].replace(np.nan, 0)
    df_update['following'] = df_update['following'].replace(np.nan, 0)
    df_update['followers'] = df_update['followers'].replace(np.nan, 0)
    df_update['is_retweet'] = df_update['is_retweet'].replace(np.nan, 0)


    # FOR SPLIT THE DATA INTO FEATURES VECTORS AND CLASSIFIERS
    features_vectors = df_update.drop('Type', axis=1)
    classifiers = df_update.Type

    # FOR SPLIT THE DATA
    x_train, x_test, y_train, y_test = train_test_split(features_vectors, classifiers, test_size=0.2, random_state=10)

    # FOR MAKE CSV TRAINING FILE
    train_csv_file = pa.DataFrame(x_train)
    train_csv_file["Type"] = y_train

    # FOR MAKE CSV TESTING FILE
    test_csv_file = pa.DataFrame(x_test)
    test_csv_file["Type"] = y_test

    return train_csv_file, test_csv_file



# FOR SHOW THE INFORMATION GAIN
def show_IG_for_features(x_train, y_train):
    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from data
    fs.fit(x_train, y_train)

    for i in range(len(fs.scores_)):
        print('Feature[%d]: %s = %f' % (i, fs.feature_names_in_[i], fs.scores_[i]))

    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.show()


# FOR GET THE CLASSIFICATION MODEL RESULT
# ADN CALCULATE THE ACCURACY
# AND PRECISION AND RECALL AND F1 SCORES
def cal_scores(model, x, y_true):

    y_pred = model.predict(x)

    acc = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return acc, recall, precision, f1, y_pred


# FOR PRINT THE ACCURACY AND PRECISION AND RECALL AND F1 SCORES
# IN ADDITION TO CONFUSION MATRIX
def print_score(model, x, y_true):
    acc, recall, precision, f1, y_pred = cal_scores(model, x, y_true)

    print("Accuracy: ", np.round(acc, 2))
    print("Recall: ", np.round(recall, 2))
    print("Precision: ", np.round(precision, 2))
    print("F1: ", np.round(f1, 2))
    print("confusion_matrix: \n", confusion_matrix(y_true, y_pred))


train, test = read_csv("train.csv")
model = naive_bias_model(*get_x_y(train))
show_IG_for_features(*get_x_y(train))
print_score(model, *get_x_y(test))

