// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563


from sklearn.feature_selection import SelectKBest, mutual_info_classif  # FOR GET THE INFORMATION GAIN
from sklearn.model_selection import train_test_split  # FOR SPLIT THE DATA
from sklearn.preprocessing import LabelEncoder  # FOR REFORMAT THE DATA TO FEATURES VECTORS
import pandas as pa  # FOR READ THE CSV FILES
from matplotlib import pyplot as plt  # FOR PLOT THE INFORMATION GAIN

# THE COLUMNS IN THE CSV FILE THAT WE NEED
features_labels = ["Time", "Weather", "Car", "Holiday", "Travel"]


# FOR PRINT THE PERCENTAGE OF THE TEST DATA
def data_percentage(x_train, x_test):
    length_x_train = len(x_train)
    length_x_test = len(x_test)
    print('size of test dataset = {0}, size of training data = {1}, percentage(test/test+train) = {2}%'.format(
        length_x_test,
        length_x_train,
        length_x_test * 100 / (length_x_test + length_x_train))
    )

# FOR PRINT THE INFORMATION OF THE CSV FILE
def print_travel_info(csv_file):

    unavailable = csv_file[csv_file.Travel == "unavailable"]
    No_unavailable = unavailable.shape[0]
    available = csv_file[csv_file.Travel == "available"]
    No_available = available.shape[0]
    Per_unavailable = No_unavailable / (No_unavailable + No_available)

    print(csv_file.info())
    print('unavailable = {}, available = {} , Percentage of unavailable = {} %'.format(No_unavailable, No_available,
                                                                                       Per_unavailable * 100))

# FOR PRINT THE INFORMATION GAIN FOR EACH FEATURE
def show_IG_for_features(x, y):

    # configure to select all features
    fs = SelectKBest(score_func=mutual_info_classif, k='all')
    # learn relationship from data
    fs.fit(x, y)

    for i in range(len(fs.scores_)):
        print('Feature[%d]: %s = %f' % (i, fs.feature_names_in_[i], fs.scores_[i]))

    plt.bar([i for i in range(len(fs.scores_))], fs.scores_)
    plt.show()

# FOR PLOT THE INFORMATION GAIN FOR EACH FEATURE
def plot_information_gain(features_vector, x, y):

    importances = mutual_info_classif(x, y)
    feat_importances = pa.Series(importances, features_vector.columns[0:len(features_vector.columns)])
    feat_importances.plot(kind='barh', color='green')
    plt.show()

# FOR READ THE DATA CSV FILE
# THEN MAKE TRAINING AND TESTING FILE
def read_csv(file_path, new_train_path, new_test_path):


    # FOR READ THE CSV FILE
    df = pa.read_csv(file_path)

    # FOR REPLACE THE NULL VALUES WITH 0
    df['not important'] = df['not important'].apply(lambda x: 0 if pa.isnull(x) else x)

    # FOR SELECT THE NEEDED COLUMNS
    df_GF = df[features_labels]

    # FOR MAKE COPY OF THE CSV FILE
    df_update = pa.DataFrame.copy(df_GF)

    # FOR MAKE NUMERIC FORMAT FOR FEATURES VECTORS
    le = LabelEncoder()

    # FOR REFORMAT THE DATA TO DIGITS
    df_update['Time'] = le.fit_transform(df_update['Time'])
    df_update['Weather'] = le.fit_transform(df_update['Weather'])
    df_update['Car'] = le.fit_transform(df_update['Car'])
    df_update['Holiday'] = le.fit_transform(df_update['Holiday'])

    # FOR SET FORMAT AS WE LIKE
    df_update['Travel'] = df_update['Travel'].apply(lambda x: 1 if x == "available" else 0)

    # FOR SPLIT THE DATA INTO FEATURES VECTORS AND CLASSIFIERS
    features_vectors = df_update.drop('Travel', axis=1)
    classifiers = df_update.Travel

    # FOR SPLIT THE DATA
    x_train, x_test, y_train, y_test = train_test_split(features_vectors, classifiers, test_size=0.2, random_state=10)
    
    # FOR MAKE CSV TRAINING FILE
    train_csv_file = pa.DataFrame(x_train)
    train_csv_file["Travel"] = y_train

    # FOR MAKE CSV TESTING FILE
    test_csv_file = pa.DataFrame(x_test)
    test_csv_file["Travel"] = y_test

    # FOR SAVE THE FILES WITHOUT INDEX COLUMN
    train_csv_file.to_csv(new_train_path, index=False)
    test_csv_file.to_csv(new_test_path, index=False)







