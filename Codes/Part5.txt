// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563


from nltk.corpus import stopwords  # FOR GET THE STOPWORDS
from sklearn.feature_selection import SelectKBest, mutual_info_classif  # FOR GET THE INFORMATION GAIN
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix  # FOR PRINT THE SCORES OF THE MODEL
from sklearn.model_selection import train_test_split  # FOR SPLIT THE DATA TO TRAINING AND TESTING DATA
from sklearn.naive_bayes import GaussianNB  # FOR CREATE NAIVE BIAS MODEL
from sklearn.neural_network import MLPClassifier  # FOR CREATE NEURAL NETWORK MODEL
from sklearn import tree  # FOR CREATE THE DECISION TREE MODEL
import pandas as pa  # FOR READ THE CSV FILE
import re  # FOR REGEX
import string  # FOR GET THE PUNCTUATIONS
from nltk.tokenize import word_tokenize  # FOR DIVIDE THE TWEET TO LIST OF WORDS
from better_profanity import profanity  # FOR KNOW IF THE TWEET HAS SWEAR WORDS OR NOT
import numpy as np  # FOR MATH
from matplotlib import pyplot as plt  # FOR PLOT THE INFORMATION GAIN


# FEATURES THAT WE WANT EXTRACTION FROM THE TWEET
text_features_labels = ["length_before", "num_words_before", "length_after", "num_words_after", "num_urls",
                        "num_hashtags", "num_minations", "swear"]


# FOR RETURN THE FEATURES VECTORS AND THE CLASSIFIERS
def get_x_y(csv_file):
    x = csv_file[text_features_labels]
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


# FOR REMOVE THE STOPWORDS FROM THE TWEET
# AND RETURN LIST OF WORDS
def get_list_without_stopwords(tweet):
    # FOR DIVIDE THE TWEET INTO LIST OF WORDS AFTER CHANGE IT TO LOWERCASE
    divide_tweet_to_words = list(word_tokenize(str(tweet).lower()))

    # FOR GET THE STOPWORDS OF THE ENGLISH LANGUAGE
    stop_words = set(stopwords.words("english"))

    # FOR REMOVE THE STOPWORDS FROM THE TWEET
    remaining_words = [word for word in divide_tweet_to_words if word not in stop_words]

    return remaining_words


# FOR CLEAN THE LIST OF WORDS
def clean_text(list_words):
    # CLEAN THE TEXT FROM URLs AND NUMBERS
    regex1 = re.compile(r'^http[s]*|[0-9]')
    filtered1 = [i for i in list_words if not regex1.search(i)]

    # CLEAN THE TEXT FROM THE PUNCTUATIONS
    regex2 = re.compile(r'[%s]' % re.escape(string.punctuation))
    filtered2 = [i for i in filtered1 if not regex2.search(i)]

    return filtered2



# FOR EXTRACTION THE FEATURES FROM THE TWEET
def preprocessing(tweet):
    tweet = str(tweet)

    # create length_before feature
    length_tweet_before = len(tweet)

    # create num_words_before feature
    num_of_words_before = len(tweet.split(" "))

    # create num_urls feature
    url = re.findall("(http[s]:\/\/)?([\w-]+\.)+([a-z]{2,5})(\/+\w+)?", tweet)
    num_of_urls = len(url)

    # create swear feature
    contain_swear_words = profanity.contains_profanity(tweet)
    if contain_swear_words:
        contain_swear_words = 1
    else:
        contain_swear_words = 0

    # for delete the stop words
    remaining_words_without_stop_words = get_list_without_stopwords(tweet)

    # create num_hashtags feature
    num_of_hashtags = remaining_words_without_stop_words.count("#")

    # create num_minations feature
    num_of_minations = remaining_words_without_stop_words.count("@")

    # for clean the tweet
    cleaning_tweet = clean_text(remaining_words_without_stop_words)

    # create num_words_after feature
    num_of_words_after = len(cleaning_tweet)

    # create length_after feature
    length_tweet_after = len("".join(cleaning_tweet))

    return length_tweet_before, length_tweet_after, num_of_words_before, num_of_words_after, num_of_urls, \
           num_of_hashtags, num_of_minations, contain_swear_words


# FOR READ THE DATA CSV FILE
# THEN MAKE TRAINING AND TESTING FILE
def read_csv(file_path):
    # FOR READ THE CSV FILE
    df = pa.read_csv(file_path)

    # FOR SELECT THE NEEDED COLUMNS
    df_GF = df[["Tweet", "Type"]]

    # FOR MAKE COPY OF THE CSV FILE
    df_update = pa.DataFrame.copy(df_GF)

    # FOR SET FORMAT AS WE LIKE
    df_update['Type'] = df_update['Type'].apply(lambda x: 1 if x == "Quality" else 0)

    # FOR GET THE FEATURES VECTORS FOR ALL TWEETS
    length_before_column = []
    length_after_column = []
    num_words_before_column = []
    num_words_after_column = []
    num_urls_column = []
    num_hashtags_column = []
    num_minations_column = []
    swear_column = []

    tweet_column = list(df_update["Tweet"])

    for tweet in tweet_column:
        lb, la, nwb, nwa, nu, nh, nm, s = preprocessing(tweet)
        length_before_column.append(lb)
        length_after_column.append(la)
        num_words_before_column.append(nwb)
        num_words_after_column.append(nwa)
        num_urls_column.append(nu)
        num_hashtags_column.append(nh)
        num_minations_column.append(nm)
        swear_column.append(s)

    new_csv = pa.DataFrame({"length_before": length_before_column, "num_words_before": num_words_before_column,
                            "length_after": length_after_column,
                            "num_words_after": num_words_after_column, "num_urls": num_urls_column,
                            "num_hashtags": num_hashtags_column,
                            "num_minations": num_minations_column,
                            "swear": swear_column, "Type": df_update['Type']})

    # FOR SPLIT THE DATA INTO FEATURES VECTORS AND CLASSIFIERS
    features_vectors = new_csv.drop('Type', axis=1)
    classifiers = new_csv.Type

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


train, test = read_csv("spam.csv")
model = tree_model(*get_x_y(train))
show_IG_for_features(*get_x_y(train))
print_score(model, *get_x_y(test))

