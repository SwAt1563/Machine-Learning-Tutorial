// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563


from sklearn.model_selection import train_test_split  # FOR SPLIT THE DATA
import pandas as pa  # FOR READ THE CSV FILES
import string  # FOR GET PUNCTUATION
from nltk.corpus import stopwords  # FOR GET THE UNNECESSARY WORDS AND CHARACTERS
from sklearn.naive_bayes import MultinomialNB  # FOR CREATE NAIVE BIAS MODEL
from sklearn.feature_extraction.text import CountVectorizer  # FOR MAKE MULT. FEATURES

# FOR TEACH
test_tweet = "follow qutaiba qutaiba olayyan swat1563 you should follow follow follow"
c = CountVectorizer()
print(c.fit_transform([test_tweet]))

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
    df_update['Type'] = df_update['Type'].apply(lambda x: 0 if x == "ham" else 1)

    # FOR SPLIT THE DATA INTO FEATURES VECTORS AND CLASSIFIERS
    features_vectors = df_update.drop('Type', axis=1)
    classifiers = df_update.Type

    # FOR SPLIT THE DATA
    x_train, x_test, y_train, y_test = train_test_split(features_vectors, classifiers, test_size=0.2, random_state=5)

    # FOR MAKE CSV TRAINING FILE
    train_csv_file = pa.DataFrame(x_train)
    train_csv_file["Type"] = y_train

    # FOR MAKE CSV TESTING FILE
    test_csv_file = pa.DataFrame(x_test)
    test_csv_file["Type"] = y_test

    return train_csv_file, test_csv_file




# FOR CLEANING THE TWEET FROM PUNCTUATIONS AND STOPWORDS
def text_processing(tweet):

    tweet = [char for char in tweet if char not in string.punctuation]
    tweet = ''.join(tweet)
    tweet = [word for word in tweet.split() if word.lower() not in stopwords.words("english")]

    return tweet




# FOR CREATE NAIVE BIAS MODEL WITH MULTIPLE FEATURES
def create_naive_bias_mul_features_model(csv_training_file):

    csv_file = csv_training_file[["Tweet", "Type"]]

    # FIND THE FEATURES DEPEND ON THE TWEET WORDS AFTER CLEAN IT
    count_vector = CountVectorizer(analyzer=text_processing)

    # MAKE MULTI. FEATURES DEPEND ON THE WORDS AND ITS FREQUENCIES
    training_data = count_vector.fit_transform(csv_file['Tweet'])

    # FOR CREATE NAIVE BIAS MODEL
    naive_bias_text_model = MultinomialNB()
    # FOR LET THE MODEL TRAIN ON THE DATA
    naive_bias_text_model.fit(training_data, csv_file['Type'])

    return naive_bias_text_model, count_vector

def print_score(model, vectorizer, test_file):
    data = vectorizer.transform(test_file['Tweet'])
    scores = model.score(data, test_file['Type'])
    print(scores)

train_file, test_file = read_csv("spam.csv")
model, countVectorizer = create_naive_bias_mul_features_model(train_file)
print_score(model, countVectorizer, test_file)