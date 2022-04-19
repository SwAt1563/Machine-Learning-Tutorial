// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563


from sklearn.model_selection import train_test_split  # FOR SPLIT THE DATA
import pandas as pa  # FOR READ THE CSV FILES
import string  # FOR GET PUNCTUATION
import re  # FOR CLEAN TEXT BY REGEX
import nltk  # FOR PROCESSING THE TEXT
from nltk import WordNetLemmatizer  # FOR GET THE ROOT OF THE WORD
from nltk.tokenize import word_tokenize  # FOR DIVIDE THE TEXT INTO WORDS
from nltk.corpus import stopwords  # FOR GET THE UNNECESSARY WORDS AND CHARACTERS


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


# FOR CONVERT LIST OF WORDS TO ITS ROOTS
def get_roots(tweet):
    # FOR GET THE TYPE OF WORDS
    check_types = nltk.pos_tag(tweet)

    # THIS CLASS FOR GET THE ROOTS
    root = WordNetLemmatizer()

    root_words = set()
    for word in check_types:
        type = word[1][0]
        if type == 'N':  # IF THE WORDS IS NOUN
            root_words.add(root.lemmatize(word[0], pos="n"))
        elif type == "V":  # IF THE WORDS IS VERB
            root_words.add(root.lemmatize(word[0], pos="v"))
        elif type == "A" or type == "R":  # IF THE WORDS IS ADJECTIVE
            root_words.add(root.lemmatize(word[0], pos="a"))
        elif type == "S":  # IF THE WORDS IS ADVERB
            root_words.add(root.lemmatize(word[0], pos="s"))
        else:
            root_words.add(root.lemmatize(word[0]))

    return list(root_words)


# FOR CONVERT THE TWEET TO LIST OF ROOT WORDS
# AFTER CLEAN THE TWEET
def convert_tweet_to_clean_words(tweet):
    list_without_stopwords = get_list_without_stopwords(tweet)
    clean_list = clean_text(list_without_stopwords)
    tweet_to_roots = get_roots(clean_list)

    return " ".join(tweet_to_roots)


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
    df_update['Type'] = df_update['Type'].apply(lambda x: 1 if x == "ham" else 0)

    # FOR SPLIT THE DATA INTO FEATURES VECTORS AND CLASSIFIERS
    features_vectors = df_update.drop('Type', axis=1)
    classifiers = df_update.Type

    # FOR SPLIT THE DATA
    x_train, x_test, y_train, y_test = train_test_split(features_vectors, classifiers, test_size=0.2, random_state=5)

    # FOR MAKE CSV TRAINING FILE
    train_csv_file = pa.DataFrame(x_train)
    train_csv_file['Roots'] = train_csv_file['Tweet'].apply(lambda x: convert_tweet_to_clean_words(x))
    train_csv_file["Type"] = y_train

    # FOR MAKE CSV TESTING FILE
    test_csv_file = pa.DataFrame(x_test)
    test_csv_file['Roots'] = test_csv_file['Tweet'].apply(lambda x: convert_tweet_to_clean_words(x))
    test_csv_file["Type"] = y_test

    return train_csv_file, test_csv_file


# FOR SAVE THE SPAM AND HAM WORDS IN DICTIONARIES WITH THEIR FREQUENCIES
def categorize_words(csv_data):
    spam_words = {}
    ham_words = {}


    for line in csv_data['Roots'][csv_data['Type'] == 0]:
        for word in str(line).split(" "):
            if spam_words.get(word) is not None:
                spam_words[word] += 1
            else:
                spam_words[word] = 1


    for line in csv_data['Roots'][csv_data['Type'] == 1]:
        for word in str(line).split(" "):
            if ham_words.get(word) is not None:
                ham_words[word] += 1
            else:
                ham_words[word] = 1

    return spam_words, ham_words


# FOR CHECK IF THE TWEET IS SPAM OR HAM
def bias_feature(roots, spam_words, ham_words):
    ham = 0
    spam = 0

    for word in roots.split(" "):
        if spam_words.get(word) is not None:
            spam += spam_words[word]
        if ham_words.get(word) is not None:
            ham += ham_words[word]

    if ham > spam:
        return 1

    return 0


train, test = read_csv("spam.csv")
spam, ham = categorize_words(train)

correct_answers = 0
wrong_answers = 0

roots_column = list(test['Roots'])
types_column = list(test['Type'])

for i in range(len(roots_column)):
    if bias_feature(roots_column[i], spam, ham) == types_column[i]:
        correct_answers += 1
    else:
        wrong_answers += 1

print("Number of correct answers: {}\nNumber of wrong answers: {}".format(correct_answers, wrong_answers))
