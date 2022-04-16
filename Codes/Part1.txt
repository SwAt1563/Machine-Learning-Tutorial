// THIS CODE WRITTEN BY QUTAIBA OLAYYAN
// GitHub: github.com/SwAt1563

from better_profanity import profanity  # FOR SWEAR WORDS
import string  # FOR GET PUNCTUATION
import re  # FOR CLEAN TEXT BY REGEX
import nltk  # FOR PROCESSING THE TEXT
from nltk import WordNetLemmatizer  # FOR GET THE ROOT OF THE WORD
from nltk.tokenize import word_tokenize  # FOR DIVIDE THE TEXT INTO WORDS
from nltk.corpus import stopwords  # FOR GET THE UNNECESSARY WORDS AND CHARACTERS

# TESTING TWEETS
tweet1 = "It's the everything else that's complicated. 100 #PESummit #PXpic.twitter.com/Jsv6BAFQMl"
tweet2 = "#jan Idiot Chelsea @Handler Diagnoses Trump 9With a 100 Disease https://t.co/k8PrqcWTRI https://t.co/dRN35xtSJZ"
tweet3 = "@playbingobash @ Gems are sparkling everywhere! in #BingoBash!!! http://bash.gg/1Y35AQ0"
tweet4 = "let me went to school plz, nice city, i'm going to Palestine, better for u need helping from me"
tweet5 = "You should follow SwAt1563 for this tutorial"


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

# FOR CHECK IF THE TWEET HAS SWEAR WORDS
def checkSwear(tweet):
    return profanity.contains_profanity(tweet)

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

