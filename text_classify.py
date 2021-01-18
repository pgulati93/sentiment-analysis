import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string
import random

# Create function to read data and tokenize text
def read_data(file, colspecs, columns):
    # Import train data from text to dataframe
    df = pd.read_fwf(file, colspecs=colspecs, header=None, names=columns)
    print("Text file converted to dataframe.")

    # Tokenize the text
    '''
    Tokenizing the text data converts it from a single string to a list of words
    and punctuation.
    '''
    df["tokens"] = df["text"].apply(word_tokenize)
    print("Text data tokenized.\n")

    return df

# Create function to normalize text
'''
Normalizing the text will involve:
- Removing special characters from words
- Lowercasing words
- Lemmatizing words by finding their root
- POS-tagging words to understand their grammatical position,
  and using this information to improve lemmatizing accuracy
'''

def normalize_text(word):
    lemmatizer = WordNetLemmatizer()
    normalized_sentence = []
    printable = set(string.printable)

    for word, tag in pos_tag(word):
        word = ''.join(filter(lambda x: x in printable, word))
        word = word.lower()

        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        pos = tag_dict.get(tag[0].upper(), wordnet.NOUN) # noun is the default

        word = lemmatizer.lemmatize(word,pos)

        stop_words = stopwords.words('english')
        if len(word) > 0 and word not in string.punctuation and word not in stop_words:
            normalized_sentence.append(word)

    return normalized_sentence

# Create a function to extract the list of frequent words
'''
This function creates a list of distinct words, organized by
frequency, with an option to limit the max selection of words.
Frequent words will be known as "features" in this code.
'''

def fdist_text(tokens_normalized, max=False):
    # Create list of all words in text
    words_list = []
    for l in tokens_normalized:
        words_list.extend(l)

    fdist = nltk.FreqDist(w.lower() for w in words_list)
    if max:
        fdist_list = list(fdist)[:max]
    else:
        fdist_list = list(fdist)
    print("Created list of {} most frequent words\n".format(len(fdist_list)))

    return fdist_list

# Create a function to check presence of feature words
'''
This function creates a list with a dictionary that detects the
presence of features (aka frequent words) in each text.
'''
def features_identify(text):
    text_lower = [w.lower() for w in text]
    # "set" creates a set of unique words in each text
    text_words = set(text_lower)
    features = {}
    for word in fdist_list:
        features[word] = (word in text_words)

    return features

# Create a function to create a tuple for more effecient analysis
'''
This function creates a tuple [(a,b)] where "a" is the text and
"b" is the sentiment.
'''
def tuple_df(text_df, sentiment_df):
    tupled_df = tuple(zip(list(text_df), list(sentiment_df)))

    return tupled_df

# Create a function to train the Naive Bayes classifier
def train_nb_classifier(df, train_val=False, test_val=False):
    tupled_df = tuple_df(df["tokens"], df["sentiment"])
    print("Created tuple of text and sentiment.")

    featuresets = [(features_identify(t),s) for (t,s) in tupled_df]
    random.shuffle(featuresets)
    print("Created features-presence sets.\n")

    train_val = int(len(featuresets)/2) if train_val == False else train_val
    test_val = int(len(featuresets)/2) if test_val == False else test_val
    train_set, test_set = featuresets[train_val:], featuresets[:test_val]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    print("Created NB classifier.")
    print("Classifier accuracy is: {}".format(accuracy))
    print(classifier.show_most_informative_features(5))
    print("\n")

    return classifier

# Create a function to test the Naive Bayes classifier
def test_nb_classifier(tokens):
    tokens_normalized = normalize_text(tokens)
    sentiment = classifier.classify(dict([t, True] for t in tokens_normalized))

    return sentiment

# Run the main function
if __name__ == "__main__":
    train_file = "training.txt" # specify training file name
    train_colspecs = [(0,1), (2,None)] # specify column widths
    train_columns = ["sentiment", "text"] # specify column names

    # Import data, convert to dataframe, and tokenize text
    train_df = read_data(train_file, train_colspecs, train_columns)

    # Normalize and clean up tokenized text
    train_df["tokens_normalized"] = train_df["tokens"].apply(normalize_text)
    print("Normalized and cleaned up text.\n")

    # Identify list of most frequent words
    fdist_list = fdist_text(train_df["tokens_normalized"])

    # Create classifier and report on accuracy
    classifier = train_nb_classifier(train_df)

    # Import test data, and test classifier against this data
    test_file = "testdata.txt"
    test_colspecs = [(0, None)] # specify column widths
    test_columns = ["text"] # specify column names

    test_df = read_data(test_file, test_colspecs, test_columns)

    test_df["sentiment"] = test_df["tokens"].apply(test_nb_classifier)
    print("Classified sentiment of test data.\n")

    # Create text file of test data with classified sentiments
    test_file_classified = "testdata_classified.txt"
    test_columns_classified = ["sentiment","text"]

    test_df.to_csv(test_file_classified,
                   sep=" ",
                   columns=test_columns_classified,
                   header=False,
                   index=False)
                   
    print("Created text file with test data and classified sentiment!")
