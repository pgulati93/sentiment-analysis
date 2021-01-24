import logging
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import random

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.info("Logging is setup")


def read_data(file, colspecs, columns):
    """ Create function to read data and tokenize text """

    # Import train data from text to dataframe
    df = pd.read_fwf(file, colspecs=colspecs, header=None, names=columns)
    logger.info("Text file converted to dataframe.")

    '''
    Tokenizing the text data converts it from a single string to a list of words
    and punctuation.
    '''
    df["tokens"] = df["text"].apply(word_tokenize)
    logger.info("Text data tokenized.\n")

    return df


def normalize_text(word):
    '''Normalizing the text will involve:
    - Removing special characters from words
    - Lowercasing words
    - Lemmatizing words by finding their root
    - POS-tagging words to understand their grammatical position,
      and using this information to improve lemmatizing accuracy
    '''

    lemmatizer = WordNetLemmatizer()
    normalized_sentence = []
    logger.infoable = set(string.logger.infoable)

    for word, tag in pos_tag(word):
        word = ''.join(filter(lambda x: x in logger.infoable, word))
        word = word.lower()

        tag_dict = {
            "J": wordnet.ADJ,
            "N": wordnet.NOUN,
            "V": wordnet.VERB,
            "R": wordnet.ADV,
        }
        pos = tag_dict.get(tag[0].upper(), wordnet.NOUN)  # noun is the default

        word = lemmatizer.lemmatize(word, pos)

        stop_words = stopwords.words('english')
        if len(word) > 0 and word not in string.punctuation and word not in stop_words:
            normalized_sentence.append(word)

    return normalized_sentence


def fdist_text(tokens_normalized, max=False):
    ''' Creates a list of distinct words, organized by
    frequency, with an option to limit the max selection of words.

    Frequent words will be known as "features" in this code.
    '''
    words_list = []
    for word in tokens_normalized:
        words_list.extend(word)

    fdist = nltk.FreqDist(w.lower() for w in words_list)

    fdist_list = list(fdist)
    if max:
        fdist_list = list(fdist)[:max]

    logger.info(f"Created list of {len(fdist_list)} most frequent words\n")

    return fdist_list


def features_identify(text):
    '''
    This function creates a list with a dictionary that detects the
    presence of features (aka frequent words) in each text.
    '''
    text_lower = [w.lower() for w in text]

    text_words = set(text_lower)  # create a set of unique words in each text
    features = {}
    for word in fdist_list:
        features[word] = (word in text_words)

    return features


def tuple_df(text_df, sentiment_df):
    ''' This function creates a tuple [(a,b)] where "a" is the text and
    "b" is the sentiment.

    Are these dfs? They look like pd.Series objects to me
    '''
    tupled_df = tuple(zip(list(text_df), list(sentiment_df)))

    return tupled_df


def train_nb_classifier(df, train_val=False, test_val=False):
    """ Create a function to train the Naive Bayes classifier """
    tupled_df = tuple_df(df["tokens"], df["sentiment"])
    logger.info("Created tuple of text and sentiment.")

    featuresets = [(features_identify(t), s) for (t, s) in tupled_df]
    random.shuffle(featuresets)
    logger.info("Created features-presence sets.\n")

    train_val = int(len(featuresets) / 2) if train_val == False else train_val
    test_val = int(len(featuresets) / 2) if test_val == False else test_val
    train_set, test_set = featuresets[train_val:], featuresets[:test_val]

    classifier = nltk.NaiveBayesClassifier.train(train_set)
    accuracy = nltk.classify.accuracy(classifier, test_set)
    logger.info("Created NB classifier.")
    logger.info("Classifier accuracy is: {accuracy}")
    logger.info(classifier.show_most_informative_features(5))
    logger.info("\n")

    return classifier


def test_nb_classifier(tokens):
    """ Create a function to test the Naive Bayes classifier """
    tokens_normalized = normalize_text(tokens)
    sentiment = classifier.classify(dict([t, True] for t in tokens_normalized))

    return sentiment


if __name__ == "__main__":

    train_file = "input_files/training.txt"  # specify training file name
    train_colspecs = [(0, 1), (2, None)]  # specify column widths
    train_columns = ["sentiment", "text"]  # specify column names

    # Import data, convert to dataframe, and tokenize text
    train_df = read_data(train_file, train_colspecs, train_columns)

    # Normalize and clean up tokenized text
    train_df["tokens_normalized"] = train_df["tokens"].apply(normalize_text)
    logger.info("Normalized and cleaned up text.\n")

    # Identify list of most frequent words
    fdist_list = fdist_text(train_df["tokens_normalized"])

    # Create classifier and report on accuracy
    classifier = train_nb_classifier(train_df)

    # Import test data, and test classifier against this data
    test_file = "input_files/testdata.txt"
    test_colspecs = [(0, None)]  # specify column widths
    test_columns = ["text"]  # specify column names

    # read_data() also creates "tokens" column
    test_df = read_data(test_file, test_colspecs, test_columns)

    test_df["sentiment"] = test_df["tokens"].apply(test_nb_classifier)
    logger.info("Classified sentiment of test data.\n")

    # Create text file of test data with classified sentiments
    test_file_classified = "output_files/testdata_classified.txt"
    test_columns_classified = ["sentiment", "text"]

    test_df.to_csv(
        test_file_classified,
        sep=" ",
        columns=test_columns_classified,
        header=False,
        index=False,
    )

    logger.info("Created text file with test data and classified sentiment!")
