import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
import re, string
import random

# Import train data from text to dataframe
file = "~/Documents/Coding/Text Classification/training.txt"
colspecs = [(0,1), (2,None)] # specify column widths
columns = ["sentiment", "text"] # specify column names

df = pd.read_fwf(file, colspecs=colspecs, header=None, names=columns)

# Tokenize the text
'''
Tokenizing the text data converts it from a single string to a list of words
and punctuation.
'''

df["tokens"] = df["text"].apply(word_tokenize)

# Create function to normalize text
'''
Normalizing the text will involve:
- Removing special characters from words
- Lowercasing words
- Lemmatizing words by finding their root
- POS-tagging words to understand their is the grammatical
  position, and using this information to improve lemmatizing
  accuracy
'''

def normalize_text(word,stop_words=()):
    lemmatizer = WordNetLemmatizer()
    cleaned_sentence = []
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

        if len(word) > 0 and word not in string.punctuation and word not in stop_words:
            cleaned_sentence.append(word)

    return cleaned_sentence

# Normalize the text
stop_words = stopwords.words('english')
df["tokens_normalized"] = df["tokens"].apply(normalize_text)

# Combine tokenized text into one list
words_list = []
for l in df["tokens_normalized"]:
    words_list.extend(l)

# Find frequency of words, and select most frequent words
all_words = nltk.FreqDist(w.lower() for w in words_list)
feature_words = list(all_words)

# Define the feature extractor
def text_features(text):
    text_lower = [w.lower() for w in text]
    # "set" creates a set of unique words in each text
    text_words = set(text_lower)
    features = {}
    for word in feature_words:
        features[word] = (word in text_words)
    return features

# Create a list of tuples [(a,b)] to more efficiently read data
def merge_df(df1, df2):
    merged_df = tuple(zip(list(df1), list(df2)))
    return merged_df

df_list = merge_df(df["tokens"], df["sentiment"])

# Train Naive Bayes classifier
featuresets = [(text_features(t),s) for (t,s) in df_list]
random.shuffle(featuresets)
train_set, test_set = featuresets[3000:], featuresets[:3000]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Report classifier accuracy
print(nltk.classify.accuracy(classifier, test_set))

# Show the most important features as interpreted by Naive Bayes
classifier.show_most_informative_features(5)

# Import test data from text to dataframe
file_test = "~/Documents/Coding/Text Classification/testdata.txt"
colspecs_test = [(0, None)] # specify column widths
columns_test = ["text"] # specify column names

df_test = pd.read_fwf(file_test, colspecs=colspecs_test, names=columns_test)

# Test classifier on test data
def test_classifier(text):
    tokens = normalize_text(word_tokenize(text))
    sentiment = classifier.classify(dict([t, True] for t in tokens))
    return sentiment

df_test["sentiment"] = df_test["text"].apply(test_classifier)

# View results of sentiment analysis
for i in range(100):
    print(df_test["text"][i],df_test["sentiment"][i])
