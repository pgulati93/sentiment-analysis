'''
We're given a training dataset with fixed widths. The first column contains
binary values (1 for positive sentiment, 0 for negative), the second column
contains text (opinions on a subject).

i.e.
1	The Da Vinci Code book is just awesome.
0	Da Vinci Code sucks.

Using this training dataset, we're going to build a machine learning algorithm
that will learn how to classify text as positive or negative sentiment.
'''

'''
After parsing through the data, we can categorize it as:
Line          | Sentiment | Subject            | Note
————————————————————————————————————————————————————————————————————————————————
1 - 999       | Positive  | Davinci            | Changed some to Negative
1000 - 1997   | Positive  | Mission Impossible |
1998 - 2996   | Positive  | Harry Potter       |
2997 - 3995   | Positive  | Brokeback          |
3996 - 4993   | Negative  | Davinci            |
4994 - 5086   | Negative  | Mission Impossible | Changed some to Positive
5087 - 6086   | Negative  | Harry Potter       | Changed some to Positive(*)
6087 - 7086   | Negative  | Brokeback          | Changed some to Positive

(*)many of these were sexual
'''

import pandas as pd
import nltk
import matplotlib.pyplot as plt

# 1. Import data from text to dataframe
file = "~/Documents/Coding/Text Classification/training.txt"
colspecs = [(0,1), (2,None)] # specify column widths
columns = ["sentiment", "text"] # specify column names

df = pd.read_fwf(file, colspecs=colspecs, header=None, names=columns)

# 2. Tokenize the text
'''
Tokenizing the text data converts it from a single string to a list of words
and punctuation.
'''

df["tokenized_text"] = df["text"].apply(nltk.word_tokenize)

# 3. Add part-of-speech (POS) tag to text
'''
POS tagging will help us grammatically label the text for further analysis.

We'll add a column to our dataframe ("tokenized_text_pos") to include this
POS-tagged text.
'''

df["tokenized_text_pos"] = df["tokenized_text"].apply(nltk.pos_tag)

## 3.1 Separate POS-tag into its own list
'''
Currently POS-tag is in a tuple with the word it is tagging. Let's separate
the POS-tag into its own list for easier analysis.
'''

def pos_separate(list_tuple):
    pos_list = []
    for t in list_tuple:
        (text,pos) = t
        pos_list.append(pos)

    return pos_list

df["pos_list"] = df["tokenized_text_pos"].apply(pos_separate)

## 3.2 Identify list of unique POS-tags
'''
Understanding the unique POS tags (in terms of which ones are present,
how many there are, and what they mean) will help us narrow down which
grammatical terms are best to focus our algorithm on.
'''

pos_unique = []

def find_unique_pos(pos_list):
    for p in pos_list:
        if p not in pos_unique:
            pos_unique.append(p)
    return

df["pos_list"].apply(find_unique_pos)

pos_count = [0]*len(pos_unique)

for i in range(len(pos_unique)):
    for l in df["pos_list"]:
        for p in l:
            if pos_unique[i] == p:
                pos_count[i] += 1

pos_data = {"pos_unique": pos_unique,
            "pos_count" : pos_count}

df_pos = pd.DataFrame(pos_data,columns=['pos_unique','pos_count'])
df_pos = df_pos.sort_values(by=["pos_count"],ascending=False)

df_pos.plot(x='pos_unique', y='pos_count', kind='bar')
plt.show()

for pos in df_pos["pos_unique"]:
    nltk.help.upenn_tagset(pos)

'''
We find that there are 43 unique POS-tags:
['DT', 'NNP', 'NN', 'VBZ', 'RB', 'JJ', '.', 'VBD', 'VBP', 'VBN', ',', 'CC',
'NNS', 'IN', 'RBR', 'PRP', 'VB', 'TO', ')', 'WDT', 'VBG', ':', 'CD', 'RBS',
'PRP$', 'MD', '#', 'JJR', 'POS', '(', '``', 'SYM', 'WRB', 'UH', 'RP', 'PDT',
'JJS', 'NNPS', "''", 'WP', 'EX', 'FW', '$']

The 14 most common POS-tags (excluding punctuation) are:
POS  |  Count  |  Defition
————————————————————————————————————————————————————————————————
NNP     17510     noun, proper, singular
NN      11371     noun, common, singular or mass
PRP     6009      pronoun, personal
DT      5412      determiner
JJ      5142      adjective or numeral, ordinal
IN      4700      preposition or conjunction, subordinating
VBP     4039      verb, present tense, not 3rd person singular
RB      3866      adverb
VBD     3273      verb, past tense
VBZ     2983      verb, present tense, 3rd person singular
CC      2644      conjunction, coordinating
NNS     2615      noun, common, plural
VB      1768      verb, base form
VBG     1671      verb, present participle or gerund
'''

## 3.3 Identify the sentiments which POS-tagging could not tag a subject
'''
Each sentiment should have a subject (proper noun or 'NNP'). We'll need to
triage the sentiments that do not have an 'NNP' tag to improve our algorithm.
'''

def no_nnp(pos_list):
    if 'NNP' in pos_list:
        nnp_flag = 1
    else:
        nnp_flag = 0

    return nnp_flag

df["nnp_flag"] = df["pos_list"].apply(no_nnp)

print("count_1:",len(df[(df["sentiment"] == 1) & (df["nnp_flag"] == 0)]),
      "count_0:",len(df[(df["sentiment"] == 0) & (df["nnp_flag"] == 0)]))
print("total_1:",len(df[(df["sentiment"] == 1)]),
      "total_0:",len(df[(df["sentiment"] == 0)]))
print(len(df))

'''
The count of positive and negative sentiments without a subject ('NNP') is:

count_1: 866 count_0: 268
total_1: 4249 total_0: 2837

This makes up 20% of positive and 9.4% of negative sentiment, a total of 16%.

This value is significant enough to warrant a triage to fix these sentiments.
'''
