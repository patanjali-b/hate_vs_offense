from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import sys
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem.porter import *
import string
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat.textstat import *
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import seaborn
import pprint
#%matplotlib inline

df = pd.read_csv("labeled_data.csv")
df['class'].hist()
tweets = df.tweet

stopwords = stopwords = nltk.corpus.stopwords.words("english")
extra = ["#ff", "ff", "rt", "&amp"]
stopwords.extend(extra)
sentiment_analyzer = VS()
stemmer = PorterStemmer()


def preprocess(text_string):
    #print("entered preprocess")
    #print(type(text_string), text_string)
    space_pattern = '\s+'
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub(hashtag_regex, '', parsed_text)
    #print(len(parsed_text))
    #print(parsed_text)
    return parsed_text


def preprocess_diff(text_string):
    #print("entered preprocess")
    #print(type(text_string), text_string)
    space_pattern = '\s+'
    url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
                 '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(url_regex, 'URLHERE', parsed_text)
    parsed_text = re.sub(mention_regex, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(hashtag_regex, 'HASHTAGHERE', parsed_text)
    #print(len(parsed_text))
    #print(parsed_text)
    return parsed_text


def tokenize(tweet):
    #print("tokenizing")
    tweet = (tweet.lower()).strip()
    lisst = tweet.split()
    #print(lisst)
    new_tweet = []
    pattern = r'[0-9]'
    for i in lisst:
        i = re.sub(r'[^\w\s]', '', i)
        i = re.sub(pattern, '', i)
        i = stemmer.stem(i)
        if i != "":
            new_tweet.append(i)
    return new_tweet


vectorizer = TfidfVectorizer(
    tokenizer=tokenize,
    preprocessor=preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=1000,
    min_df=5,
    max_df=0.75
)


def basic_tokenize(tweet):
    #print("tokenizing")
    tweet = (tweet.lower()).strip()
    lisst = tweet.split()
    #print(lisst)
    new_tweet = []
    pattern = r'[0-9]'
    for i in lisst:
        i = re.sub(r'[^\w\s]', '', i)
        i = re.sub(pattern, '', i)
        if i != "":
            new_tweet.append(i)
    return new_tweet

import warnings
warnings.simplefilter(action='ignore', category = FutureWarning)

tfidf = vectorizer.fit_transform(tweets).toarray()
vocab = {v:i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals = vectorizer.idf_
idf_dict = {i:idf_vals[i] for i in vocab.values()}


def get_pos_tags(tweets):
    tweet_tags = []
    #print(type(tweets))
    i = 1
    for t in tweets:
        if type(t) == str:
            #print(i, "th tweet")
            tokens = basic_tokenize(preprocess(t))
            #print(tokens)
            tags = nltk.pos_tag(tokens)
            #pprint.pprint((tags))
            i += 1


tweet_tags = []
for t in tweets:
    tokens = basic_tokenize(preprocess(t))
    tags = nltk.pos_tag(tokens)
    tag_list = [x[1] for x in tags]
    tag_str = " ".join(tag_list)
    tweet_tags.append(tag_str)

pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.75,
)
pos = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}


def count_twit(tweet):
    parsed_text = preprocess_diff(tweet)
    return(parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


def other_features(tweet):
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    words = preprocess(tweet)
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syllables = round(float((syllables+0.001))/float(num_words+0.001), 4)
    num_unique_terms = len(set(words.split()))
    #finding modified Flescher Kincaid Score
    FKRA = round(float(0.39 * float(num_words)/1.0) +
                 float(11.8 * avg_syllables) - 15.59, 1)
    ##Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) -
                (84.6*float(avg_syllables)), 2)
    twitter_objs = count_twit(tweet)
    retweet = 0
    features = [FKRA, FRE, syllables, avg_syllables, num_chars, num_chars_total, num_terms, num_words,
                num_unique_terms, sentiment['neg'], sentiment['pos'], sentiment['neu'], sentiment['compound'],
                twitter_objs[2], twitter_objs[1],
                twitter_objs[0], retweet]

    return features


def get_feature_array(tweets):
    feats = list()
    for tweet in tweets:
        feats.append(other_features(tweet))
    feats = np.array(feats)
    return feats


other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",
                        "num_terms", "num_words", "num_unique_words", "vader neg", "vader pos", "vader neu",
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

feats = get_feature_array(tweets)
M = np.concatenate([tfidf,pos,feats],axis=1)
M.shape
X = pd.DataFrame(M)
y = df['class'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3
)
pipe = Pipeline(
    [('select', SelectFromModel(LogisticRegression(class_weight='balanced',
                                                   penalty="l1", C=0.01, solver='liblinear'))),
        ('model', LogisticRegression(class_weight='balanced', penalty='l1', solver='liblinear'))])

param_grid = [{}]
grid_search = GridSearchCV(pipe,
                           param_grid,
                           cv=StratifiedKFold(n_splits=5).split(
                               X_train, y_train),
                           verbose=2)

model = grid_search.fit(X_train, y_train)

y_preds = model.predict(X_test)

report = classification_report(y_test, y_preds)
print(report)
