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
from contextlib import redirect_stdout
#%matplotlib inline

tfidf_shape = []
tfidf_shape.append(24783)
tfidf_shape.append(1000)
pos_shape = []
pos_shape.append(24783)
pos_shape.append(4100)
feats_shape = []
feats_shape.append(24783)
feats_shape.append(17)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
print("check point 1 ---------------------------->")

def get_pos_tags(tweets):
    tweet_tags = []
    print(type(tweets))
    i = 1
    for t in tweets:
        if type(t) == str:
            #print(i, "th tweet")
            tokens = basic_tokenize(preprocess(t))
            #print(tokens)
            tags = nltk.pos_tag(tokens)
            #pprint.pprint((tags))
            i += 1


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
    max_features=4000,
    min_df=5,
    max_df=0.75,
)


def count_twit(tweet):
    parsed_text = preprocess_diff(tweet)
    return(parsed_text.count('URLHERE'), parsed_text.count('MENTIONHERE'), parsed_text.count('HASHTAGHERE'))


print("check point 2 ---------------------------->")

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
print("check point 3 ---------------------------->")

other_features_names = ["FKRA", "FRE", "num_syllables", "avg_syl_per_word", "num_chars", "num_chars_total",
                        "num_terms", "num_words", "num_unique_words", "vader neg", "vader pos", "vader neu",
                        "vader compound", "num_hashtags", "num_mentions", "num_urls", "is_retweet"]

df_test = pd.read_csv("labeled_data.csv", error_bad_lines=False)
test_tweets = df_test.tweet
test_tweets = list(test_tweets)
add_df = pd.read_csv("file.csv",error_bad_lines=False)
add_tweet = add_df.tweet
add_tweet = list(add_tweet)
test_tweets.extend(add_tweet)
k=0
for i in test_tweets:
    k+=1
    print(k,"  ",type(i))
tfidf_test = vectorizer.fit_transform(test_tweets).toarray()
vocab_test = {v: i for i, v in enumerate(vectorizer.get_feature_names())}
idf_vals_test = vectorizer.idf_
idf_dict_test = {i: idf_vals_test[i] for i in vocab_test.values()}

print("check point 4 ---------------------------->")
test_tweet_tags = []
for t in test_tweets:
    test_tokens = basic_tokenize(preprocess(t))
    test_tags = nltk.pos_tag(test_tokens)
    test_tag_list = [x[1] for x in test_tags]
    test_tag_str = " ".join(test_tag_list)
    test_tweet_tags.append(test_tag_str)

test_pos = pos_vectorizer.fit_transform(pd.Series(test_tweet_tags)).toarray()
test_pos_vocab = {v: i for i, v in enumerate(
    pos_vectorizer.get_feature_names())}

test_feats = get_feature_array(test_tweets)
t = tfidf_shape[1] - tfidf_test.shape[1]
mat = [[0 for _ in range(t)] for _ in range(tfidf_test.shape[0])]
mat_f = np.array(mat)
tfidf_test = np.concatenate([tfidf_test, mat_f], axis=1)
t = pos_shape[1] - test_pos.shape[1]
mat = [[0 for _ in range(t)] for _ in range(test_pos.shape[0])]
mat_f = np.array(mat)
print("-------------->")
print(test_pos.shape)
print(mat_f.shape)
test_pos = np.concatenate([test_pos, mat_f], axis=1)
t = feats_shape[1] - test_feats.shape[1]
mat = [[0 for _ in range(t)] for _ in range(test_feats.shape[0])]
mat_f = np.array(mat)
test_feats = np.concatenate([test_feats, mat_f], axis=1)
print(tfidf_test.shape)
print(test_pos.shape)
print(test_feats.shape)
M_test = np.concatenate([tfidf_test, test_pos, test_feats], axis=1)
x_tests = pd.DataFrame(M_test)

print("check point 5 ---------------------------->")
x_tests = pd.DataFrame(M_test)

with open('model_pkl', 'rb') as f:
    loaded_model = pickle.load(f)
y_preds = loaded_model.predict(x_tests)
i = y_preds.size
with open('out.txt', 'w') as f:
    with redirect_stdout(f):
        for t in range(24783, i):
            if y_preds[t] == 1:
                print("offensive  ---->", test_tweets[t])
            if y_preds[t] == 0:
                print("hate  ---->", test_tweets[t])


