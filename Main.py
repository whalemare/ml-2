import pandas as panda
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn import metrics

# load data
df_pos = panda.read_csv('./data/twitter/positive.csv', sep=';', header=None)
df_neg = panda.read_csv('./data/twitter/negative.csv', sep=';', header=None)
np_pos = df_pos.as_matrix()
np_neg = df_neg.as_matrix()

# preprocessing data

np_all = np.concatenate((np_pos, np_neg), axis=0)

X = np_all[:, 3]
Y = np_all[:, 4]
Y = np.array(Y).astype(int)

vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=None,
    preprocessor=None,
    stop_words=None,
    max_features=9000
)

train_data_features = vectorizer.fit_transform(np_all[:, 3])
train_data_features = train_data_features.toarray()

# create classifier
X_train, X_test, Y_train, Y_test = train_test_split(train_data_features, Y, test_size=0.2, random_state=0)

clf = BernoulliNB()
# clf = GaussianNB()
# clf = MultinomialNB()
clf.fit(X_train, Y_train)
res = clf.predict(X_test)

# test classifier
print metrics.accuracy_score(Y_test, res)
