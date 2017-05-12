import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_predict

from pprint import pprint

def _show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-25s\t\t%.4f\t%-25s" % (coef_1, fn_1, coef_2, fn_2))


print("Reading in data from `standardized_data.csv`")

# read data file in a DataFrame
df = pd.read_csv('./standardized_data.csv', sep='\t', header=0)

kfold = KFold(n_splits=5, shuffle=True, random_state=np.random.seed(7))
texts = df['text'].tolist()
targets = df['authenticity'].tolist()

# construct our main pipeline
clf = Pipeline([                
    ('count_vect', CountVectorizer(ngram_range=(1, 3))),
    ('tfidf', TfidfTransformer(norm='l2', use_idf = True)),
    ('svm', LinearSVC())
])

print("Running k-fold cross validation...")
predicted = cross_val_predict(clf, texts, targets, cv=kfold)

print()
print(metrics.classification_report(targets, predicted))    
print(metrics.confusion_matrix(targets, predicted))
print(metrics.accuracy_score(targets, predicted))
print()

# Fit agianst the entire dataset to generate most informative features
clf = clf.fit(texts, targets)
_show_most_informative_features(clf.named_steps['count_vect'], clf.named_steps['svm'])
