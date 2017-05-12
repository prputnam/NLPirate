import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import LinearSVC
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.feature_extraction import DictVectorizer

from extract_features import extract_all_features, extract_most_performant_features

class FeatureExtractor(object):
    
    def transform(self, X):
        # can be swapped to extract_all_features for full feature list
        return extract_most_performant_features(X)

    def fit(self, X, y=None):
        return self


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

# here be dragons! - things get a bit hacky through here
# overriding the Pipeline to have a get_feature_names method that
# points back to the vectorizer used
feat_extractor = FeatureExtractor()
dict_vectorizer = DictVectorizer()
feat_pipe = Pipeline([
    ('feat_extractor', feat_extractor),
    ('dict_vectorizer', dict_vectorizer)
]);

def get_dict_vectorizer_names(self):
    return dict_vectorizer.get_feature_names()

feat_pipe.get_feature_names = get_dict_vectorizer_names.__get__(feat_pipe)    


# construct our main pipeline
clf = Pipeline([
    ('union', FeatureUnion([
        ('dict_vect', feat_pipe),                  
        ('count_vect', CountVectorizer(ngram_range=(1, 3)))
    ])),
    ('tfidf', TfidfTransformer()),
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
_show_most_informative_features(clf.named_steps['union'], clf.named_steps['svm'])
