import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# read data file in a DataFrame
df = pd.read_csv('./standardized_data.csv', sep='\t', header=0)

# split the data out into a test and train set
train_df, test_df = train_test_split(df, train_size=0.8,  random_state=np.random.seed())

# build a basic pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB())])

# fit the classifier to the training data
text_clf = text_clf.fit(train_df['text'], train_df['authenticity'])

# predict for the test data
predicted = text_clf.predict(test_df['text'])

# print some basic stats about our predictions
print(metrics.classification_report(test_df['authenticity'], predicted))