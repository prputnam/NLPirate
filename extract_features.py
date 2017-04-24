from textblob import textblob
from collections import defaultdict

def extract_features(trainingData):
    feature_list = []

    for data_entry in trainingData:
        entry_feature_dict = defaultdict(int)
        entryTextBlob = TextBlob(data_entry)
        entry_feature_dict["polarity"] = entryTextBlob.sentiment.polarity
        entry_feature_dict["subjectivity"] = entryTextBlob.sentiment.subjectivity

        for t in entryTextBlob.tags:
            #counting personal pronouns
            if t[1] == 'PRP':
                entry_feature_dict["personal_pronouns"] += 1

         feature_list.append(entry_feature_dict)
         return feature_list