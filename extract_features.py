import os, sys
import enchant

from tqdm import tqdm
from pprint import pprint

parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, 'vendor')

sys.path.append(vendor_dir)

from textblob import TextBlob

def _count_possible_misspellings(d, words):
    return sum(not d.check(word.string) for word in words)


def extract_features(texts, report=False):
    d = enchant.Dict("en_US")

    features_list = []

    for text in tqdm(texts):
        features = {}

        tb = TextBlob(text)
        words = tb.words
        char_length = len(text)
        word_length = len(words)

        features['polarity'] = tb.sentiment.polarity
        features['subjectivity'] = tb.sentiment.subjectivity
        features['percent_personal_pronouns'] = sum(tag[1] == 'PRP' for tag in tb.tags)/char_length
        features['percent_proper_noun'] = sum(tag[1] in ['NNP', 'NNPS'] for tag in tb.tags)/char_length
        features['percent_proper_noun'] = sum(tag[1] in ['NNP', 'NNPS'] for tag in tb.tags)/char_length
        features['avg_word_length'] = char_length/word_length
        features['length'] = word_length
        features['count_possible_misspellings'] = _count_possible_misspellings(d, words)
        features['percent_possible_misspellings'] = _count_possible_misspellings(d, words)/word_length

        if report:
            print("\nText:")
            print(text)
            print("Features:")
            pprint(features)

        features_list.append(features)


    return features_list;
