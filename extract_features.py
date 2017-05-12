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

    print("Starting feature extraction...")
    for text in tqdm(texts):
        features = {}

        tb = TextBlob(text)
        words = tb.words
        sentences = tb.sentences
        char_length = len(text)
        word_length = len(words)
        sentences_length = len(sentences)

        features['subjectivity'] = tb.sentiment.subjectivity
        features['count_personal_pronouns'] = sum(tag[1] == 'PRP' for tag in tb.tags)
        features['count_proper_noun'] = sum(tag[1] in ['NNP', 'NNPS'] for tag in tb.tags)
        features['avg_word_length'] = char_length/word_length
        features['length'] = word_length
        features['count_misspellings'] = _count_possible_misspellings(d, words)
        features['count_of_sentences'] = sentences_length
        features['avg_sentence_length'] = char_length/sentences_length

        features_list.append(features)

        if report:
            print("\nText:")
            print(text)
            print("Features:")
            pprint(features)


    return features_list;