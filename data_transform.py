import os
import pandas as pd
import csv

DATA_SOURCES = [('NEGATIVE', 'DECEPTIVE', './data/negative_polarity/deceptive_from_MTurk'),
                ('NEGATIVE', 'AUTHENTIC', './data/negative_polarity/truthful_from_Web'),
                ('POSITIVE', 'DECEPTIVE', './data/positive_polarity/deceptive_from_MTurk'),
                ('POSITIVE', 'AUTHENTIC', './data/positive_polarity/truthful_from_TripAdvisor')]

OUTPUT_FILE = 'standardized_data.csv'

df = pd.DataFrame(columns=['sentiment', 'authenticity', 'file', 'text'])

for source in DATA_SOURCES:

    sentiment = source[0]
    authenticity = source[1]
    path_to_data_source = source[2]

    for folder_name in os.listdir(path_to_data_source):

        path_to_folder = path_to_data_source + '/' + folder_name

        for file_name in os.listdir(path_to_folder):

            path_to_file = path_to_folder + '/' + file_name

            with open(path_to_file, 'r') as file:
                text = file.read()
                df = df.append({
                    'sentiment': sentiment,
                    'authenticity': authenticity,
                    'file': path_to_file,
                    'text': text.strip()
                    }, ignore_index=True)

df.to_csv(OUTPUT_FILE, sep='\t', header=True, index=False, quoting=csv.QUOTE_NONE)
