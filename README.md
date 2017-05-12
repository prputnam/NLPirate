# NLPirate

This project contains two finalized classifiers, designed to determine the authenticity of a review.

`initial_classifier.py` is a classifier built based on the [work of Myle Ott and team on review authenticity](http://myleott.com/op_spam/).
`final_classifier.py` is our attempt to develop a classifier that more accurately classifies review authenticity.

## Dependencies

Requreid for `final_classifier.py`:

* [pyenchant](https://github.com/rfk/pyenchant/)

## Data

The data for our project is courtesy of Myle Ott, and can be found on his [website](http://myleott.com/op_spam/).

Included is a `data_transform.py` script to transform the data out of its folder structure and into `standardized_data.csv`, a tab-delimited file. This does not need to be run, as the `standardized_data.csv` file is included in the repository.

## Execution

To run either of the classifiers, simply invoke them from the command line: `python final_classifier.py`. The classifiers will use the entirity of the dataset, making use of k-fold validation, and then report on their results at the end of classification, as well as the most useful features for the classification.

Also included are two Python notebook files that make use of the visualization library [ELI5](https://github.com/TeamHG-Memex/eli5). Running these from Jupyter will allow for investigation into the classification of particular reviews, with highlisted features in the text, and feature weighting for the particular review.
