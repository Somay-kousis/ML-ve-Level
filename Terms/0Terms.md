### Feature vector

The array of [**feature**](https://developers.google.com/machine-learning/glossary#feature) values comprising an [**example**](https://developers.google.com/machine-learning/glossary#example). The feature vector is input during [**training**](https://developers.google.com/machine-learning/glossary#training) and during [**inference**](https://developers.google.com/machine-learning/glossary#inference). For example, the feature vector for a model with two discrete features might be:

### Unlabeled examples

In machine learning, the process of making predictions by applying a trained model to [**unlabeled examples**](https://developers.google.com/machine-learning/glossary#unlabeled_example).

###  Feature engineering

Converting raw data from the dataset into efficient versions of useful features.

### Bucketing

Converting a single [**feature**](https://developers.google.com/machine-learning/glossary#feature) into multiple binary features called **buckets** or **bins**, typically based on a value range. The chopped feature is typically a [**continuous feature**](https://developers.google.com/machine-learning/glossary#continuous_feature).


### Clipping

For example, suppose that <0.5% of values for a particular feature fall outside the range 40–60. In this case, you could do the following:

- Clip all values over 60 (the maximum threshold) to be exactly 60.
- Clip all values under 40 (the minimum threshold) to be exactly 40.


### Outliers

Values distant from most other values. In machine learning, any of the following are outliers:

- Input data whose values are more than roughly 3 standard deviations from the mean.
- [**Weights**](https://developers.google.com/machine-learning/glossary#weight) with high absolute values.
- Predicted values relatively far away from the actual values.


