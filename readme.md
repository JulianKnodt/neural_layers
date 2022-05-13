# Structural Dropout

TL;DR: Dropout can be used as a way to ensure that certain features in a hidden state are more
important. By dropping out all later features in a hidden state, we ensure that earlier features
are more important, and we can prune later ones to achieve compression.

## Problem Statement

Neural Networks are extremely over-parametrized, wasting a large amount of memory and compute at
minimal gain. It is often not clear how to cut a network down, as cutting it down ad-hoc
afterwards will not preserve the original accuracy. Instead, if we could know which featues are
least "important" (i.e. contribute least to performance), we can remove them, at minimal cost to
accuracy. Now, the difficulty is actually determining which features are least important.

Instead of trying to examine a neural net after training, maybe it would be possible to force
the NN to make specific features more important. The way we do that is by dropping out certain
features much more frequently than others, forcing those features to be unimportant. We decide
which features to drop out based on their index. i.e. Probability of dropping out h[0] <
probability of dropping out h[10] < h[20], etc. We also ensure that models are nested inside
each other, by dropping out _all_ elements above a given index.

## Experiments

- MNIST

Basic experiment to verify that structural dropout works.

- PointNet

Experimenting with different values to run the whole model.

- LIIF

Experimenting on a deeper architecture to show that it works in a variety of cases.

## TODOs

It would be difficult for me to train and validate a large image classification model with my
current setup. Using that as a metric for compression would be good, and a great way to
demonstrate that it works well.
