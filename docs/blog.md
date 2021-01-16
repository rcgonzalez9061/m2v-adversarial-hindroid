---
layout: page
title: m2vDroid
permalink: /
---

{% include head.html %}

# m2vDroid: Perturbation-resilient metapath-based Android Malware Detection

## INTRODUCTION

Over the past decade, malware has established itself as a constant issue for the Android operating system. In 2018, Symantec reported that they blocked more than 10 thousand malware apps per day, while nearly 3 quarters of Android devices remained on older versions of Android. With billions active Android devices, millions are only a swipe away from becoming victims. Naturally, automated machine learning-based detection systems have become commonplace solutions to address this threat. However, it has been shown that many of these models are vulnerable to adversarial attacks, notably attacks that add redundant code to malware to consfuse detectors. 

First, we intend to break an existing malware detection system, [Hindroid](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) using a method similar to that proposed in [Android HIV](https://ieeexplore.ieee.org/document/8782574). Second, we introduce a new model that extends HinDroid that we hope to be more resilient to perturbations in malware code. 

OR

First, we introduce a new model that extends the [Hindroid detection system](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) by employing node embeddings using [metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf). We believe that the introduction of node embeddings will improve the performance of the model beyond the capabilities of HinDroid. Second, we intend to break these models using a method similar to that proposed in [Android HIV](https://ieeexplore.ieee.org/document/8782574) and measure the performance of each model after recursively feeding adversarial examples back into them.

## m2vDroid
### Preliminaries
- Include definitions etc.

### HIN Construction
Our heterogeneous information network contains 4 types of nodes which we define: 
- $Apps$: Android apps determined by name.
- $APIs$: APIs determined by their smali representation, i.e. `Lpackage/Class;->method();V`
- $Packages$: the package an API originates from.
- $Methods$: Methods (or "functions") that API calls appear in.

Apps and APIs share an edge if the API is used within the App. Likewise with APIs and Methods, they share an edge if a Method contains an API. Packages and APIs share an edge if the API orginates from the Package. With this representation, we believe we should retain more information about the apps we aim to represent versus HinDroid.

### Feature extraction (Metapath2vec)
To generate our features, we apply the metapath2vec algorithm on the $App$ nodes of our HIN. That is we 1) perform a random-walk starting from each app following a designated metapath pattern(s) to generate a corpus, then we 2) pass this corpus into the [word2vec](https://arxiv.org/pdf/1301.3781.pdf) model to transform each $App$ into a vector.

{% include 3D-plot.html %}

## EXPERIMENT SETUP
Using multiple models:
- Our HinDroid implementation
- Our improved model (and possible variations)
    - random forest and a gradient-boosted model 

We...

1. Train on normal data
2. Train Android HIV on these models and output perturbed sourced code, perturbing only the malware
3. Retrain models on original code pool + perturbed code
4. Repeat if necessary (or possible)

## RESULTS

- Initial performance of models on normal data
- Performance after Android HIV trained on data
- Performance of models 

## REFERENCES
