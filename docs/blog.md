---
layout: page
title: m2vDroid
permalink: /
---

{% include head.html %}

# m2vDroid: Perturbation-resilient metapath-based Android Malware Detection

## INTRODUCTION

Over the past decade, malware has established itself as a constant issue for the Android operating system. In 2018, Symantec reported that they blocked more than 10 thousand malicious Android apps per day, while nearly 3 quarters of Android devices remained on older versions of Android. With billions active Android devices, millions are only a swipe away from becoming victims. Naturally, automated machine learning-based detection systems have become commonplace solutions as they can drastically speed up the labeling process. However, it has been shown that many of these models are vulnerable to adversarial attacks, notably attacks that add redundant code to malware to consfuse detectors. 

First, we introduce a new model that extends the [Hindroid detection system](https://www.cse.ust.hk/~yqsong/papers/2017-KDD-HINDROID.pdf) by employing node embeddings using [metapath2vec](https://ericdongyx.github.io/papers/KDD17-dong-chawla-swami-metapath2vec.pdf). We believe that the introduction of node embeddings will improve the performance of the model beyond the capabilities of HinDroid. Second, we intend to break these two models using a method similar to that proposed in  the [Android HIV paper](https://ieeexplore.ieee.org/document/8782574). That is we train an adversarial model that perturbs malware such that a detector mislabels it as a benign app. We then measure the performance of each model after recursively feeding adversarial examples back into them. We believe that by doing so, our model will be able outperform the Hindroid implementation in its ability to label malware even after adversarial examples have been added.

## METHODOLOGY
First, we will describe the details of our proposed model, m2vDroid, and then we will describe the method we used to attack each model in the Adversarial Attack section.

## m2vDroid
m2vDroid is another malware detection model that we implemented that is largely based off HinDroid. However it uses node embeddings for the final vector representations of apps instead of the bag-of-APIs/commuting matrix solution that HinDroid applied.

### Preliminaries
There are a few concepts that we should introduce before we get into details:

- *Definition 1)* A **Heterogeneous Information Network (HIN)** is a graph in which its nodes and edges have diferent types. 

- *Definition 2)* A **Metapath** is a path within a HIN that follows certain node types. For example, let us define a HIN with a set of node types $T$ and a path $P = n_1 \longrightarrow n_2 \longrightarrow ... \longrightarrow n_{N}$ of length $N$. $P$ follows metapath $M_P = t_1 \longrightarrow t_2 \longrightarrow ... \longrightarrow t_{N}$ if $type(n_i) = t_i$ for all $i \in [1,2,...,N]$.

### Feature extraction
Our ETL pipeline begins with Android apps in the form of APK files. These APKs are unpacked using [Apktool](https://ibotpeaches.github.io/Apktool/) to reveal the contents of the app, but we are primarily concerned with `classes.dex`, the app's bytecode. We decompile the bytecode using [Smali](https://github.com/JesusFreke/smali) into readable `.smali` text files. From here we extract each API call, along with the app and method it appears in, and the package it is from. This is done for every API in an app and for every app in a dataset, forming a table with the the information needed for the next step.

### HIN Construction
Using the data extracted previously, we construct a heterogeneous information network using the [Stellargraph](https://github.com/stellargraph/stellargraph) library. Our HIN contains 4 types of nodes which we define as: 
- $Apps$: Android apps determined by name or md5, i.e. `com.microsoft.excel` or `09d347c6f4d7ec11b32083c0268cc570`.
- $APIs$: APIs determined by their smali representation, i.e. `Lpackage/Class;->method();V`
- $Packages$: the package an API originates from, i.e. `Lpackage`.
- $Methods$: Methods (or "functions") that API calls appear in, i.e. `LclassPackage/class/method();V`.

The distinct nodes for each type correspond to the distinct values of their column in the API data table described earlier. $Apps$ and $APIs$ share an edge if an $API$ is used within an $App$. Likewise with $APIs$ and $Methods$, they share an edge if a $Method$ contains an $API$. $Packages$ and $APIs$ share an edge if an $API$ orginates from a $Package$.

### Metapath2vec
To generate our features, we apply the metapath2vec algorithm on the $App$ nodes of our HIN. That is we 1) perform a random-walk starting from each app following designated metapaths to generate a corpus consisting of nodes in our HIN, then we 2) pass this corpus into the gensim implmentation of the [word2vec](https://arxiv.org/pdf/1301.3781.pdf) model to transform each $App$ into a vector. For the metapath walk, we leveraged Stellargraph's `MetaPathRandomWalk` algorithm and specified a walk length of 60, walking on each of the following metapaths 3 times per $App$ node:
- $App$ $\rightarrow$ $Api$ $\rightarrow$ $App$
- $App$ $\rightarrow$ $Api$ $\rightarrow$ $Method$ $\rightarrow$ $Api$ $\rightarrow$ $App$
- $App$ $\rightarrow$ $Api$ $\rightarrow$ $Package$ $\rightarrow$ $Api$ $\rightarrow$ $App$
- $App$ $\rightarrow$ $Api$ $\rightarrow$ $Package$ $\rightarrow$ $Api$ $\rightarrow$ $Method$ $\rightarrow$ $Api$ $\rightarrow$ $App$
- $App$ $\rightarrow$ $Api$ $\rightarrow$ $Method$ $\rightarrow$ $Api$ $\rightarrow$ $Package$ $\rightarrow$ $Api$ $\rightarrow$ $App$

We chose these metapaths as they formed some of the top performing kernels in the HinDroid paper that we were able to compare.

For word2vec, we used a skip-gram model trained over 5 epochs. We used a window size of 7 so that 2 connected apps could appear in a window even in the longest metapaths and `min_count=0` so that all nodes in the metapath walk were incorporated. We were sure to include negative sampling as part of the process, as negative samples would help further distinguish nodes the are not associated with each other. For this we specified `negative=5` for a final output of vector of length 128.

After running this ETL on our data, we observed clear clustering after plotting a TSNE transformation. For the most, part it seems that this method is able to distinguish between not only malware and non-malware, but can also distinguish between different classes of malware to a reasonable extent. Notably, we have not tested the node2vec or metapath2vec++ algorithms for generating our random walk.

{% include 3D-plot.html %}

## ADVERSARIAL ATTACK
Our adversarial attack follows many of the techniques applied by Chen, et al. 2018 to attack the MaMaDroid and Drebin models. To perform our adversarial attack, we simulated *Scenerio FB* with the following conditions: we have blackbox access to a malware classifier so we may query it to predict examples as they are generated and we also have access to the feature set with which to input into the classifier. The classifier may vary between m2vDroid and different kernels of HinDroid for initial benchmarks, but we will primarily use the m2vDroid classifier once we begin recursive training. The feature set will be the set of APIS derived from our training app set. The input vector will be the one-hot-encoded set of APIs for the example app. Our goal is to have the adversarial model add APIs to the example until it is misclassified or we reach a maximum. We want the app to retain its original function, so no APIs can be removed. Therefore, we modified the C&W attack from Android HIV to fit our application.

## EXPERIMENT
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

## APENDIX
[Source code](https://github.com/rcgonzalez9061/m2v-adversarial-hindroid)
