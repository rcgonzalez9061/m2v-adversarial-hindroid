# Perturbation-resilient metapath-based Android Malware Detection

## INTRODUCTION

- Problem: Android HIV breaks many models, shortcomings of HinDroid
- Proposition:
    - metapath2vec + better models
    - retraining on android HIV output

## METHODOLOGY
### Heterogeneous Information Network
### Feature Construction (Metapath2vec)
### Fortifying Against Adversarial Models (Maybe leave out and use next section only instead)

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
4. Repeat if necessary or possible

## RESULTS

- Initial performance of models on normal data
- Performance after Android HIV trained on data
- Performance of models 

## REFERENCES
