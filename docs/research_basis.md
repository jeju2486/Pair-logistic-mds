# Research basis

KOVAR combines established ideas but is an independent experimental
implementation.

- Breslow NE, Clayton DG (1993), *Approximate inference in generalized linear
  mixed models*: basis for penalized quasi-likelihood.
- Chen H et al. (2016), *Control for population structure and relatedness for
  binary traits in genetic association studies via logistic mixed models*
  (GMMAT): basis for a logistic mixed-model null followed by score testing.
- Dey R et al. (2017), *A fast and accurate algorithm to test for binary
  phenotypes and its application to PheWAS* (SPAtest): basis for saddlepoint
  approximation of binary score tails.
- Zhou W et al. (2018), *Efficiently controlling for case-control imbalance and
  sample relatedness in large-scale genetic association studies* (SAIGE): basis
  for combining logistic mixed-model score tests, variance-ratio ideas, and SPA.
- Lees JA et al. (2018), *pyseer: a comprehensive tool for microbial pangenome-
  wide association studies*: basis for scalable bacterial association testing
  with population-structure adjustment and reusable null structure.

KOVAR differs from these programs by treating another binary locus as the
outcome for each candidate pair, deriving covariance from a required rooted
tree, reusing exact/complement response patterns, and reporting one unordered
pair. Its SPA is not a full SAIGE implementation, and its PQL calibration must
be established specifically for bacterial pangenome covariation.
