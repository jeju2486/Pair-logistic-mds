# Research basis

KOVAR combines ideas from several established methods, but it is a standalone
implementation and should not be described as a reimplementation of any one of
them.

## Approximate generalized linear mixed models

**Breslow NE, Clayton DG (1993). Approximate Inference in Generalized Linear
Mixed Models.** Journal of the American Statistical Association 88:9-25.
[doi:10.1080/01621459.1993.10594284](https://doi.org/10.1080/01621459.1993.10594284)

This paper provides the foundational penalized quasi-likelihood framework for
approximate GLMM inference. KOVAR uses a one-variance-component PQL fit with a
binary logistic mean. KOVAR's numerical profiling and convergence rules are its
own implementation and require independent validation.

## Logistic mixed null model and prospective score testing

**Chen H et al. (2016). Control for Population Structure and Relatedness for
Binary Traits in Genetic Association Studies via Logistic Mixed Models.**
American Journal of Human Genetics 98:653-666.
[doi:10.1016/j.ajhg.2016.02.012](https://doi.org/10.1016/j.ajhg.2016.02.012)
([open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC4833218/))

GMMAT motivates KOVAR's main architecture: fit a logistic mixed null model and
reuse it for prospective score tests of many predictors. KOVAR differs in the
following ways:

- each binary locus can act as a response;
- candidate pairs are expanded into explicit statistical directions;
- covariance may come from a microbial phylogeny;
- the software and numerical solver are independent; and
- equivalence to GMMAT's AI-REML implementation is not claimed.

## Saddlepoint calibration

**Dey R et al. (2017). A Fast and Accurate Algorithm to Test for Binary
Phenotypes and Its Application to PheWAS.** American Journal of Human Genetics
101:37-49.
[doi:10.1016/j.ajhg.2017.05.014](https://doi.org/10.1016/j.ajhg.2017.05.014)
([open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC5501775/))

SPAtest demonstrates saddlepoint approximation for binary-trait score
statistics, especially under strong outcome imbalance.

**Zhou W et al. (2018). Efficiently controlling for case-control imbalance and
sample relatedness in large-scale genetic association studies.** Nature
Genetics 50:1335-1341.
[doi:10.1038/s41588-018-0184-y](https://doi.org/10.1038/s41588-018-0184-y)
([open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6119127/))

SAIGE combines a logistic mixed model, score testing, saddlepoint calibration,
variance-ratio approximation, and scalable numerical techniques. KOVAR borrows
the motivation for optional tail calibration, but it does not implement the
complete SAIGE algorithm and is not SAIGE-equivalent.

## Microbial population structure

**Lees JA et al. (2018). pyseer: a comprehensive tool for microbial
pangenome-wide association studies.** Bioinformatics 34:4310-4312.
[doi:10.1093/bioinformatics/bty539](https://doi.org/10.1093/bioinformatics/bty539)
([open full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289128/))

pyseer motivates the microbial-GWAS context: clonal population structure,
pangenome variation, strict sample matching, and tree- or genotype-derived
relatedness can materially affect association tests. KOVAR 0.8.0 replaces its
earlier pyseer-like LMM with an experimental binary logistic mixed model.

## Related phylogenetic and pangenome methods

These studies inform interpretation but are not algorithms implemented by
KOVAR 0.8.0:

- **Ives AR, Garland T Jr (2010). Phylogenetic Logistic Regression for Binary
  Dependent Variables.** Systematic Biology 59:9-26.
  [doi:10.1093/sysbio/syp074](https://doi.org/10.1093/sysbio/syp074)
- **Collins C, Didelot X (2018). A phylogenetic method to perform genome-wide
  association studies in microbes that accounts for population structure and
  recombination.** PLOS Computational Biology 14:e1005958.
  [doi:10.1371/journal.pcbi.1005958](https://doi.org/10.1371/journal.pcbi.1005958)
- **Whelan FJ, Rusilowicz M, McInerney JO (2020). Coinfinder: detecting
  significant associations and dissociations in pangenomes.** Microbial
  Genomics 6:000338.
  [doi:10.1099/mgen.0.000338](https://doi.org/10.1099/mgen.0.000338)

Phylogenetic logistic regression supports explicitly modelling dependence in a
binary response. treeWAS and Coinfinder illustrate complementary approaches to
phylogenetic association and pangenome co-occurrence. KOVAR does not reproduce
their ancestral-state, simulation, or lineage-independence procedures.

## KOVAR-specific design decisions

The following are project decisions rather than procedures copied from one
paper:

- testing both predictor/response directions;
- using a 5% minor-state-frequency filter and a separate four-cell filter;
- preferring an external core-genome tree over a same-matrix GRM;
- removing the LMM `h2` cap, GRM identity shrinkage, and rank truncation;
- applying BH and Bonferroni correction across finite directional tests; and
- calling results maximally inferred epistasis candidates while explicitly
  withholding causal or mechanistic interpretation.
