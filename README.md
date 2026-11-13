# sblreg: symmetric bilinear regression in R

`sblreg` regresses a scalar outcome on a network. Each observation *i* has a symmetric *V* × *V* matrix *W*<sub>i</sub> (a
connectome, a co-expression or partial-correlation network, an adjacency matrix) and an outcome *y*<sub>i</sub>. The package fits the
L1-penalised symmetric bilinear model of Wang, Zhang and Dunson (2019):

$$
E(y_i \mid W_i) = \alpha + \sum_{h=1}^{K} \lambda_h \, \beta_h^\top W_i \beta_h ,
$$

$$
\min_{\alpha,\lambda,\beta} \; \frac{1}{2n} \sum_{i=1}^{n} \Big( y_i - \alpha - \sum_{h} \lambda_h \beta_h^\top W_i \beta_h \Big)^2
\;+\; \gamma \sum_{h} |\lambda_h| \sum_{u>v} |\beta_{hu}\beta_{hv}| .
$$

Each nonzero component λ<sub>h</sub>β<sub>h</sub>β<sub>h</sub><sup>⊤</sup> is supported on a small set of nodes, a *clique subgraph*,
whose connections are related to the outcome. The coefficient matrix *B* = Σ<sub>h</sub> λ<sub>h</sub>β<sub>h</sub>β<sub>h</sub><sup>⊤</sup>
then gives the edge-level effects. Compared with a lasso on the *V*(*V*−1)/2 edges, the bilinear form groups the selected edges
into a few node sets and needs only 1 + *K* + *KV* parameters.

The fit is the coordinate-descent algorithm of the paper (Algorithm 1, updates (16), (19), (20)), written in C++ (RcppArmadillo).
On top of it the package provides penalty paths, cross-validated tuning, and a nested cross-validation assessment with a
permuted-outcome control.

> Wang L, Zhang Z, Dunson DB (2019). Symmetric bilinear regression for signal subgraph estimation.
> *IEEE Transactions on Signal Processing* 67(7):1929–1940. [doi:10.1109/TSP.2019.2899818](https://doi.org/10.1109/TSP.2019.2899818)
> (open access: [PMC10198064](https://pmc.ncbi.nlm.nih.gov/articles/PMC10198064/))

## Installation

```r
# install.packages("remotes")
remotes::install_github("sagnikbhadury/sblreg")
```

The package contains C++ code, so it needs a compiler: Rtools on Windows, Xcode command-line tools on macOS, `r-base-dev` on Linux.
It depends on Rcpp and RcppArmadillo.

## Quick start

```r
library(sblreg)

# Networks and outcome from the simulation design of the paper (Sec. V-B): 20 nodes, the outcome
# depends on three cliques (1 edge, a triangle and a 4-node clique; 10 signal edges)
sim <- sbl_simulate(n = 100, V = 20, snr = "high", seed = 1)
str(sim$W[[1]])          # a 20 x 20 symmetric matrix
length(sim$y)            # 100

# One fit at a fixed penalty
fit <- sbl_fit(sim$W, sim$y, gamma = 0.5, K = 5, nstart = 5)
fit                      # active components, nonzero edges, nodes involved
head(sbl_edges(fit))     # node1, node2, edge coefficient 2*B[u, v], component(s)
sbl_nodes(fit)           # nodes and the number of selected edges they touch
predict(fit, sim$W[1:5])
```

## Choosing the penalty

The penalty γ is the only tuning parameter that matters. *K* is an upper bound: the L1 penalty removes the components that are
not needed, and the paper found results insensitive to *K* between 5 and 7 (Table IV).

```r
# Largest useful penalty and a log-spaced grid (paper: 50 values from 0.01 * gamma_max)
gmax <- sbl_gamma_max(sim$W, sim$y)
path <- sbl_path(sim$W, sim$y, ngamma = 50, ratio = 0.01)

# 5-fold cross-validation over the grid; gamma with the smallest mean CV error; refit on all data
cvf <- cv_sbl(sim$W, sim$y, nfolds = 5, ngamma = 50)
cvf
plot(log(cvf$gammas), cvf$cvm, type = "b", xlab = "log gamma", ylab = "CV mean squared error")
sbl_edges(cvf$fit)
```

## Honest assessment: nested CV with a permuted-outcome control

Choosing γ on the data you then evaluate on gives an optimistic error. The paper and the authors' MATLAB code both tune on the test
set. `sbl_assess()` avoids this:

- **Outer folds:** each outer training fold chooses γ by an inner `cv_sbl()`.
- **Held-out prediction:** each fitted model predicts its held-out fold.
- **Permuted control:** the whole pipeline is repeated on `nperm` permutations of the outcome.

```r
a <- sbl_assess(sim$W, sim$y, outer_folds = 5, inner_folds = 5, nperm = 20, ngamma = 20)
a
# Nested-CV assessment: held-out r = ..., RMSE / sd(y) = ...
# Permuted outcome (20): r mean ..., range [...]; permutation p = ...

a$observed$pred                     # held-out predictions
a$observed$rmse_ratio               # ~1 means no better than predicting the mean
a$observed$edge_freq                # V x V: share of outer folds that selected each edge
sapply(a$permuted, `[[`, "r")       # null distribution of the held-out correlation
```

Each permutation reruns the whole nested CV. That is `outer_folds` × (`inner_folds` × `ngamma` × `nstart` + `nstart`) fits, plus
the γ<sub>max</sub> searches. One nested-CV run for the simulated data above (*V* = 20, *n* = 100, `ngamma = 20`) took about
3 minutes on one core, so `nperm = 20` takes about an hour. Run permutations in parallel, for example with `parallel::parLapply()`
over calls with `nperm = 0` and permuted `y`, when you need many.

On this example the held-out correlation was 0.99 (RMSE / sd(*y*) = 0.13), against −0.25 to −0.13 for two permutations, and all
10 true signal edges were selected in every outer fold.

`edge_freq` is a simple selection-stability summary. The method gives point estimates only: it has no standard errors and no
posterior probabilities.

## Input

| | |
|---|---|
| `W` | a list of *n* symmetric *V* × *V* matrices, or a *V* × *V* × *n* array |
| entries | binary, weighted or signed; the only requirement is symmetry (the diagonal is set to zero, as the method requires) |
| node names | taken from `rownames(W[[1]])` if present, else `node1`, `node2`, ... |
| `y` | numeric vector of length *n*; adjust it for covariates beforehand (the model has an intercept only) |

For repeated fits on the same networks, convert once with `D <- sbl_data(W)` and pass `D` in place of `W`.

The penalty is not scale-free. Rescaling *W* or *y* changes the useful range of γ, which is why `sbl_gamma_max()` builds the grid
from the data.

## Output of a fit

| element | meaning |
|---|---|
| `alpha` | intercept |
| `lambda` | length-*K* component weights (0 = component removed) |
| `beta` | *V* × *K* component vectors; the nonzero rows of column *h* are the nodes of clique *h* |
| `B` | *V* × *V* coefficient matrix Σ λ<sub>h</sub>β<sub>h</sub>β<sub>h</sub><sup>⊤</sup>; the contribution of edge (*u*,*v*) is 2*B*<sub>uv</sub>*W*<sub>i,uv</sub> |
| `loss` | objective value at each iteration (non-increasing) |
| `iter`, `gamma`, `K`, `nodes` | bookkeeping |

## Algorithm and settings

- **Initialisation (Sec. IV-D):** β<sub>hu</sub> ~ U(−1, 1). α and λ come from least squares of *y* on β<sub>h</sub><sup>⊤</sup>*W*<sub>i</sub>β<sub>h</sub>. Starting at zero would get stuck at zero.
- **Updates:** coordinate descent with the closed-form soft-thresholding updates (16) for β, (19) for λ and (20) for α. A component with fewer than two nonzero entries is removed and not revived.
- **Active set:** after `fullit` full sweeps (default 100), only nonzero entries are updated, up to `maxit` iterations (default 1000).
- **Stopping rule:** stop when the objective decreases by less than `tol` (default 1e-5) times its previous value.
- **Multiple starts:** the problem is non-convex, so `sbl_fit()` runs `nstart` random starts (default 5) and keeps the lowest objective. The paper used 10 starts in simulations and 5 for the HCP data.
- **Cost:** an iteration costs O(*nKV*<sup>2</sup>) here. *B* enters only through the *n* × *V*<sup>2</sup> layout of the networks, so the authors' O(*V*<sup>3</sup>) intermediate matrices are not stored.

## Validation against the paper

`inst/scripts/reproduce_wang2019.R` reruns the simulation of Sec. V-B with this package:
- 100 replicates of *V* = 20 and *n* = 100, split into 50 for training and 50 for testing;
- *K* = 5, 10 starts, 50 γ values;
- the paper's γ rules: at high SNR, the largest γ whose test MSE is below 3% of the all-zero MSE; at low SNR, the minimum test MSE.

Values are mean (sd).

| setting | source | test MSE | TPR | FPR |
|---|---|---|---|---|
| high SNR | paper, Table II | 10.08 (4.51) | 0.848 (0.169) | 0.005 (0.007) |
| high SNR | sblreg | 9.41 (3.09) | 0.880 (0.152) | 0.006 (0.010) |
| low SNR | paper, Table III | 393.7 (159.2) | 0.539 (0.210) | 0.029 (0.038) |
| low SNR | sblreg | 384.95 (148.37) | 0.499 (0.175) | 0.042 (0.059) |

The simulated data and the random starts use R's random number generator, so individual replicates differ from the paper's
MATLAB runs. Agreement is in distribution. MSE and TPR match the paper at both noise levels. At low SNR the mean FPR is somewhat higher than reported (0.042 vs 0.029): a few replicates select many edges at the minimum-MSE γ.

```r
# from the installed package
source(system.file("scripts", "reproduce_wang2019.R", package = "sblreg"))   # or Rscript ... 100 4
```

## Practical notes and limitations

- **Small samples.** The paper's smallest simulated training set is 50 observations on 20 nodes (190 edges), with dense Gaussian
  networks. Behaviour with fewer observations, more nodes or sparse binary networks has not been studied. At small *n*, check
  `sbl_assess()` against its permuted control before reading the selected edges.
- **Effect sizes in practice.** In the paper's HCP application, with 565 training subjects, the best test MSE was a few percent
  below predicting the mean (Fig. 15, γ chosen on the test set). A held-out correlation near zero is a common outcome.
- **Signed networks.** Within a component the edge coefficients share a rank-one sign pattern λβ<sub>u</sub>β<sub>v</sub>, so
  positive and negative entries can cancel inside β<sup>⊤</sup>*Wβ*. Positive and negative parts can be analysed as separate
  networks.
- **Sparse binary networks.** An edge that is present in only a few observations carries little information. Consider removing
  edges that are rarely present before fitting. The paper gives no guidance on this.
- **Local optima.** Use several starts, and check `fit$loss`. Components can overlap: two components may select the same edge.
- **Covariates.** The model has no covariate term. Residualise *y* on the covariates first, inside each training fold if the
  covariate adjustment should not see held-out data.

## Function reference

| function | purpose |
|---|---|
| `sbl_fit()` | fit at one γ (best of several random starts) |
| `predict()` | predictions for new networks |
| `sbl_edges()`, `sbl_nodes()` | selected edges (with coefficients and components) and nodes |
| `sbl_gamma_max()`, `sbl_path()` | largest useful γ; fits over a γ grid |
| `cv_sbl()` | *k*-fold CV choice of γ and refit |
| `sbl_assess()` | nested CV held-out assessment with permuted-outcome control and edge selection frequencies |
| `sbl_data()` | convert networks once for repeated fits |
| `sbl_simulate()` | the simulation design of Sec. V-B of the paper |

## Relation to the original code

The authors' MATLAB implementation is at
[wangronglu/Symmetric-Bilinear-Regression](https://github.com/wangronglu/Symmetric-Bilinear-Regression). `sblreg` is an
independent R/C++ implementation of the published algorithm. It follows the paper's equations, matches the documented behaviour
of the MATLAB code (component removal, active set, stopping rule, γ<sub>max</sub> search), and adds nested cross-validation and
permutation assessment. Please cite the paper when you use the method.

## License

MIT © Sagnik Bhadury. See `LICENSE`.
