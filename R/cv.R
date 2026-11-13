.grid <- function(gmax, ngamma, ratio) exp(seq(log(ratio * gmax), log(gmax), length.out = ngamma))

.folds <- function(n, nfolds, seed) { set.seed(seed); sample(rep(seq_len(nfolds), length.out = n)) }

#' Choose the penalty by cross-validation
#'
#' K-fold cross-validation over a log-spaced \eqn{\gamma} grid, choosing the value with the smallest mean held-out squared error
#' (the rule used for the low-SNR simulation and the HCP application in Wang et al. 2019), then refitting on all data.
#' The grid is built from `gamma_max` of the full data.
#'
#' @inheritParams sbl_fit
#' @inheritParams sbl_path
#' @param nfolds number of folds.
#' @param fold_seed seed for the fold assignment.
#' @param refit refit on all data at the chosen \eqn{\gamma}.
#' @return an object of class `cv_sbl`: `gammas`, `cvm` (mean CV error per \eqn{\gamma}), `cvse`, `gamma_min`, `index_min`,
#'   `folds`, and `fit` (the refit, or `NULL`).
#' @examples
#' sim <- sbl_simulate(n = 60, V = 12, seed = 2)
#' cvf <- cv_sbl(sim$W, sim$y, ngamma = 10, nfolds = 5)
#' cvf$gamma_min
#' sbl_edges(cvf$fit)
#' @export
cv_sbl <- function(W, y, nfolds = 5, ngamma = 50, ratio = 0.01, gammas = NULL, K = 5, nstart = 5, maxit = 1000, fullit = 100,
                   tol = 1e-5, seed = 1, fold_seed = 1, refit = TRUE) {
  D <- sbl_data(W); y <- as.numeric(y); n <- D$n
  if (n < 2 * nfolds) stop("too few observations for ", nfolds, "-fold CV")
  if (is.null(gammas)) gammas <- .grid(sbl_gamma_max(D, y, K, nstart, maxit, fullit, tol, seed), ngamma, ratio)
  folds <- .folds(n, nfolds, fold_seed)
  err <- matrix(NA, length(gammas), nfolds)
  for (f in seq_len(nfolds)) {
    tr <- which(folds != f); te <- which(folds == f); Dtr <- .subset(D, tr); Dte <- .subset(D, te)
    for (j in seq_along(gammas)) err[j, f] <- mean((y[te] - predict(sbl_fit(Dtr, y[tr], gammas[j], K, nstart, maxit, fullit, tol, seed), Dte))^2)
  }
  cvm <- rowMeans(err); j <- which.min(cvm)
  structure(list(gammas = gammas, cvm = cvm, cvse = apply(err, 1, sd) / sqrt(nfolds), gamma_min = gammas[j], index_min = j,
                 folds = folds, fit = if (refit) sbl_fit(D, y, gammas[j], K, nstart, maxit, fullit, tol, seed) else NULL),
            class = "cv_sbl")
}

#' @export
print.cv_sbl <- function(x, ...) {
  cat(sprintf("Cross-validated symmetric bilinear regression: %d gamma values, chosen gamma = %.4g (index %d)\n", length(x$gammas),
              x$gamma_min, x$index_min))
  if (!is.null(x$fit)) print(x$fit)
  invisible(x)
}

#' Honest held-out assessment with a permuted-outcome control
#'
#' Nested cross-validation: in each outer fold, \eqn{\gamma} is chosen by [cv_sbl()] on the training part only, and the model is
#' evaluated on the held-out part. The same pipeline is run on `nperm` permutations of the outcome. The paper and the authors' code
#' choose \eqn{\gamma} on the evaluation set, which makes the reported error optimistic; this function does not.
#'
#' @inheritParams cv_sbl
#' @param outer_folds number of outer folds.
#' @param inner_folds number of inner folds used to choose \eqn{\gamma}.
#' @param nperm number of outcome permutations for the control (0 for none).
#' @param perm_seed seed for the permutations.
#' @return an object of class `sbl_assess` with `observed` and `permuted` (a list), each holding the held-out predictions `pred`,
#'   correlation `r`, `rmse_ratio` (held-out RMSE / sd(y); about 1 means no better than the mean), chosen `gamma` per fold,
#'   and edge selection frequencies `edge_freq` (V x V, fraction of outer folds selecting each edge); plus `p_value` (share of
#'   permutations with r at least the observed r, with the +1 correction) when `nperm > 0`.
#' @examples
#' \donttest{
#' sim <- sbl_simulate(n = 80, V = 12, snr = "high", seed = 3)
#' a <- sbl_assess(sim$W, sim$y, ngamma = 10, nperm = 2)
#' a
#' }
#' @export
sbl_assess <- function(W, y, outer_folds = 5, inner_folds = 5, nperm = 1, ngamma = 50, ratio = 0.01, K = 5, nstart = 5,
                       maxit = 1000, fullit = 100, tol = 1e-5, seed = 1, fold_seed = 1, perm_seed = 2026) {
  D <- sbl_data(W); y <- as.numeric(y); n <- D$n
  folds <- .folds(n, outer_folds, fold_seed)
  run <- function(yy) {
    pred <- rep(NA_real_, n); gam <- numeric(outer_folds); sel <- array(FALSE, c(D$V, D$V, outer_folds))
    for (f in seq_len(outer_folds)) {
      tr <- which(folds != f); te <- which(folds == f)
      cvf <- cv_sbl(.subset(D, tr), yy[tr], inner_folds, ngamma, ratio, NULL, K, nstart, maxit, fullit, tol, seed, fold_seed + f)
      pred[te] <- predict(cvf$fit, .subset(D, te)); gam[f] <- cvf$gamma_min; sel[, , f] <- cvf$fit$B != 0
    }
    ef <- apply(sel, c(1, 2), mean); dimnames(ef) <- list(D$nodes, D$nodes)
    list(pred = pred, r = suppressWarnings(cor(pred, yy)), rmse_ratio = sqrt(mean((pred - yy)^2)) / sd(yy), gamma = gam, edge_freq = ef)
  }
  obs <- run(y)
  perm <- list()
  if (nperm > 0) { set.seed(perm_seed); P <- replicate(nperm, sample(y)); perm <- lapply(seq_len(nperm), function(k) run(P[, k])) }
  pr <- vapply(perm, function(p) p$r, numeric(1))
  structure(list(observed = obs, permuted = perm, folds = folds,
                 p_value = if (nperm > 0) (1 + sum(pr >= obs$r, na.rm = TRUE)) / (1 + nperm) else NA_real_), class = "sbl_assess")
}

#' @export
print.sbl_assess <- function(x, ...) {
  cat(sprintf("Nested-CV assessment: held-out r = %.3f, RMSE / sd(y) = %.3f\n", x$observed$r, x$observed$rmse_ratio))
  if (length(x$permuted)) {
    pr <- vapply(x$permuted, function(p) p$r, numeric(1))
    cat(sprintf("Permuted outcome (%d): r mean %.3f, range [%.3f, %.3f]; permutation p = %.3f\n", length(pr), mean(pr, na.rm = TRUE),
                min(pr, na.rm = TRUE), max(pr, na.rm = TRUE), x$p_value))
  }
  ef <- x$observed$edge_freq; cat(sprintf("Edges selected in all outer folds: %d\n", sum(ef[lower.tri(ef)] == 1)))
  invisible(x)
}

#' Simulate networks and outcome as in Wang et al. (2019), Sec. V-B
#'
#' \eqn{W_i = \sum_{h=1}^{10} \lambda_{ih} q_h q_h^\top + \Delta_i}, with \eqn{q_h} binary and \eqn{h+1} nonzero entries,
#' \eqn{\lambda_{ih} \sim N(0,1)}, \eqn{\Delta_{i,uv} \sim N(0, 0.1^2)}; \eqn{y_i = \sum_{h=1}^3 q_h^\top W_i q_h + \epsilon_i}, with
#' noise sd 10\% (`"high"`) or 100\% (`"low"`) of sd\eqn{(E[y \mid W])}.
#' @param n number of networks.
#' @param V number of nodes (the paper uses 20).
#' @param snr `"high"` or `"low"`.
#' @param seed random seed.
#' @return list with `W` (list of matrices), `y`, and `truth` (logical V x V matrix of true signal edges).
#' @export
sbl_simulate <- function(n = 100, V = 20, snr = c("high", "low"), seed = NULL) {
  snr <- match.arg(snr); if (!is.null(seed)) set.seed(seed)
  if (V < 11) stop("V must be at least 11 (the largest basis subgraph has 11 nodes)")
  q <- lapply(1:10, function(h) { v <- numeric(V); v[sample(V, h + 1)] <- 1; v })
  W <- lapply(seq_len(n), function(i) {
    M <- Reduce(`+`, lapply(1:10, function(h) rnorm(1) * tcrossprod(q[[h]])))
    E <- matrix(0, V, V); E[lower.tri(E)] <- rnorm(V * (V - 1) / 2, 0, 0.1); M <- M + E + t(E); diag(M) <- 0; M })
  mu <- vapply(W, function(M) sum(vapply(1:3, function(h) drop(t(q[[h]]) %*% M %*% q[[h]]), numeric(1))), numeric(1))
  y <- mu + rnorm(n, 0, (if (snr == "high") 0.1 else 1) * sd(mu))
  Bt <- Reduce(`+`, lapply(1:3, function(h) tcrossprod(q[[h]]))); truth <- Bt != 0; diag(truth) <- FALSE
  list(W = W, y = y, truth = truth)
}
