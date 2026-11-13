#' sblreg: Symmetric Bilinear Regression for Signal Subgraph Estimation
#'
#' Regression of a scalar outcome on a network (symmetric matrix) predictor with the L1-penalised symmetric bilinear model of
#' Wang, Zhang and Dunson (2019), \doi{10.1109/TSP.2019.2899818}:
#' \deqn{E(y_i \mid W_i) = \alpha + \sum_{h=1}^K \lambda_h \beta_h^\top W_i \beta_h,}
#' fitted by the coordinate descent of their Algorithm 1. Each nonzero component \eqn{\lambda_h \beta_h \beta_h^\top} picks out a small
#' clique of nodes whose connections are related to the outcome.
#'
#' @keywords internal
#' @useDynLib sblreg, .registration = TRUE
#' @importFrom Rcpp evalCpp
#' @importFrom stats cor sd runif rnorm predict
"_PACKAGE"

# ---- input handling ---------------------------------------------------------------------------------------------------------------

.as_list <- function(W) {
  if (is.array(W) && length(dim(W)) == 3) W <- lapply(seq_len(dim(W)[3]), function(i) W[, , i])
  if (!is.list(W) || !length(W)) stop("W must be a list of V x V matrices or a V x V x n array")
  V <- nrow(as.matrix(W[[1]]))
  lapply(W, function(M) {
    M <- as.matrix(M)
    if (nrow(M) != V || ncol(M) != V) stop("all networks must be V x V with the same V")
    if (max(abs(M - t(M))) > 1e-8 * max(1, max(abs(M)))) stop("networks must be symmetric")
    diag(M) <- 0; storage.mode(M) <- "double"; M
  })
}

#' Prepare networks for repeated fits
#'
#' Converts networks to the two layouts used by the coordinate descent. Preparing once and passing the result to [sbl_fit()],
#' [sbl_path()], [cv_sbl()] or [sbl_assess()] avoids repeating the conversion. Diagonals are set to zero, as the method requires
#' (no self-loops; Wang et al. 2019, Table I).
#'
#' @param W a list of \eqn{n} symmetric \eqn{V \times V} matrices, or a \eqn{V \times V \times n} array. Binary, weighted and signed
#'   entries are all allowed.
#' @return an object of class `sbl_data`.
#' @export
sbl_data <- function(W) {
  if (inherits(W, "sbl_data")) return(W)
  Wl <- .as_list(W); V <- nrow(Wl[[1]]); n <- length(Wl)
  nm <- rownames(Wl[[1]]); if (is.null(nm)) nm <- paste0("node", seq_len(V))
  Wf <- t(vapply(Wl, as.vector, numeric(V * V)))
  if (n == 1) Wf <- matrix(Wf, 1)
  Wu <- lapply(seq_len(V), function(u) { m <- t(vapply(Wl, function(M) M[u, ], numeric(V))); if (n == 1) m <- matrix(m, 1); m })
  structure(list(Wf = Wf, Wu = Wu, V = V, n = n, nodes = nm), class = "sbl_data")
}

.subset <- function(D, idx) structure(list(Wf = D$Wf[idx, , drop = FALSE], Wu = lapply(D$Wu, function(M) M[idx, , drop = FALSE]),
                                           V = D$V, n = length(idx), nodes = D$nodes), class = "sbl_data")

# ---- single fit ---------------------------------------------------------------------------------------------------------------------

.one_start <- function(y, D, K, gamma, start, maxit, fullit, tol, seed) {
  set.seed(seed + start - 1)
  B0 <- matrix(1 - 2 * runif(D$V * K), D$V, K)          # beta ~ U(-1, 1) (Sec. IV-D)
  sbl_fit_cpp(y, D$Wu, D$Wf, K, gamma, maxit, fullit, tol, B0)
}

#' Fit the L1-penalised symmetric bilinear model at one penalty value
#'
#' Runs Algorithm 1 of Wang et al. (2019) from `nstart` random starts and keeps the start with the lowest final objective (as in
#' the authors' tuning code). The problem is non-convex, so several starts are recommended (Sec. IV-D; the paper used 10 in
#' simulations and 5 for the HCP data).
#'
#' @param W networks (list, array, or [sbl_data()] object).
#' @param y numeric outcome of length \eqn{n}.
#' @param gamma penalty factor \eqn{\gamma \ge 0}.
#' @param K upper bound on the number of components (Sec. IV-D: the L1 penalty removes unneeded components).
#' @param nstart number of random starts.
#' @param maxit maximum iterations; `fullit` full sweeps over all entries are made before only nonzero entries are updated.
#' @param fullit see `maxit`.
#' @param tol stop when the objective decreases by less than `tol` times its previous value.
#' @param seed base random seed; start `s` uses `seed + s - 1`.
#' @return an object of class `sbl`: `alpha`, `lambda` (length K), `beta` (V x K), `B` (V x V coefficient matrix
#'   \eqn{\sum_h \lambda_h \beta_h \beta_h^\top}), `loss` (objective by iteration), `iter`, `gamma`, `nodes`.
#' @details The model contribution of edge \eqn{(u,v)} is \eqn{2 B_{uv} W_{i,uv}}, because \eqn{W_i} is symmetric; [sbl_edges()]
#'   reports \eqn{2 B_{uv}} as the edge coefficient.
#' @examples
#' sim <- sbl_simulate(n = 60, V = 15, seed = 1)
#' fit <- sbl_fit(sim$W, sim$y, gamma = 0.05)
#' sbl_edges(fit)
#' @export
sbl_fit <- function(W, y, gamma, K = 5, nstart = 5, maxit = 1000, fullit = 100, tol = 1e-5, seed = 1) {
  D <- sbl_data(W); y <- as.numeric(y)
  if (length(y) != D$n) stop("length(y) must equal the number of networks")
  if (fullit < 2 || maxit < fullit) stop("need 2 <= fullit <= maxit")
  best <- NULL
  for (s in seq_len(nstart)) {
    f <- .one_start(y, D, K, gamma, s, maxit, fullit, tol, seed)
    if (is.null(best) || utils::tail(f$LF, 1) < utils::tail(best$LF, 1)) best <- f
  }
  lam <- as.vector(best$Lambda); Bt <- best$Beta
  B <- Bt %*% diag(lam, K) %*% t(Bt); dimnames(B) <- list(D$nodes, D$nodes); rownames(Bt) <- D$nodes
  structure(list(alpha = best$alpha, lambda = lam, beta = Bt, B = B, loss = as.vector(best$LF), iter = best$iter, gamma = gamma,
                 K = K, nodes = D$nodes), class = "sbl")
}

#' @export
print.sbl <- function(x, ...) {
  e <- sum(x$B[lower.tri(x$B)] != 0)
  cat(sprintf("Symmetric bilinear regression (gamma = %.4g, K = %d)\n  active components: %d; nonzero edges: %d; nodes involved: %d\n",
              x$gamma, x$K, sum(x$lambda != 0), e, sum(rowSums(x$B != 0) > 0)))
  invisible(x)
}

#' Predict from a fitted symmetric bilinear model
#' @param object an `sbl` fit.
#' @param newW networks to predict for (list, array, or [sbl_data()] object).
#' @param ... unused.
#' @return numeric vector of predictions \eqn{\hat\alpha + \sum_{u,v} \hat B_{uv} W_{i,uv}}.
#' @export
predict.sbl <- function(object, newW, ...) {
  D <- sbl_data(newW)
  if (D$V != nrow(object$B)) stop("newW has a different number of nodes from the fit")
  as.vector(object$alpha + D$Wf %*% as.vector(object$B))
}

#' Edges and nodes selected by a fit
#' @param fit an `sbl` fit.
#' @return `sbl_edges()`: data frame of nonzero edges with coefficient \eqn{2 \hat B_{uv}} and the components that contain them;
#'   `sbl_nodes()`: data frame of nodes with the number of selected edges they touch.
#' @export
sbl_edges <- function(fit) {
  B <- fit$B; ix <- which(lower.tri(B) & B != 0, arr.ind = TRUE)
  if (!nrow(ix)) return(data.frame(node1 = character(), node2 = character(), coef = numeric(), components = character()))
  comp <- apply(ix, 1, function(r) paste(which(fit$beta[r[1], ] != 0 & fit$beta[r[2], ] != 0 & fit$lambda != 0), collapse = ","))
  d <- data.frame(node1 = fit$nodes[ix[, 2]], node2 = fit$nodes[ix[, 1]], coef = 2 * B[ix], components = comp)
  d[order(-abs(d$coef)), , drop = FALSE]
}

#' @rdname sbl_edges
#' @export
sbl_nodes <- function(fit) {
  nz <- fit$B != 0; diag(nz) <- FALSE
  d <- data.frame(node = fit$nodes, n_edges = rowSums(nz)); d <- d[d$n_edges > 0, , drop = FALSE]
  d[order(-d$n_edges), , drop = FALSE]
}

# ---- penalty path -------------------------------------------------------------------------------------------------------------------

#' Largest useful penalty and the penalty grid
#'
#' `sbl_gamma_max()` finds (up to a factor of 2) the smallest \eqn{\gamma} at which every component is zero, by the doubling and
#' halving search of the authors' tuning code. `sbl_path()` fits the model on a log-spaced grid from `ratio * gamma_max` to
#' `gamma_max` (Sec. V-B1 of the paper: 50 values, ratio 0.01).
#' @inheritParams sbl_fit
#' @param ngamma number of grid values.
#' @param ratio smallest grid value as a fraction of `gamma_max`.
#' @param gammas optional user-supplied grid (overrides `ngamma` and `ratio`).
#' @return `sbl_gamma_max()`: a number. `sbl_path()`: an object of class `sbl_path` with `gammas` and a list of `fits`.
#' @export
sbl_gamma_max <- function(W, y, K = 5, nstart = 5, maxit = 1000, fullit = 100, tol = 1e-5, seed = 1) {
  D <- sbl_data(W); y <- as.numeric(y)
  nz <- function(g) { for (s in seq_len(nstart)) if (any(.one_start(y, D, K, g, s, maxit, fullit, tol, seed)$Beta != 0)) return(TRUE); FALSE }
  g <- 1; s <- nz(g)
  while (s) { g <- 2 * g; s <- nz(g) }
  while (!s) { g <- g / 2; s <- nz(g); if (g < 1e-12) stop("no nonzero fit found at any gamma") }
  2 * g
}

#' @rdname sbl_gamma_max
#' @export
sbl_path <- function(W, y, ngamma = 50, ratio = 0.01, gammas = NULL, K = 5, nstart = 5, maxit = 1000, fullit = 100, tol = 1e-5, seed = 1) {
  D <- sbl_data(W)
  if (is.null(gammas)) gammas <- .grid(sbl_gamma_max(D, y, K, nstart, maxit, fullit, tol, seed), ngamma, ratio)
  fits <- lapply(gammas, function(g) sbl_fit(D, y, g, K, nstart, maxit, fullit, tol, seed))
  structure(list(gammas = gammas, fits = fits), class = "sbl_path")
}
