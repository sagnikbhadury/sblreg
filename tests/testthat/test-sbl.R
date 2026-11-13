test_that("objective decreases monotonically and a planted clique is recovered", {
  set.seed(1); V <- 15; n <- 80
  W <- lapply(1:n, function(i) { M <- matrix(rnorm(V * V), V); M <- M + t(M); diag(M) <- 0; M })
  y <- vapply(W, function(M) M[1, 2] + M[2, 3] + M[1, 3], numeric(1)) + rnorm(n, 0, 0.2)
  fit <- sbl_fit(W, y, gamma = 0.01, nstart = 3)
  expect_true(all(diff(fit$loss) <= 1e-10 * abs(utils::head(fit$loss, -1))))
  e <- sbl_edges(fit); top <- utils::head(e, 3)
  expect_setequal(paste(pmin(top$node1, top$node2), pmax(top$node1, top$node2)),
                  c("node1 node2", "node1 node3", "node2 node3"))
  expect_equal(top$coef, rep(1, 3), tolerance = 0.1)
})

test_that("large gamma gives the empty model and predict returns the intercept", {
  sim <- sbl_simulate(n = 40, V = 12, seed = 4)
  gmax <- sbl_gamma_max(sim$W, sim$y, nstart = 2)
  fit <- sbl_fit(sim$W, sim$y, gamma = 2 * gmax, nstart = 2)
  expect_equal(sum(fit$B != 0), 0)
  expect_equal(predict(fit, sim$W), rep(fit$alpha, 40))
  expect_equal(nrow(sbl_edges(fit)), 0)
})

test_that("inputs are validated and arrays are accepted", {
  sim <- sbl_simulate(n = 30, V = 12, seed = 5)
  A <- simplify2array(sim$W)
  f1 <- sbl_fit(sim$W, sim$y, gamma = 1, nstart = 1); f2 <- sbl_fit(A, sim$y, gamma = 1, nstart = 1)
  expect_equal(f1$B, f2$B)
  bad <- sim$W; bad[[1]][1, 2] <- bad[[1]][1, 2] + 1
  expect_error(sbl_fit(bad, sim$y, gamma = 1), "symmetric")
  expect_error(sbl_fit(sim$W, sim$y[-1], gamma = 1), "length")
})

test_that("high-SNR simulation of the paper: good edge recovery", {
  sim <- sbl_simulate(n = 100, V = 20, snr = "high", seed = 1001)
  cvf <- cv_sbl(sim$W[1:50], sim$y[1:50], ngamma = 20, nstart = 5)
  sel <- cvf$fit$B != 0
  tpr <- mean(sel[lower.tri(sel)][sim$truth[lower.tri(sim$truth)]])
  fpr <- mean(sel[lower.tri(sel)][!sim$truth[lower.tri(sim$truth)]])
  expect_gt(tpr, 0.5); expect_lt(fpr, 0.2)
})

test_that("nested assessment runs and returns honest held-out quantities", {
  sim <- sbl_simulate(n = 60, V = 12, snr = "high", seed = 7)
  a <- sbl_assess(sim$W, sim$y, ngamma = 8, nstart = 2, nperm = 1)
  expect_length(a$observed$pred, 60)
  expect_true(a$observed$r > a$permuted[[1]]$r)
  expect_true(all(a$observed$edge_freq >= 0 & a$observed$edge_freq <= 1))
})
