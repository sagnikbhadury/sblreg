# Reproduce the simulation of Wang, Zhang & Dunson (2019), Sec. V-B (Tables II and III) with sblreg.
# V = 20, n = 100 split 50 train / 50 test; K = 5, tol 1e-5, 10 starts, 50 gamma values from 0.01 gamma_max.
# gamma rule as in the paper: high SNR, largest gamma whose test MSE is < 3% of the all-zero test MSE; low SNR, minimum test MSE.
# Usage: Rscript reproduce_wang2019.R [replicates = 100] [cores = 4]
library(sblreg); library(parallel)
a <- commandArgs(TRUE); R <- if (length(a)) as.integer(a[1]) else 100; NC <- if (length(a) > 1) as.integer(a[2]) else 4
one <- function(rep, snr) {
  sim <- sblreg::sbl_simulate(100, 20, snr, seed = 1000 + rep); tr <- 1:50; te <- 51:100
  D <- sblreg::sbl_data(sim$W); Dtr <- sblreg::sbl_data(sim$W[tr]); Dte <- sblreg::sbl_data(sim$W[te])
  path <- sblreg::sbl_path(Dtr, sim$y[tr], ngamma = 50, ratio = 0.01, nstart = 10)
  mse <- sapply(path$fits, function(f) mean((sim$y[te] - predict(f, Dte))^2)); mse0 <- mean((sim$y[te] - mean(sim$y[tr]))^2)
  j <- if (snr == "high") suppressWarnings(max(which(mse < 0.03 * mse0))) else which.min(mse); if (!is.finite(j)) j <- which.min(mse)
  sel <- path$fits[[j]]$B != 0; lt <- lower.tri(sel); tru <- sim$truth[lt]
  data.frame(rep = rep, snr = snr, mse = mse[j], tpr = mean(sel[lt][tru]), fpr = mean(sel[lt][!tru]))
}
g <- expand.grid(rep = seq_len(R), snr = c("high", "low"), stringsAsFactors = FALSE)
cl <- makeCluster(NC); clusterEvalQ(cl, library(sblreg)); clusterExport(cl, "one")
res <- do.call(rbind, parLapply(cl, split(g, seq_len(nrow(g))), function(x) one(x$rep, x$snr))); stopCluster(cl)
for (s in c("high", "low")) { d <- res[res$snr == s, ]
  cat(sprintf("%-4s SNR (%d reps): MSE %.2f (%.2f)  TPR %.3f (%.3f)  FPR %.3f (%.3f)\n", s, nrow(d), mean(d$mse), sd(d$mse),
              mean(d$tpr), sd(d$tpr), mean(d$fpr), sd(d$fpr))) }
