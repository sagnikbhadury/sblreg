// Symmetric bilinear regression with L1 penalty (SBL): coordinate descent, R port.
// Wang L, Zhang Z, Dunson DB (2019) "Symmetric bilinear regression for signal subgraph estimation", IEEE Trans Signal Process
// 67(7):1929-1940, doi:10.1109/TSP.2019.2899818 (PMC10198064). Model (7): E(y|W) = alpha + sum_h lambda_h beta_h' W beta_h; objective (9).
// Algorithm 1 with updates (16) beta_hu, (19) lambda_h, (20) alpha; initialisation Sec. IV-D (beta ~ U(-1,1), alpha and lambda by least
// squares on beta_h' W_i beta_h). Behaviour matched to the authors' MATLAB SBL1.m (github wangronglu/Symmetric-Bilinear-Regression):
// a component with < 2 nonzero entries is removed and not revived; after `fullit` full sweeps only nonzero beta entries are updated, up to
// `maxit`; stop when the decrease of the objective is < tol * |previous objective| or the objective is NaN.
// Independent R/C++ implementation; validated against the simulation of Sec. V-B of the paper (see README and tests).
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace Rcpp;

// Wu: field of V matrices, Wu(u) is n x V with row i = W_i[u, ]; Wf: n x V^2 with row i = vec(W_i)
static void bwb_col(const arma::mat& Wf, const arma::vec& b, arma::vec& out) {
  arma::mat bb = b * b.t(); out = Wf * arma::vectorise(bb);
}

// [[Rcpp::export]]
List sbl_fit_cpp(const arma::vec& y, const arma::field<arma::mat>& Wu, const arma::mat& Wf, int K, double gamma,
                 int maxit, int fullit, double tol, arma::mat Beta) {
  const int n = y.n_elem, V = Beta.n_rows;
  arma::vec LF(maxit, arma::fill::zeros);
  arma::uvec act(K);
  for (int h = 0; h < K; h++) { act(h) = arma::accu(Beta.col(h) != 0) > 1; if (!act(h)) Beta.col(h).zeros(); }
  arma::mat bWb(n, K, arma::fill::zeros); arma::vec bbs(K, arma::fill::zeros);
  arma::vec tmpc(n);
  for (int h = 0; h < K; h++) if (act(h)) {
    bwb_col(Wf, Beta.col(h), tmpc); bWb.col(h) = tmpc;
    double s = 0; arma::vec b = Beta.col(h);
    for (int u = 0; u < V; u++) for (int v = 0; v < u; v++) s += std::fabs(b(u) * b(v));
    bbs(h) = s;
  }
  // initial alpha, lambda by least squares (Sec. IV-D)
  arma::uvec ai = arma::find(act);
  arma::mat X(n, ai.n_elem + 1); X.col(0).ones();
  for (arma::uword j = 0; j < ai.n_elem; j++) X.col(j + 1) = bWb.col(ai(j));
  arma::vec cf = arma::solve(X.t() * X, X.t() * y);
  double alpha = cf(0); arma::vec Lambda(K, arma::fill::zeros);
  for (arma::uword j = 0; j < ai.n_elem; j++) Lambda(ai(j)) = cf(j + 1);
  arma::vec resid = y - X * cf;
  LF(0) = arma::accu(arma::square(resid)) / 2.0 / n + gamma * arma::dot(bbs, arma::abs(Lambda));
  int iter = 1; bool active_only = false;
  for (iter = 1; iter < maxit; iter++) {
    if (iter >= fullit) active_only = true;
    // beta updates, eq. (16)
    for (int h = 0; h < K; h++) {
      if (!act(h)) continue;
      for (int u = 0; u < V; u++) {
        if (active_only && Beta(u, h) == 0) continue;
        arma::vec comp = Lambda(h) * (Wu(u) * Beta.col(h));          // lambda_h W_i[u,] beta_h
        resid += 2.0 * Beta(u, h) * comp;                            // partial residual without beta_hu
        double A = 2.0 * arma::dot(resid, comp) / n, D = 4.0 * arma::dot(comp, comp) / n;
        if (D > 0) {
          double t = std::fabs(A) - gamma * std::fabs(Lambda(h)) * (arma::accu(arma::abs(Beta.col(h))) - std::fabs(Beta(u, h)));
          if (t > 0) { Beta(u, h) = (A > 0 ? 1.0 : -1.0) * std::exp(std::log(t) - std::log(D)); resid -= 2.0 * Beta(u, h) * comp; }
          else Beta(u, h) = 0;
        } else Beta(u, h) = 0;
      }
      if (arma::accu(Beta.col(h) != 0) < 2) { act(h) = 0; Beta.col(h).zeros(); Lambda(h) = 0; }
    }
    // lambda updates, eq. (19)
    for (int h = 0; h < K; h++) {
      if (!act(h)) continue;
      bwb_col(Wf, Beta.col(h), tmpc); bWb.col(h) = tmpc;
      double s = 0; arma::vec b = Beta.col(h);
      for (int u = 0; u < V; u++) for (int v = 0; v < u; v++) s += std::fabs(b(u) * b(v));
      bbs(h) = s;
      resid += Lambda(h) * bWb.col(h);
      double C = arma::dot(bWb.col(h), resid) / n, B = arma::dot(bWb.col(h), bWb.col(h)) / n;
      if (B > 0) {
        double t = std::fabs(C) - gamma * bbs(h);
        if (t > 0) { Lambda(h) = (C > 0 ? 1.0 : -1.0) * std::exp(std::log(t) - std::log(B)); resid -= Lambda(h) * bWb.col(h); }
        else { Lambda(h) = 0; act(h) = 0; Beta.col(h).zeros(); }
      } else { Lambda(h) = 0; act(h) = 0; Beta.col(h).zeros(); }
    }
    // alpha update, eq. (20)
    for (int h = 0; h < K; h++) if (!act(h)) { bWb.col(h).zeros(); }
    resid = y - bWb * Lambda; alpha = arma::mean(resid); resid -= alpha;
    LF(iter) = arma::accu(arma::square(resid)) / 2.0 / n + gamma * arma::dot(bbs % arma::conv_to<arma::vec>::from(act), arma::abs(Lambda));
    if ((LF(iter - 1) - LF(iter)) < tol * std::fabs(LF(iter - 1)) || std::isnan(LF(iter))) break;
  }
  int last = std::min(iter, maxit - 1);
  return List::create(_["alpha"] = alpha, _["Lambda"] = Lambda, _["Beta"] = Beta, _["LF"] = LF.head(last + 1), _["iter"] = last + 1);
}
