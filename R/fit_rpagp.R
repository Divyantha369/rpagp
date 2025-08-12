#' Fit the Random Phase-Amplitude Gaussian Process (RPAGP) Model
#'
#' @description
#' Runs a Markov Chain Monte Carlo (MCMC) sampler to fit the RPAGP model,
#' designed for trial-level analysis of Event-Related Potential (ERP) data.
#' It provides robust, high-performance inference on trial-specific amplitude (`beta`),
#' latency (`tau`), and the underlying structural ERP waveform (`f`).
#'
#' @param y A numeric matrix of observed trial data, with time points in rows and trials in columns.
#' @param n_iter The total number of MCMC iterations to run.
#' @param theta0 A named list of initial parameter values:
#'   \itemize{
#'     \item `rho`: A single numeric value for the GP length-scale parameter.
#'     \item `beta`: A numeric vector of initial values for the trial-specific amplitudes, length `ncol(y)`.
#'     \item `tau`: A numeric vector of initial values for the trial-specific latencies, length `ncol(y)`.
#'     \item `sigma`: A single numeric value for the white noise standard deviation.
#'   }
#' @param hyperparam A named list of hyperparameter values for the priors and proposals.
#' @param seed An optional integer seed for reproducible results. Defaults to 0 for a random seed.
#'
#' @return A list object of class `rpagp_fit` containing the full MCMC chains and diagnostics.
#'
#' @details
#' This function is a wrapper around a highly optimized C++ backend that uses Rcpp and RcppArmadillo.
#' The implementation leverages parallel processing via OpenMP.
#'
#' @export
#' @useDynLib rpagp, .registration = TRUE
#' @import Rcpp
fit_rpagp <- function(y, n_iter, theta0, hyperparam, seed = 0) {

  # --- Input Validation ---
  if (!is.matrix(y) || !is.numeric(y)) {
    stop("Input 'y' must be a numeric matrix.")
  }
  if (length(theta0$beta) != ncol(y) || length(theta0$tau) != ncol(y)) {
    stop("Length of initial 'beta' and 'tau' vectors must match the number of trials (columns in y).")
  }

  # Call the C++ backend function
  results <- fit_rpagp_cpp(
    y = y,
    n_iter = as.integer(n_iter),
    theta0 = theta0,
    hyperparam = hyperparam,
    seed = as.numeric(seed)
  )

  # Assign a class for custom methods (e.g., plot, summary)
  class(results) <- "rpagp_fit"

  return(results)
}
