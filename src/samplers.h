#ifndef RPAGP_SAMPLERS_H
#define RPAGP_SAMPLERS_H

#include "rpagp_types.h"

// Kernel & Covariance Functions 
arma::mat sq_exp_kernel_adaptive(const arma::vec& x, double rho, double alpha);
arma::mat get_Sigma_nu(double sigma, int n_time);
arma::mat get_trial_K_i(const arma::vec& x, double rho, double tau, double beta);
arma::mat get_y_hat_matrix(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, const arma::mat& K_f_inv);

// Likelihood & Prior Functions 
double likelihood(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, const arma::mat& K_f, const arma::mat& K_f_inv);
double likelihood_single_trial(int i, const arma::vec& y_i, const arma::vec& f, double beta_i, double tau_i, double rho, double sigma, const arma::mat& K_f_inv, const arma::mat& Sigma_nu, const arma::vec& x);
double prior_beta_lognormal(double beta, double mu_log, double sigma_log);
double prior_rho_lognormal(double rho, double mu_log, double sigma_log);

//  MCMC Sampling Functions 
arma::mat sample_f_maxnorm(const arma::mat& y, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, int n_draws);
arma::mat compute_orthogonal_basis(int n);
arma::vec sample_eta_ess(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& eta, const arma::mat& Ur, double rho, double sigma, double tau_prior_sd);
arma::vec sample_beta_lognormal_mh(const arma::mat& y, const arma::vec& f, const arma::vec& taus, double rho, double sigma, const arma::vec& betas_current, double beta_prior_mu_log, double beta_prior_sigma_log, double beta_proposal_sd);
double sample_rho_logspace_lognormal(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, double mu_log, double sigma_log, double rho_proposal_sd);
double sample_sigma(const arma::mat& z, double sigma_prior_shape, double sigma_prior_scale);

#endif // RPAGP_SAMPLERS_H