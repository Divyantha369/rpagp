// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]
#include <RcppArmadillo.h>
#include "rpagp_types.h"
#include "utils.h"
#include "samplers.h"

//' @name fit_rpagp_cpp
//' @title The C++ backend for the RPAGP model sampler.
//' @description This is the core C++ function that runs the MCMC sampler.
//' It is not intended to be called directly by the user. Please use the
//' R wrapper function \code{fit_rpagp()} instead.
//' @param y A matrix of observed trial data.
//' @param n_iter The total number of MCMC iterations.
//' @param theta0 A list of initial parameter values.
//' @param hyperparam A list of hyperparameter values.
//' @param seed An integer seed for reproducibility.
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List fit_rpagp_cpp(const arma::mat& y, int n_iter,
                         const Rcpp::List& theta0,
                         const Rcpp::List& hyperparam,
                         uint64_t seed = 0) {
    global_diagnostics.reset();
    global_kernel_cache.reset();
    configure_parallel_environment(seed);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    int n_time = y.n_rows;
    int n = y.n_cols;
    arma::vec x = arma::linspace(0, 1, n_time);
    
    double rho0 = Rcpp::as<double>(theta0["rho"]);
    arma::vec beta0 = Rcpp::as<arma::vec>(theta0["beta"]);
    arma::vec tau0 = Rcpp::as<arma::vec>(theta0["tau"]);
    double sigma0 = Rcpp::as<double>(theta0["sigma"]);

    double tau_prior_sd = Rcpp::as<double>(hyperparam["tau_prior_sd"]);
    double rho_prior_mu_log = Rcpp::as<double>(hyperparam["rho_prior_mu_log"]);
    double rho_prior_sigma_log = Rcpp::as<double>(hyperparam["rho_prior_sigma_log"]);
    double rho_proposal_sd = Rcpp::as<double>(hyperparam["rho_proposal_sd"]);
    double beta_prior_mu_log = Rcpp::as<double>(hyperparam["beta_prior_mu_log"]);
    double beta_prior_sigma_log = Rcpp::as<double>(hyperparam["beta_prior_sigma_log"]);
    double beta_proposal_sd = Rcpp::as<double>(hyperparam["beta_proposal_sd"]);
    double sigma_prior_shape = Rcpp::as<double>(hyperparam["sigma_prior_shape"]);
    double sigma_prior_scale = Rcpp::as<double>(hyperparam["sigma_prior_scale"]);

    arma::mat Ur = compute_orthogonal_basis(n);
    arma::vec eta0 = Ur.t() * tau0;
    
    Rcpp::List chain(n_iter), chain_f(n_iter), chain_y_hat(n_iter), chain_z(n_iter), chain_eta(n_iter);
    Rcpp::NumericVector likelihood_values(n_iter);
    
    chain[0] = Rcpp::clone(theta0);
    chain_eta[0] = eta0;
    
    Rcpp::Rcout << "Sampling initial f...\n";
    arma::vec f = sample_f_maxnorm(y, beta0, tau0, rho0, sigma0, 1).col(0);
    chain_f[0] = f;
    
    auto kernel_pair = global_kernel_cache.get_kernel_matrices(x, rho0);
    likelihood_values[0] = likelihood(y, f, beta0, tau0, rho0, sigma0, kernel_pair.first, kernel_pair.second);
    
    Rcpp::Rcout << "Starting MCMC... Initial likelihood: " << likelihood_values[0] << "\n";
    int rho_accepts = 0;
    
    for (int iter = 1; iter < n_iter; ++iter) {
        if ((iter % (n_iter / 10)) == 0) {
            Rcpp::Rcout << " ... " << static_cast<int>((iter / static_cast<double>(n_iter)) * 100) << "% \n";
        }
        
        Rcpp::List current = chain[iter - 1];
        double rho = Rcpp::as<double>(current["rho"]);
        arma::vec beta = Rcpp::as<arma::vec>(current["beta"]);
        arma::vec eta = Rcpp::as<arma::vec>(chain_eta[iter - 1]);
        double sigma = Rcpp::as<double>(current["sigma"]);
        arma::vec tau = Ur * eta;
        
        try {
            f = sample_f_maxnorm(y, beta, tau, rho, sigma, 1).col(0);
            beta = sample_beta_lognormal_mh(y, f, tau, rho, sigma, beta, beta_prior_mu_log, beta_prior_sigma_log, beta_proposal_sd);
            f = sample_f_maxnorm(y, beta, tau, rho, sigma, 1).col(0);
            eta = sample_eta_ess(y, f, beta, eta, Ur, rho, sigma, tau_prior_sd);
            tau = Ur * eta;
            
            double rho_old = rho;
            rho = sample_rho_logspace_lognormal(y, f, beta, tau, rho, sigma, rho_prior_mu_log, rho_prior_sigma_log, rho_proposal_sd);
            if (rho != rho_old) rho_accepts++;
            
            auto kp_new = global_kernel_cache.get_kernel_matrices(x, rho);
            arma::mat y_hat = get_y_hat_matrix(y, f, beta, tau, rho, kp_new.second);
            arma::mat z = y - y_hat;
            sigma = sample_sigma(z, sigma_prior_shape, sigma_prior_scale);
            
            likelihood_values[iter] = likelihood(y, f, beta, tau, rho, sigma, kp_new.first, kp_new.second);
            
            Rcpp::List current_new = Rcpp::clone(current);
            current_new["beta"] = beta;
            current_new["tau"] = tau;
            current_new["rho"] = rho;
            current_new["sigma"] = sigma;
            
            chain[iter] = current_new;
            chain_f[iter] = f;
            chain_y_hat[iter] = y_hat;
            chain_z[iter] = z;
            chain_eta[iter] = eta;
            
        } catch (const std::exception& e) {
            Rcpp::Rcout << "Error in iteration " << iter << ": " << e.what() << "\n";
            chain[iter] = chain[iter - 1];
            chain_f[iter] = chain_f[iter - 1];
            chain_y_hat[iter] = chain_y_hat[iter - 1];
            chain_z[iter] = chain_z[iter - 1];
            chain_eta[iter] = chain_eta[iter - 1];
            likelihood_values[iter] = likelihood_values[iter - 1];
        }
    }
    
    double runtime = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start_time).count();
    Rcpp::Rcout << "\nRPAGP MCMC completed in " << runtime << " seconds\n";
    global_kernel_cache.print_stats();
    global_diagnostics.report_summary();
    
    return Rcpp::List::create(
        Rcpp::Named("chain") = chain,
        Rcpp::Named("chain_f") = chain_f,
        Rcpp::Named("chain_y_hat") = chain_y_hat,
        Rcpp::Named("chain_z") = chain_z,
        Rcpp::Named("chain_eta") = chain_eta,
        Rcpp::Named("runtime") = runtime,
        Rcpp::Named("rho_acceptance_rate") = static_cast<double>(rho_accepts) / std::max(1, n_iter - 1)
    );
}