#include "samplers.h"
#include "utils.h"

// Constants specific to samplers 
constexpr int parallel_threshold = 500;
constexpr double machine_epsilon = std::numeric_limits<double>::epsilon();
constexpr double likelihood_floor = -1e8;

// Kernel & Covariance Function Definitions 
arma::mat sq_exp_kernel_adaptive(const arma::vec& x, double rho, double alpha) {
    int n = x.n_elem;
    double rho_safe = std::max(rho, 0.01);
    double rho2 = rho_safe * rho_safe;
    double alpha2 = alpha * alpha;
    
    arma::vec v(n);
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        double xi = x(i);
        v(i) = alpha2 * std::exp(-rho2 / 2.0 * (xi * xi));
    }
    arma::mat K = arma::toeplitz(v);
    RegularizationResult reg_result = adaptive_regularize(K, alpha2, true);
    return reg_result.regularized_matrix;
}

arma::mat get_Sigma_nu(double sigma, int n_time) {
    double sigma2 = sigma * sigma;
    return sigma2 * arma::eye(n_time, n_time);
}

arma::mat get_trial_K_i_base(const arma::vec& x, double rho, double tau) {
    int n_time = x.n_elem;
    arma::mat K(n_time, n_time);
    double rho_safe = std::max(rho, 0.01);
    double rho2 = rho_safe * rho_safe;
    arma::vec x_shifted = x - tau;
    
    #pragma omp parallel for schedule(static) if (n_time > parallel_threshold)
    for (int i = 0; i < n_time; i++) {
        for (int j = 0; j < n_time; j++) {
            double dist = x_shifted(i) - x(j);
            K(i, j) = std::exp(-0.5 * rho2 * (dist * dist));
        }
    }
    return K;
}

arma::mat get_trial_K_i(const arma::vec& x, double rho, double tau, double beta) {
    return beta * get_trial_K_i_base(x, rho, tau);
}

arma::mat get_Sigma_y_i(double beta_i, const arma::mat& K_f, const arma::mat& Sigma_nu) {
    return (beta_i * beta_i) * K_f + Sigma_nu;
}

arma::mat get_y_hat_matrix(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, const arma::mat& K_f_inv) {
    int n_time = y.n_rows;
    int n = y.n_cols;
    arma::mat y_hat(n_time, n);
    arma::vec x = arma::linspace(0, 1, n_time);

    #pragma omp parallel for schedule(dynamic) if(n > parallel_threshold)
    for (int i = 0; i < n; i++) {
        arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
        y_hat.col(i) = K_i * K_f_inv * f;
    }
    return y_hat;
}

// Likelihood & Prior Function Definitions 
double likelihood(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, const arma::mat& K_f, const arma::mat& K_f_inv) {
    int n = y.n_cols;
    int n_time = y.n_rows;
    arma::vec x = arma::linspace(0, 1, n_time);
    arma::mat Sigma_nu = get_Sigma_nu(sigma, n_time);
    double total_log_lik = 0.0;
    
    #pragma omp parallel for schedule(dynamic) reduction(+:total_log_lik)
    for (int i = 0; i < n; i++) {
        try {
            arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
            arma::vec mu = K_i * K_f_inv * f;
            arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
            arma::mat Sigma_i_cond = Sigma_y_i - K_i.t() * K_f_inv * K_i;
            double trial_ll = dmvnorm_chol(y.col(i), mu, Sigma_i_cond, true);
            total_log_lik += trial_ll;
        } catch (...) {
            total_log_lik += likelihood_floor;
        }
    }
    return total_log_lik;
}

double likelihood_single_trial(int i, const arma::vec& y_i, const arma::vec& f, double beta_i, double tau_i, double rho, double sigma, const arma::mat& K_f_inv, const arma::mat& Sigma_nu, const arma::vec& x) {
    try {
        auto kernel_pair = global_kernel_cache.get_kernel_matrices(x, rho);
        arma::mat K_i = get_trial_K_i(x, rho, tau_i, beta_i);
        arma::mat Sigma_y_i = get_Sigma_y_i(beta_i, kernel_pair.first, Sigma_nu);
        arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
        arma::vec mu = K_i * K_f_inv * f;
        return dmvnorm_chol(y_i, mu, Sigma_i, true);
    } catch (...) {
        return likelihood_floor;
    }
}

double prior_beta_lognormal(double beta, double mu_log, double sigma_log) {
    if (beta <= 0 || sigma_log <= 0) return likelihood_floor;
    double log_beta = std::log(beta);
    double z = (log_beta - mu_log) / sigma_log;
    return -log_beta - std::log(sigma_log) - 0.5 * std::log(2.0 * M_PI) - 0.5 * (z*z);
}

double prior_rho_lognormal(double rho, double mu_log, double sigma_log) {
    if (rho <= 0 || sigma_log <= 0) return likelihood_floor;
    double log_rho = std::log(rho);
    double z = (log_rho - mu_log) / sigma_log;
    return -log_rho - std::log(sigma_log) - 0.5 * std::log(2.0 * M_PI) - 0.5 * (z*z);
}

// MCMC Sampling Function Definitions 
arma::mat sample_f_maxnorm(const arma::mat& y, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, int n_draws) {
    int n_time = y.n_rows;
    int n = y.n_cols;
    arma::mat f_draws(n_time, n_draws);
    arma::vec x = arma::linspace(0, 1, n_time);

    auto kernel_pair = global_kernel_cache.get_kernel_matrices(x, rho);
    arma::mat K_f = kernel_pair.first;
    arma::mat K_f_inv = kernel_pair.second;
    arma::mat Sigma_nu = get_Sigma_nu(sigma, n_time);

    #pragma omp parallel for schedule(dynamic) if(n_draws > 1)
    for (int iter = 0; iter < n_draws; iter++) {
        arma::mat A = K_f_inv;
        arma::vec b = arma::zeros(n_time);

        #pragma omp parallel
        {
            arma::mat A_local = arma::zeros<arma::mat>(n_time, n_time);
            arma::vec b_local = arma::zeros<arma::vec>(n_time);

            #pragma omp for schedule(dynamic) nowait
            for (int i = 0; i < n; i++) {
                arma::mat K_i = get_trial_K_i(x, rho, taus(i), betas(i));
                arma::mat Sigma_y_i = get_Sigma_y_i(betas(i), K_f, Sigma_nu);
                arma::mat Sigma_i = Sigma_y_i - K_i.t() * K_f_inv * K_i;
                Sigma_i = (Sigma_i + Sigma_i.t()) / 2.0;
                
                RegularizationResult reg = adaptive_regularize(Sigma_i, 1.0, true);
                arma::mat L_mat = K_i * K_f_inv;
                arma::mat G;

                if (!arma::solve(G, reg.regularized_matrix, L_mat, arma::solve_opts::fast + arma::solve_opts::likely_sympd)) {
                    G = arma::pinv(reg.regularized_matrix) * L_mat;
                    global_diagnostics.add_trial_error(i, "arma::solve failed in sample_f; used pinv");
                }
                
                A_local += L_mat.t() * G;
                b_local += G.t() * y.col(i);
            }

            #pragma omp critical
            {
                A += A_local;
                b += b_local;
            }
        }

        A = (A + A.t()) / 2.0;
        arma::mat K_f_post = robust_inv(A, true);
        arma::vec mu_post = K_f_post * b;

        arma::vec z = arma::randn<arma::vec>(n_time);
        arma::mat L_chol;
        if (arma::chol(L_chol, K_f_post, "lower")) {
            f_draws.col(iter) = mu_post + L_chol * z;
        } else {
            f_draws.col(iter) = mu_post;
        }
        
        double max_abs_f = arma::max(arma::abs(f_draws.col(iter)));
        if (max_abs_f > machine_epsilon) {
            f_draws.col(iter) /= max_abs_f;
            global_diagnostics.max_norm_applications++;
        }
    }
    return f_draws;
}

arma::mat compute_orthogonal_basis(int n) {
    arma::mat Z = arma::eye(n, n);
    Z.col(0).fill(1.0 / std::sqrt(static_cast<double>(n)));
    arma::mat Q, R;
    arma::qr(Q, R, Z);
    return Q.cols(1, n - 1);
}

arma::vec sample_eta_ess(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& eta, const arma::mat& Ur, double rho, double sigma, double tau_prior_sd) {
    int n_time = y.n_rows;
    arma::vec x = arma::linspace(0, 1, n_time);
    
    arma::vec taus_current = Ur * eta;
    
    auto kernel_pair = global_kernel_cache.get_kernel_matrices(x, rho);
    double current_ll = likelihood(y, f, betas, taus_current, rho, sigma, kernel_pair.first, kernel_pair.second);
    
    arma::vec nu = tau_prior_sd * arma::randn<arma::vec>(eta.n_elem);
    double log_u = std::log(thread_rng.runif()) + current_ll;
    double theta = thread_rng.runif(0, 2.0 * M_PI);
    double theta_min = theta - 2.0 * M_PI;
    double theta_max = theta;
    
    for(int s=0; s<100; ++s) {
        arma::vec eta_proposed = eta * std::cos(theta) + nu * std::sin(theta);
        double proposed_ll = likelihood(y, f, betas, Ur * eta_proposed, rho, sigma, kernel_pair.first, kernel_pair.second);
        
        if (proposed_ll > log_u) {
            global_diagnostics.ess_shrinkage_iterations += s;
            return eta_proposed;
        }
        
        if (theta < 0) theta_min = theta; else theta_max = theta;
        theta = thread_rng.runif(theta_min, theta_max);
    }

    global_diagnostics.ess_shrinkage_iterations += 100;
    return eta * std::cos(theta) + nu * std::sin(theta);
}

arma::vec sample_beta_lognormal_mh(const arma::mat& y, const arma::vec& f, const arma::vec& taus, double rho, double sigma, const arma::vec& betas_current, double beta_prior_mu_log, double beta_prior_sigma_log, double beta_proposal_sd) {
    int n = y.n_cols;
    arma::vec betas_new = betas_current;
    arma::vec x = arma::linspace(0, 1, y.n_rows);
    
    auto kernel_pair = global_kernel_cache.get_kernel_matrices(x, rho);
    arma::mat Sigma_nu = get_Sigma_nu(sigma, y.n_rows);

    for (int i = 0; i < n; i++) {
        double beta_current = betas_current(i);
        double log_beta_current = std::log(beta_current);
        double log_beta_proposed = log_beta_current + beta_proposal_sd * thread_rng.rnorm();
        double beta_proposed = std::exp(log_beta_proposed);

        double lik_current = likelihood_single_trial(i, y.col(i), f, beta_current, taus(i), rho, sigma, kernel_pair.second, Sigma_nu, x);
        double lik_proposed = likelihood_single_trial(i, y.col(i), f, beta_proposed, taus(i), rho, sigma, kernel_pair.second, Sigma_nu, x);
        
        double prior_current = prior_beta_lognormal(beta_current, beta_prior_mu_log, beta_prior_sigma_log);
        double prior_proposed = prior_beta_lognormal(beta_proposed, beta_prior_mu_log, beta_prior_sigma_log);
        
        double log_ratio = (lik_proposed - lik_current) + (prior_proposed - prior_current) + (log_beta_proposed - log_beta_current);
        
        global_diagnostics.beta_mh_proposals++;
        
        if (std::isfinite(log_ratio) && std::log(thread_rng.runif()) < log_ratio) {
            betas_new(i) = beta_proposed;
            global_diagnostics.beta_mh_accepts++;
        }
    }
    return betas_new;
}

double sample_rho_logspace_lognormal(const arma::mat& y, const arma::vec& f, const arma::vec& betas, const arma::vec& taus, double rho, double sigma, double mu_log, double sigma_log, double rho_proposal_sd) {
    double log_rho = std::log(rho);
    double log_rho_proposed = log_rho + rho_proposal_sd * thread_rng.rnorm();
    double rho_proposed = std::exp(log_rho_proposed);

    if (std::abs(rho_proposed - rho) < 1e-7) return rho;

    arma::vec x = arma::linspace(0,1,y.n_rows);
    auto k_curr = global_kernel_cache.get_kernel_matrices(x, rho);
    auto k_prop = global_kernel_cache.get_kernel_matrices(x, rho_proposed);
    
    double lik_current = likelihood(y, f, betas, taus, rho, sigma, k_curr.first, k_curr.second);
    double lik_proposed = likelihood(y, f, betas, taus, rho_proposed, sigma, k_prop.first, k_prop.second);
    
    double prior_current = prior_rho_lognormal(rho, mu_log, sigma_log);
    double prior_proposed = prior_rho_lognormal(rho_proposed, mu_log, sigma_log);
    
    double log_ratio = (lik_proposed - lik_current) + (prior_proposed - prior_current) + (log_rho_proposed - log_rho);
    
    return (std::isfinite(log_ratio) && thread_rng.runif() < std::exp(log_ratio)) ? rho_proposed : rho;
}

double sample_sigma(const arma::mat& z, double sigma_prior_shape, double sigma_prior_scale) {
    double ss_resid = arma::accu(z % z);
    double a_post = sigma_prior_shape + z.n_elem / 2.0;
    double b_post = 1.0 / sigma_prior_scale + ss_resid / 2.0;
    return std::sqrt(b_post / thread_rng.rchisq(2.0 * a_post) * 2.0);
}