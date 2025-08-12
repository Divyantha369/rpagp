#include "utils.h"

// Constants 
constexpr double machine_epsilon = std::numeric_limits<double>::epsilon();
constexpr double min_eigenvalue_ratio = 1e-10;
constexpr double likelihood_floor = -1e8;

// Global Object Definitions 
KernelCache global_kernel_cache;
DiagnosticInfo global_diagnostics;

// Thread-Local RNG Definition 
// This is the one and only place where the thread_rng object is actually created.
thread_local Rng_t thread_rng;

// Method Definitions for Rng_t struct 
void Rng_t::seed(uint64_t seed_value) { engine.seed(seed_value); }
double Rng_t::rnorm(double mean, double sd) { return normal_dist(engine) * sd + mean; }
double Rng_t::runif(double min, double max) { return uniform_dist(engine) * (max - min) + min; }
double Rng_t::rchisq(double df) {
    std::gamma_distribution<double> gamma(df/2.0, 2.0);
    return gamma(engine);
}

// Method Definitions for KernelCache 
KernelCache::KernelCache() { omp_init_lock(&cache_lock); }
KernelCache::~KernelCache() { omp_destroy_lock(&cache_lock); }

std::pair<arma::mat, arma::mat> KernelCache::get_kernel_matrices(const arma::vec& x, double rho) const {
    omp_set_lock(&cache_lock);
    if (std::abs(rho - cached_rho) < 1e-12) {
        cache_hits++;
        auto result = std::make_pair(cached_K_f, cached_K_f_inv);
        omp_unset_lock(&cache_lock);
        return result;
    }
    
    cache_misses++;
    omp_unset_lock(&cache_lock);
    
    arma::mat K_f = sq_exp_kernel_adaptive(x, rho, 1.0);
    arma::mat K_f_inv = robust_inv(K_f, true);
    
    omp_set_lock(&cache_lock);
    cached_rho = rho;
    cached_K_f = K_f;
    cached_K_f_inv = K_f_inv;
    omp_unset_lock(&cache_lock);
    
    return std::make_pair(K_f, K_f_inv);
}

void KernelCache::reset() {
    omp_set_lock(&cache_lock);
    cached_rho = -1.0;
    cached_K_f.reset();
    cached_K_f_inv.reset();
    cache_hits = 0;
    cache_misses = 0;
    omp_unset_lock(&cache_lock);
}

void KernelCache::print_stats() const {
    int hits = cache_hits.load();
    int misses = cache_misses.load();
    int total = hits + misses;
    
    if (total > 0) {
        double hit_rate = static_cast<double>(hits) / total * 100.0;
        Rcpp::Rcout << "Kernel cache statistics:\n";
        Rcpp::Rcout << "  Cache hits: " << hits << "\n";
        Rcpp::Rcout << "  Cache misses: " << misses << "\n";
        Rcpp::Rcout << "  Hit rate: " << std::fixed << std::setprecision(1) << hit_rate << "%\n";
    }
}

//  Method Definitions for DiagnosticInfo 
DiagnosticInfo::DiagnosticInfo() { omp_init_lock(&diag_lock); }
DiagnosticInfo::~DiagnosticInfo() { omp_destroy_lock(&diag_lock); }

void DiagnosticInfo::reset() {
    regularization_count = 0;
    numerical_errors = 0;
    likelihood_floor_hits = 0;
    max_norm_applications = 0;
    ess_shrinkage_iterations = 0;
    beta_mh_proposals = 0;
    beta_mh_accepts = 0;
    omp_set_lock(&diag_lock);
    trial_errors.clear();
    regularization_levels.clear();
    omp_unset_lock(&diag_lock);
}

void DiagnosticInfo::add_trial_error(int trial, const std::string& error) {
    omp_set_lock(&diag_lock);
    trial_errors.push_back({trial, error});
    omp_unset_lock(&diag_lock);
    numerical_errors++;
}

void DiagnosticInfo::add_regularization(double level) {
    omp_set_lock(&diag_lock);
    regularization_levels.push_back(level);
    omp_unset_lock(&diag_lock);
    regularization_count++;
}

void DiagnosticInfo::report_summary() {
    Rcpp::Rcout << "\n=== Numerical Diagnostics Summary ===\n";
    Rcpp::Rcout << "Regularizations applied: " << regularization_count.load() << "\n";
    if (!regularization_levels.empty()) {
        double mean_reg = 0.0, max_reg = 0.0;
        for (double r : regularization_levels) {
            mean_reg += r;
            max_reg = std::max(max_reg, r);
        }
        mean_reg /= regularization_levels.size();
        Rcpp::Rcout << "  Mean regularization: " << std::scientific << mean_reg << "\n";
        Rcpp::Rcout << "  Max regularization: " << std::scientific << max_reg << "\n";
    }
    Rcpp::Rcout << "Numerical errors: " << numerical_errors.load() << "\n";
    Rcpp::Rcout << "Likelihood floor hits: " << likelihood_floor_hits.load() << "\n";
    Rcpp::Rcout << "Beta Metropolis-Hastings:\n";
    Rcpp::Rcout << "  Total proposals: " << beta_mh_proposals.load() << "\n";
    Rcpp::Rcout << "  Total accepts: " << beta_mh_accepts.load() << "\n";
    if (beta_mh_proposals > 0) {
        double acc_rate = static_cast<double>(beta_mh_accepts.load()) / beta_mh_proposals.load() * 100.0;
        Rcpp::Rcout << "  Overall acceptance rate: " << std::fixed << std::setprecision(1) << acc_rate << "%\n";
    }
    if (!trial_errors.empty()) {
        Rcpp::Rcout << "\nTrial-specific errors (first 10 of " << trial_errors.size() << "):\n";
        for (size_t i = 0; i < std::min(trial_errors.size(), (size_t)10); ++i) {
            Rcpp::Rcout << "  Trial " << trial_errors[i].first << ": " << trial_errors[i].second << "\n";
        }
    }
    Rcpp::Rcout << "===================================\n\n";
}

// Utility Function Definitions 
void configure_parallel_environment(uint64_t master_seed) {
    int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    omp_set_nested(0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        uint64_t thread_seed = master_seed ? (master_seed + tid) : std::random_device{}();
        thread_rng.seed(thread_seed);
    }
    #ifdef HAVE_OPENBLAS
        openblas_set_num_threads(1);
    #endif
    #ifdef HAVE_MKL
        mkl_set_num_threads(1);
    #endif
    Rcpp::Rcout << "Using " << max_threads << " threads.\n";
}

RegularizationResult adaptive_regularize(const arma::mat& M, double scale_factor, bool track_diagnostic) {
    RegularizationResult result;
    result.regularized_matrix = M;
    result.regularization_level = 0.0;
    result.was_regularized = false;
    
    arma::vec eigval;
    try {
        eigval = arma::eig_sym(M);
    } catch (...) {
        double trace_M = arma::trace(M);
        double n = M.n_rows;
        result.regularization_level = std::max(machine_epsilon * std::abs(trace_M) / n, 1e-8);
        result.regularized_matrix.diag() += result.regularization_level;
        result.was_regularized = true;
        if (track_diagnostic) global_diagnostics.add_regularization(result.regularization_level);
        return result;
    }
    
    double max_eig = arma::max(eigval);
    double min_eig = arma::min(eigval);
    
    if (min_eig < min_eigenvalue_ratio * max_eig || min_eig < 0) {
        double target_min_eig = std::max(min_eigenvalue_ratio * max_eig, machine_epsilon * scale_factor);
        result.regularization_level = target_min_eig - min_eig;
        result.regularized_matrix.diag() += result.regularization_level;
        result.was_regularized = true;
        if (track_diagnostic) global_diagnostics.add_regularization(result.regularization_level);
    }
    return result;
}

arma::mat robust_inv(const arma::mat& A, bool track_regularization) {
    arma::mat invA;
    if (arma::inv_sympd(invA, A)) return invA;

    RegularizationResult reg_result = adaptive_regularize(A, arma::norm(A, "fro") / A.n_rows, track_regularization);
    if (arma::inv_sympd(invA, reg_result.regularized_matrix)) return invA;
    
    return arma::pinv(A);
}

double dmvnorm_chol(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p) {
    int n = x.n_elem;
    arma::mat L;
    bool chol_success = false;
    arma::mat sym_sigma = (sigma + sigma.t()) / 2.0;

    try {
        chol_success = arma::chol(L, sym_sigma);
        if (!chol_success) {
            RegularizationResult reg = adaptive_regularize(sym_sigma, 1.0, false);
            chol_success = arma::chol(L, reg.regularized_matrix);
        }
    } catch (...) {
        chol_success = false;
    }

    if (!chol_success) {
        global_diagnostics.add_trial_error(-1, "Cholesky failed; falling back to inv_sympd.");
        return dmvnorm(x, mean, sigma, log_p); 
    }

    arma::vec z = x - mean;
    arma::vec v = arma::solve(arma::trimatl(L), z, arma::solve_opts::fast);
    double quad_form = arma::dot(v, v);
    double log_det_sigma = 2.0 * arma::sum(arma::log(L.diag()));

    if (!std::isfinite(log_det_sigma) || !std::isfinite(quad_form)) {
        global_diagnostics.likelihood_floor_hits++;
        return log_p ? likelihood_floor : 0.0;
    }

    double log_pdf = -0.5 * (n * std::log(2.0 * M_PI) + log_det_sigma + quad_form);
    
    return log_p ? std::max(log_pdf, likelihood_floor) : std::exp(log_pdf);
}

double dmvnorm(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p) {
    int n = x.n_elem;
    double log_det_sigma;
    double sign;
    
    RegularizationResult reg_result = adaptive_regularize(sigma, 1.0, false);
    arma::mat sigma_reg = reg_result.regularized_matrix;
    
    if (!arma::log_det(log_det_sigma, sign, sigma_reg) || sign <= 0) {
        global_diagnostics.likelihood_floor_hits++;
        return log_p ? likelihood_floor : 0.0;
    }
    
    arma::vec x_centered = x - mean;
    arma::mat sigma_inv = robust_inv(sigma_reg, false);
    double quadform = arma::as_scalar(x_centered.t() * sigma_inv * x_centered);

    double log_pdf = -0.5 * (n * std::log(2.0 * M_PI) + log_det_sigma + quadform);
    
    return log_p ? std::max(log_pdf, likelihood_floor) : std::exp(log_pdf);
}