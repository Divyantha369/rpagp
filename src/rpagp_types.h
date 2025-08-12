#ifndef RPAGP_TYPES_H
#define RPAGP_TYPES_H

#include <RcppArmadillo.h>
#include <omp.h>
#include <atomic>
#include <string>
#include <vector>
#include <iomanip>

//  Forward Declarations 
arma::mat sq_exp_kernel_adaptive(const arma::vec& x, double rho, double alpha);
arma::mat robust_inv(const arma::mat& A, bool track_regularization);
double dmvnorm(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p);
struct RegularizationResult adaptive_regularize(const arma::mat& M, double scale_factor, bool track_diagnostic);

//  Kernel Cache Class 
class KernelCache {
private:
    mutable double cached_rho = -1.0;
    mutable arma::mat cached_K_f;
    mutable arma::mat cached_K_f_inv;
    mutable omp_lock_t cache_lock;
    mutable std::atomic<int> cache_hits{0};
    mutable std::atomic<int> cache_misses{0};

public:
    KernelCache();
    ~KernelCache();
    std::pair<arma::mat, arma::mat> get_kernel_matrices(const arma::vec& x, double rho) const;
    void reset();
    void print_stats() const;
};

//  Diagnostic Info Struct 
struct DiagnosticInfo {
    std::atomic<int> regularization_count{0};
    std::atomic<int> numerical_errors{0};
    std::atomic<int> likelihood_floor_hits{0};
    std::atomic<int> max_norm_applications{0};
    std::atomic<int> ess_shrinkage_iterations{0};
    std::atomic<int> beta_mh_proposals{0};
    std::atomic<int> beta_mh_accepts{0};
    std::vector<std::pair<int, std::string>> trial_errors;
    std::vector<double> regularization_levels;
    omp_lock_t diag_lock;

    DiagnosticInfo();
    ~DiagnosticInfo();
    void reset();
    void add_trial_error(int trial, const std::string& error);
    void add_regularization(double level);
    void report_summary();
};

//  Regularization Result Struct 
struct RegularizationResult {
    arma::mat regularized_matrix;
    double regularization_level;
    bool was_regularized;
};

//  Global Object Declarations 
extern KernelCache global_kernel_cache;
extern DiagnosticInfo global_diagnostics;

#endif // RPAGP_TYPES_H