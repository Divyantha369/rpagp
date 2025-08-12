#ifndef RPAGP_UTILS_H
#define RPAGP_UTILS_H

#include <RcppArmadillo.h>
#include "rpagp_types.h"
#include <random> 

// Struct definition for the RNG 
struct Rng_t {
    std::mt19937_64 engine;
    std::normal_distribution<double> normal_dist{0.0, 1.0};
    std::uniform_real_distribution<double> uniform_dist{0.0, 1.0};
  
    void seed(uint64_t seed_value);
    double rnorm(double mean = 0.0, double sd = 1.0);
    double runif(double min = 0.0, double max = 1.0);
    double rchisq(double df);
};

// Extern declaration for the thread-local RNG object 
extern thread_local Rng_t thread_rng;

// Function Declarations 
void configure_parallel_environment(uint64_t master_seed);
RegularizationResult adaptive_regularize(const arma::mat& M, double scale_factor, bool track_diagnostic);
arma::mat robust_inv(const arma::mat& A, bool track_regularization);
double dmvnorm_chol(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p);
double dmvnorm(const arma::vec& x, const arma::vec& mean, const arma::mat& sigma, bool log_p);

#endif // RPAGP_UTILS_H