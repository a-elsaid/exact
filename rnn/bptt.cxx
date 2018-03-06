#include <cmath>

#include <fstream>
using std::ofstream;

#include <iostream>
using std::cout;
using std::endl;

#include <random>
using std::minstd_rand0;
using std::uniform_real_distribution;

#include <vector>
using std::vector;


#include "rnn_genome.hxx"
#include "bptt.hxx"

void get_analytic_gradient(vector<RNN_Genome*> &genomes, const vector<double> &parameters, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, double &mse, vector<double> &analytic_gradient) {

    double mse_sum = 0.0;
    double mse_current;
    for (uint32_t i = 0; i < genomes.size(); i++) {
        genomes[i]->set_weights(parameters);
        genomes[i]->forward_pass(series_data[i]);
        mse_current = genomes[i]->calculate_error_mse(expected_outputs[i]);
        //cout << "mse[" << i << "]: " << mse_current << endl;

        mse_sum += mse_current;
    }

    for (uint32_t i = 0; i < genomes.size(); i++) {
        double d_mse = mse_sum * (1.0 / expected_outputs[i][0].size()) * 2.0;

        genomes[i]->backward_pass(d_mse);
    }

    mse = mse_sum;

    vector<double> current_gradients;
    analytic_gradient.assign(parameters.size(), 0.0);
    for (uint32_t k = 0; k < genomes.size(); k++) {

        uint32_t current = 0;
        for (uint32_t i = 0; i < genomes[k]->get_number_nodes(); i++) {
            genomes[k]->get_node(i)->get_gradients(current_gradients);

            for (uint32_t j = 0; j < current_gradients.size(); j++) {
                analytic_gradient[current] += current_gradients[j];
                current++;
            }
        }

        for (uint32_t i = 0; i < genomes[k]->get_number_edges(); i++) {
            analytic_gradient[current] += genomes[k]->get_edge(i)->get_gradient();
            current++;
        }
    }
}


void backpropagate(RNN_Genome *genome, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, bool use_low_norm, string log_filename) {

    int32_t n_series = series_data.size();
    vector<RNN_Genome*> genomes;
    for (int32_t i = 0; i < n_series; i++) {
        genomes.push_back( genome->copy() );
    }

    vector<double> parameters;
    genome->get_weights(parameters);

    int n_parameters = genome->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double high_threshold = 2;
    double low_threshold = 0.001;
    double original_learning_rate = learning_rate;

    double prev_mu;
    double prev_norm;
    double prev_learning_rate;
    double prev_mse;
    double mse;

    double norm = 0.0;

    //initialize the initial previous values
    get_analytic_gradient(genomes, parameters, series_data, expected_outputs, mse, analytic_gradient);

    norm = 0.0;
    for (int32_t i = 0; i < parameters.size(); i++) {
        norm += analytic_gradient[i] * analytic_gradient[i];
    }
    norm = sqrt(norm);
    
    ofstream output_log(log_filename);

    bool was_reset = false;
    int reset_count = 0;
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        prev_mu = mu;
        prev_norm  = norm;
        prev_mse = mse;
        prev_learning_rate = learning_rate;

        prev_gradient = analytic_gradient;

        get_analytic_gradient(genomes, parameters, series_data, expected_outputs, mse, analytic_gradient);

        norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
        }
        norm = sqrt(norm);

        output_log << iteration
             << " " << mse 
             << " " << norm
             << " " << learning_rate << endl;

        cout << "iteration " << iteration
             << ", mse: " << mse 
             << ", lr: " << learning_rate 
             << ", norm: " << norm;

        if (reset_weights && prev_mse * 2 < mse) {
            cout << ", RESETTING WEIGHTS" << endl;
            parameters = prev_parameters;
            //prev_velocity = prev_prev_velocity;
            prev_velocity.assign(parameters.size(), 0.0);
            mse = prev_mse;
            mu = prev_mu;
            learning_rate = prev_learning_rate;
            analytic_gradient = prev_gradient;

            learning_rate *= 0.5;
            if (learning_rate < 0.0000001) learning_rate = 0.0000001;

            reset_count++;
            if (reset_count > 20) break;

            was_reset = true;
            continue;
        }

        if (was_reset) {
            was_reset = false;
        } else {
            reset_count = 0;
            learning_rate = original_learning_rate;
        }


        if (adapt_learning_rate) {
            if (prev_mse > mse) {
                learning_rate *= 1.10;
                if (learning_rate > 1.0) learning_rate = 1.0;

                cout << ", INCREASING LR";
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;
            cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse * 1.05 < mse) {
                    cout << ", WORSE";
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        cout << endl;

        if (nesterov_momentum) {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_prev_velocity[i] = prev_velocity[i];

                double mu_v = prev_velocity[i] * prev_mu;

                prev_velocity[i] = mu_v  - (prev_learning_rate * prev_gradient[i]);
                parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);
            }
        } else {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_gradient[i] = analytic_gradient[i];
                parameters[i] -= learning_rate * analytic_gradient[i];
            }
        }
    }

    genome->set_weights(parameters);
}


void backpropagate_stochastic(RNN_Genome *genome, const vector< vector< vector<double> > > &series_data, const vector< vector< vector<double> > > &expected_outputs, int max_iterations, double learning_rate, bool nesterov_momentum, bool adapt_learning_rate, bool reset_weights, bool use_high_norm, bool use_low_norm, string log_filename) {

    vector<double> parameters;
    genome->get_weights(parameters);

    int n_parameters = genome->get_number_weights();
    vector<double> prev_parameters(n_parameters, 0.0);

    vector<double> prev_velocity(n_parameters, 0.0);
    vector<double> prev_prev_velocity(n_parameters, 0.0);

    vector<double> analytic_gradient;
    vector<double> prev_gradient(n_parameters, 0.0);

    double mu = 0.9;
    double high_threshold = 2;
    double low_threshold = 0.001;
    double original_learning_rate = learning_rate;

    int n_series = series_data.size();
    double prev_mu[n_series];
    double prev_norm[n_series];
    double prev_learning_rate[n_series];
    double prev_mse[n_series];
    double mse;

    double norm = 0.0;

    //initialize the initial previous values
    for (uint32_t i = 0; i < n_series; i++) {
        genome->get_analytic_gradient(parameters, series_data[i], expected_outputs[i], mse, analytic_gradient);

        norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
        }
        norm = sqrt(norm);
        prev_mu[i] = mu;
        prev_norm[i] = norm;
        prev_mse[i] = mse;
        prev_learning_rate[i] = learning_rate;
    }

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    minstd_rand0 generator(seed);
    uniform_real_distribution<double> rng(0, 1);

    int random_selection = rng(generator);
    mu = prev_mu[random_selection];
    norm = prev_norm[random_selection];
    mse = prev_mse[random_selection];
    learning_rate = prev_learning_rate[random_selection];

    ofstream output_log(log_filename);

    bool was_reset = false;
    int reset_count = 0;
    for (uint32_t iteration = 0; iteration < max_iterations; iteration++) {
        prev_mu[random_selection] = mu;
        prev_norm[random_selection] = norm;
        prev_mse[random_selection] = mse;
        prev_learning_rate[random_selection] = learning_rate;

        prev_gradient = analytic_gradient;

        if (!was_reset) {
            random_selection = rng(generator) * series_data.size();
        }

        genome->get_analytic_gradient(parameters, series_data[random_selection], expected_outputs[random_selection], mse, analytic_gradient);

        norm = 0.0;
        for (int32_t i = 0; i < parameters.size(); i++) {
            norm += analytic_gradient[i] * analytic_gradient[i];
        }
        norm = sqrt(norm);

        output_log << iteration
             << " " << mse 
             << " " << norm
             << " " << learning_rate << endl;

        cout << "iteration " << iteration
             << ", series: " << random_selection
             << ", mse: " << mse 
             << ", lr: " << learning_rate 
             << ", norm: " << norm;

        if (reset_weights && prev_mse[random_selection] * 2 < mse) {
            cout << ", RESETTING WEIGHTS" << endl;
            parameters = prev_parameters;
            //prev_velocity = prev_prev_velocity;
            prev_velocity.assign(parameters.size(), 0.0);
            mse = prev_mse[random_selection];
            mu = prev_mu[random_selection];
            learning_rate = prev_learning_rate[random_selection];
            analytic_gradient = prev_gradient;

            random_selection = rng(generator) * series_data.size();

            learning_rate *= 0.5;
            if (learning_rate < 0.0000001) learning_rate = 0.0000001;

            reset_count++;
            if (reset_count > 20) break;

            was_reset = true;
            continue;
        }

        if (was_reset) {
            was_reset = false;
        } else {
            reset_count = 0;
            learning_rate = original_learning_rate;
        }


        if (adapt_learning_rate) {
            if (prev_mse[random_selection] > mse) {
                learning_rate *= 1.10;
                if (learning_rate > 1.0) learning_rate = 1.0;

                cout << ", INCREASING LR";
            }
        }

        if (use_high_norm && norm > high_threshold) {
            double high_threshold_norm = high_threshold / norm;
            cout << ", OVER THRESHOLD, multiplier: " << high_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = high_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                learning_rate *= 0.5;
                if (learning_rate < 0.0000001) learning_rate = 0.0000001;
            }

        } else if (use_low_norm && norm < low_threshold) {
            double low_threshold_norm = low_threshold / norm;
            cout << ", UNDER THRESHOLD, multiplier: " << low_threshold_norm;

            for (int32_t i = 0; i < parameters.size(); i++) {
                analytic_gradient[i] = low_threshold_norm * analytic_gradient[i];
            }

            if (adapt_learning_rate) {
                if (prev_mse[random_selection] * 1.05 < mse) {
                    cout << ", WORSE";
                    learning_rate *= 0.5;
                    if (learning_rate < 0.0000001) learning_rate = 0.0000001;
                }
            }
        }

        cout << endl;

        if (nesterov_momentum) {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_prev_velocity[i] = prev_velocity[i];

                double mu_v = prev_velocity[i] * prev_mu[random_selection];

                prev_velocity[i] = mu_v  - (prev_learning_rate[random_selection] * prev_gradient[i]);
                parameters[i] += mu_v + ((mu + 1) * prev_velocity[i]);
            }
        } else {
            for (int32_t i = 0; i < parameters.size(); i++) {
                prev_parameters[i] = parameters[i];
                prev_gradient[i] = analytic_gradient[i];
                parameters[i] -= learning_rate * analytic_gradient[i];
            }
        }
    }
}