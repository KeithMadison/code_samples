// Python implementation of the SMT algorithm and a stochastic approximation,
// as well as some tools for computing SNRs, mean VRMS values, etc.
// Keith Madison (aff. The University of Kansas)

// Boost prerequisites
#include <boost/dynamic_bitset.hpp>
#include <boost/math/distributions/binomial.hpp>
#include <boost/math/distributions/normal.hpp>

// GSL prerequisites
#include <gsl/gsl_multifit.h>
#include <gsl/gsl_statistics_double.h>

// C++ prerequisites
#include <functional>
#include <iostream>
#include <vector>
#include <random>
#include <queue>
#include <algorithm>

// ROOT prerequisites
#include <TGraph.h>
#include <TProfile.h>
#include <TCanvas.h>
#include <TH1.h>
#include <TFile.h>
#include <TTree.h>

#include "AraEventCalibrator.h"

#define N 100

/**
 * @brief Computes the trigger rate numerically.
 *
 * This method computes the trigger rate by simulating thermal noise 
 * and implementing the "literal" SMT algorithm.
 *
 * @param threshold The absolute voltage threshold for triggering.
 * @param temperature The temperature in Kelvin.
 * @return The computed trigger rate.
 */
std::uint64_t Antenna::getTriggerRateNumerical(double threshold, double temperature) {
    // Compute VRMS (root mean square voltage)
    vrms = 2 * sqrt(resistance * getThermalNoisePower(temperature));

    std::normal_distribution<double> dist(0, vrms);
    std::priority_queue<double, std::vector<double>, std::greater<double>> pq;

    std::vector<boost::dynamic_bitset<>> window(channels, boost::dynamic_bitset<>(windowSize));

    auto takeSample = [&](int index, boost::dynamic_bitset<> &channel) {
        // Take a sample and check if it exceeds the threshold
        double sample = fabs(dist(gen));
        channel.set(index, sample > threshold);

        // Maintain a priority queue of the highest samples
        if (pq.size() < channelThreshold) {
            pq.push(sample);
        } else if (sample > pq.top()) {
            pq.pop();
            pq.push(sample);
        }
    };

    auto clear = [&]() {
        // Clear the sample window by taking new samples
        for (auto &channel : window)
            for (int i = 0; i < windowSize; ++i)
                takeSample(i, channel);
    };

    clear();

    highestChanSNR = 0;
    std::uint64_t numContributingChannels = 0, numTriggers = 0;

    for (int i = 0, imod = 0; i < samplingRate; ++i, imod = (imod + 1) % windowSize) {
        numContributingChannels = 0;

        for (auto &channel : window)
            if (channel.any())
                ++numContributingChannels;

        // Check if the number of contributing channels exceeds the threshold
        if (numContributingChannels > channelThreshold) {
            highestChanSNR += pq.top() / (2 * vrms);

            ++numTriggers;
            i += writeDelay + imod;

            clear();
            continue;
        }

        for (auto &channel : window)
            takeSample(imod, channel);
    }

    highestChanSNR /= numTriggers;

    return numTriggers;
}

/**
 * @brief Computes the trigger rate probabilistically.
 *
 * This method uses my probabilistic approximation to estimate the trigger rate
 * without the need for simulating each sample.
 *
 * @param threshold The absolute voltage threshold for triggering.
 * @param temperature The temperature in Kelvin.
 * @return The estimated trigger rate.
 */
std::uint64_t Antenna::getTriggerRateProbabilistic(double threshold, double temperature) {
    // Compute VRMS (root mean square voltage)
    vrms = 2 * sqrt(resistance * getThermalNoisePower(temperature));

    std::vector<std::vector<double>> stochMatrix(windowSize + 1, std::vector<double>(windowSize + 1));
    std::vector<double> statDistribution(channels + 1);

    boost::math::normal_distribution<> normalDist(0, vrms);

    double excProbability = 1 - pow(boost::math::cdf(normalDist, threshold), windowSize);

    boost::math::binomial_distribution<> binomDist(channels, excProbability);

    // Fill the statistical distribution matrix
    for (int i = 0; i < channels + 1; ++i)
        for (int j = 0; j < channels + 1; ++j)
            statDistribution[i] += boost::math::pdf(binomDist, j) * 
                                   boost::math::pdf(normalDist, i);

    double normFactor = std::accumulate(statDistribution.begin(), statDistribution.end(), 0.0);
    double trigProb = std::accumulate(statDistribution.begin() + (channelThreshold + 1), statDistribution.end(), 0.0);

    // Approximately a logistic sigmoid
    return 1 / (1 - (trigProb / normFactor));
}

/**
 * @brief Computes the thermal noise power.
 *
 * This method calculates the thermal noise power using the standard Nyquist formula.
 *
 * @param temp The temperature in Kelvin.
 * @return The thermal noise power.
 */
double Antenna::getThermalNoisePower(double temp) {
    // Compute the thermal noise power using the given formula
    return 1.381E-23 * temp * bandwidth * (pow(10, noiseFig / 10) * pow(10, gain / 10) + 1);
}
