//
// Created by daniel on 11/05/2021.
//

#ifndef GLPKTEST_RANDOM_H
#define GLPKTEST_RANDOM_H

#include <random>
#include <iostream>
#include <cassert>
#include <chrono>
#include <mutex>
#include <ranges>

namespace deselby {
    class Random {
    protected:
        static std::mutex seedMutex;
        static std::mt19937 seedGenerator;
    public:
        thread_local static std::mt19937 gen;


        // thread-safe seed randomGenerator.
        // This can be used to uniquely seed different thread local instances of gen.
        static std::mt19937::result_type nextRandomSeed() {
            seedMutex.lock();
            std::mt19937::result_type seed = seedGenerator();
            seedMutex.unlock();
            return seed;
        }



        static double nextDouble(double start = 0.0, double end = 1.0) {
            return std::uniform_real_distribution<double>(start, end)(gen);
        }

        static int nextInt(int until) {
            return nextInt(0, until);
        }

        static int nextInt(int from, int until) {
            assert(from < until);
            return std::uniform_int_distribution<int>(from, until - 1)(gen);
        }

        static int nextLong(long from, long until) {
            assert(from < until);
            return std::uniform_int_distribution<long>(from, until - 1)(gen);
        }

        static bool nextBool() {
            return std::uniform_int_distribution<int>(0, 1)(gen) == 0;
        }

        static bool nextBool(double pTrue) {
            return std::bernoulli_distribution(pTrue)(gen);
        }

        static int nextIntFromDiscrete(const std::vector<double> &probabilities) {
            return nextIntFromDiscrete(probabilities.begin(), probabilities.end());
        }

        template<typename InputIterator>
        static int nextIntFromDiscrete(InputIterator begin, InputIterator end) {
            return std::discrete_distribution<int>(begin, end)(gen);
        }

        static int nextPoisson(double lambda) {
            return std::poisson_distribution(lambda)(gen);
        }

        static int nextBinomial(int nTirals, double p) {
            return std::binomial_distribution<int>(nTirals, p)(gen);
        }

        /** Choses a random element uniformly from a range with known size.
         * @return an iterator to an element of the container, chosen with a uniform probability,
         * or end() if the container is empty.
         */
        template<std::ranges::sized_range RANDOMACESSCONTAINER>
        static auto chooseElement(RANDOMACESSCONTAINER &container) {
            auto it = std::ranges::begin(container);
            std::ranges::advance(it, std::uniform_int_distribution<size_t>(0, std::ranges::size(container))(gen));
            return it;
        }


        /** Chooses a random element uniformly from a range with unknown size.
         *
         * Works by using a recurrence relation:
         * Suppose we have a uniformly chosen integer, r, in [1:n] such that
         *  P(r = i) = 1/n for any 1 <= i <= n
         * Suppose now with probability 1/(n+1) we set r = n+1, so P(r=n+1) = 1/(n+1)
         * and P(r = i) = (1/n)*(n/(n+1)) = 1/(n+1)
         * so r is now a uniformly chosen integer in [1:n+1].
         *
         * @return an iterator in the range, chosen with uniform probability, or end() if the range is empty.
         */
        template<std::ranges::range RANGE>
        static auto chooseElement(RANGE &range) {
            size_t size = 0;
            auto chosenIt = std::ranges::begin(range);
            for(auto it = std::ranges::begin(range); it != range.end(); ++it) {
                size_t rand = std::uniform_int_distribution<size_t>(size_t(0), size)(gen);
                if(rand == 0) chosenIt = it;
                ++size;
            }
            return chosenIt;
        }

    };

    inline std::mutex Random::seedMutex;
    inline std::mt19937 Random::seedGenerator(
            static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()) +
            static_cast<std::mt19937::result_type>(reinterpret_cast<uintptr_t>(&seedMutex))); // attempt at random initialisation
    inline thread_local std::mt19937 Random::gen;
}
#endif //GLPKTEST_RANDOM_H
