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
    namespace random {
        inline std::mutex seedMutex;
        inline std::mt19937 seedGenerator(
                static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count()) +
                static_cast<std::mt19937::result_type>(reinterpret_cast<uintptr_t>(&seedMutex))); // attempt at random initialisation
        inline thread_local std::mt19937 gen;


        /** A thread-safe randomGenerator.
         * Use this to uniquely seed your thread-local instance of gen using gen.seed(nextRandomSeed())
         * @return a globally random seed for a random generator.
         */
        inline std::mt19937::result_type nextRandomSeed() {
            seedMutex.lock();
            std::mt19937::result_type seed = seedGenerator();
            seedMutex.unlock();
            return seed;
        }

        inline bool Bernoulli(double pTrue = 0.5) {
            return std::bernoulli_distribution(pTrue)(gen);
        }

        /**
         * @param begin lowest value in the range
         * @param end upper bound to the range (inclusive/exclusive depending on endIsInclusive)
         * @tparam endIsInclusive indicates whether end indicates the last value in the range (true) or one past the last value (default)
         * @return if endIdInclusive is true then a value chosen uniformly from [begin,end] otherwise from [begin,end)
         */
        template<std::integral I, bool endIsInclusive = false> requires (!std::same_as<I,bool>)
        inline I uniform(I begin, I end) {
            if constexpr (endIsInclusive) {
                assert(begin <= end);
                return std::uniform_int_distribution<I>(begin, end)(gen);
            } else {
                assert(begin < end);
                return std::uniform_int_distribution<I>(begin, end - 1)(gen);
            }
        }

        /**
         * @param begin lower bound (inclusive)
         * @param end upper bound (exclusive)
         * @return uniformly sampled value in [begin,end)
         */
        template<std::floating_point F>
        inline F uniform(F begin, F end) {
            return std::uniform_real_distribution<F>(begin, end)(gen);
        }

        /**
         * @param end upper bound to the range (inclusive/exclusive depending on endIsInclusive)
         * @tparam endIsInclusive indicates whether end indicates the last value in the range (true) or one past the last value (default)
         * @return if endIdInclusive is true then a value chosen uniformly from [0,end] otherwise from [0,end)
         */
        template<std::integral I, bool endIsInclusive = false> requires (!std::same_as<I,bool>)
        inline I uniform(I end) {
            return uniform<I,endIsInclusive>(0,end);
        }

        /**
         * @param begin lower bound (inclusive)
         * @param end upper bound (exclusive)
         * @return uniformly sampled value in [begin,end)
         */
        template<std::floating_point F>
        inline F uniform(F end) {
            return uniform<F>(0,0, end);
        }

        /** uniform<bool>() is synonymous with Bernoulli(0.5) */
        template<class I> requires std::same_as<I,bool>
        inline bool uniform() {
            return gen() & 1;
        }


        template<std::input_iterator InputIterator> requires std::convertible_to<std::iter_value_t<InputIterator>,double>
        inline auto discrete(InputIterator begin, InputIterator end) {
            return std::discrete_distribution<std::iter_difference_t<InputIterator>>(begin, end)(gen);
        }

        inline auto discrete(const std::ranges::range auto &probabilities) {
            return discrete(std::begin(probabilities), std::end(probabilities));
        }

        inline int Poisson(double lambda) {
            return std::poisson_distribution(lambda)(gen);
        }

        inline int binomial(int nTirals, double p) {
            return std::binomial_distribution<int>(nTirals, p)(gen);
        }

        /** Choses a random element uniformly from a range with known size.
         * @return an iterator to an element of the container, chosen with a uniform probability,
         * or end() if the container is empty.
         */
        inline auto uniformIterator(std::ranges::sized_range auto &container) {
            auto it = std::ranges::begin(container);
            std::ranges::advance(it, std::uniform_int_distribution<size_t>(0, std::ranges::size(container)-1)(gen));
            return it;
        }


        /** Chooses a random element uniformly from a non-random-access range with unknown size.
         * Requires only a single iteration through the range, so this is good for lazily filtered/evaluated views.
         *
         * Works by keeping a buffer of the last few iterators, and using a recurrence relation:
         * Suppose we have a uniformly chosen integer, r, in [1:n] such that
         *  P(r = i) = 1/n for any 1 <= i <= n
         * Suppose now we draw an integer, s, uniformly from [1:n+m].
         * If 1 <= s <= n then we leave r unchanged, otherwise we set r=s.
         * The probability is now
         * P(r = i) = (1/n)(n/(n+m)) = 1/(n+m) for 1 <= i <= n
         * and
         * P(r = i) = 1/(n+m) for n+1 <= i <= n+m
         * so r is now a uniformly chosen integer in [1:n+m].
         *
         * @return an iterator in the range, chosen with uniform probability, or end() if the range is empty.
         */
        template<std::ranges::range RANGE> requires (!std::ranges::random_access_range<RANGE> && !std::ranges::sized_range<RANGE>)
        inline auto uniformIterator(RANGE &range) {
            constexpr size_t BatchSize = 4;
            std::vector<std::ranges::iterator_t<RANGE>> iteratorBuffer;
            iteratorBuffer.reserve(BatchSize);
            size_t n = 1;
            auto chosenIt = std::ranges::begin(range);
            auto it = std::ranges::begin(range);
            while(it != std::ranges::end(range)) {
                while(iteratorBuffer.size() < BatchSize && ++it != std::ranges::end(range)) {
                    iteratorBuffer.push_back(it);
                }
                size_t randIndex = std::uniform_int_distribution<size_t>(0, n + iteratorBuffer.size() -1)(gen);
                if(randIndex >= n) {
                    chosenIt = iteratorBuffer[randIndex - n];
                }
                n += iteratorBuffer.size();
                iteratorBuffer.clear();
            }
            return chosenIt;
        }

        /** Choses a random element uniformly from a random access range with unknown size (just for completeness,
         * though I don't know of any such class).
         * @return an iterator to an element of the container, chosen with a uniform probability,
         * or end() if the container is empty.
         */
        template<std::ranges::random_access_range RANGE> requires (!std::ranges::sized_range<RANGE>)
        inline auto uniformIterator(RANGE &container) {
            auto it = std::ranges::begin(container);
            size_t size = 0;
            while(it != std::ranges::end(container)) {
                ++size;
                ++it;
            }
            it = std::ranges::begin(container);
            std::ranges::advance(it, std::uniform_int_distribution<size_t>(0, size-1)(gen));
            return it;
        }

        /**  */
        class ObjectDistribution {

        };

    };

}
#endif //GLPKTEST_RANDOM_H
