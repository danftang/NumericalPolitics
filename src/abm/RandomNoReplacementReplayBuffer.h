//
// Created by daniel on 23/07/23.
//

#ifndef TORCHTEST_RANDOMREPLAYBUFFER_H
#define TORCHTEST_RANDOMREPLAYBUFFER_H

#include <cstdlib>
#include <vector>
#include <random>

namespace abm {
    template<class DATA>
    class RandomNoReplacementReplayBuffer {
    public:
        inline static std::default_random_engine gen;

        size_t batchSize;
        std::vector<DATA> data;
        std::vector<size_t> randomPermutation;

        RandomNoReplacementReplayBuffer(size_t maxBufferSize, size_t batchSize) : batchSize(batchSize) {
            data.reserve(maxBufferSize);
        }

        void store(const DATA &item) {
            data.push_back(item);
            randomPermutation.push_back(randomPermutation.size());
        }

        void store(DATA &&item) {
            data.push_back(std::move(item));
            randomPermutation.push_back(randomPermutation.size());
        }


        /** @return returns a vector of batchSize samples taken randomly from the buffer,
         * without replacement (i.e. ensuring that no two samples point to the same item).
         * However, if the number of items in the buffer is less than batchSize
         * then after all items have been sampled, they are all replaced for further
         * sampling.
         **/
        std::vector<const DATA *> sample() {
            std::vector<const DATA *> samples;
            samples.reserve(batchSize);
            size_t i = 0;
            while(samples.size() < batchSize) {
                    size_t randomIndex = std::uniform_int_distribution<size_t>(i, size() - 1)(gen);
                    size_t sampledIndex = randomPermutation[randomIndex];
                    samples.push_back(&data[sampledIndex]);
                    randomPermutation[randomIndex] = randomPermutation[i]; // remove randomIndex'th entry from sample pool
                    randomPermutation[i] = sampledIndex;
                    i = (i+1)%randomPermutation.size();
            }
            return samples;
        }


        size_t size() { return data.size(); }
    };
}

#endif //TORCHTEST_RANDOMREPLAYBUFFER_H
