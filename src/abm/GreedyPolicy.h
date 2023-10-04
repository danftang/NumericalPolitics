//
// Created by daniel on 18/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
#define MULTIAGENTGOVERNMENT_GREEDYPOLICY_H

#include <bitset>
#include <armadillo>
#include "../DeselbyStd/random.h"
#include <concepts>
#include "ActionMask.h"

namespace abm {

    template<class T>
    concept GenericQVector = requires(T obj, size_t i) {
        { obj[i] < obj[i] } -> std::convertible_to<bool>;
    };

    class GreedyPolicy {
    public:
        std::function<bool()> explorationStrategy;

        /** exploration strategies can be found in abm::explorationStrategies */
        explicit GreedyPolicy(std::function<bool()> explorationStrategy) : explorationStrategy(std::move(explorationStrategy)) {}

        /** QVector must have operator[] and elements must have an ordering  */
        template<GenericQVector QVECTOR, DiscreteActionMask ACTIONMASK>
        size_t sample(const QVECTOR &qValues, const ACTIONMASK &legalMoves) {
            size_t chosenMove;
            assert(legalMoves.count() >= 1);
            if (explorationStrategy()) {
                // choose a legal move at random
                chosenMove = sampleUniformly(legalMoves);
            } else {
                // choose a legal move with highest Q
                auto indices = legalIndices(legalMoves);
                std::shuffle(indices.begin(), indices.end(), deselby::Random::gen); // ...in-case of multiple max values
                chosenMove = indices[0];
                for(size_t ii = 1; ii < indices.size(); ++ii) {
                    auto act = indices[ii];
                    if(qValues[chosenMove] < qValues[act]) chosenMove = act;
                }
            }
            return chosenMove;
        }

    };
}

/** Some exploration strategies for use with GreedyPolicy
 *
 */
namespace abm::explorationStrategies {


    class LinearDecay {
    public:
        double pExplore;
        double exploreDecay;
        double pExploreMin;

        LinearDecay(double initialExploration, int stepsToDecayToMinimum, double minimumExploration) :
                pExplore(initialExploration),
                exploreDecay((initialExploration - initialExploration)/stepsToDecayToMinimum),
                pExploreMin(minimumExploration) {}

        bool operator()() {
            if (pExplore > pExploreMin) pExplore -= exploreDecay;
            return deselby::Random::nextBool(pExplore);
        }
    };


    class ExponentialDecay {
    public:
        double pExplore;
        double exploreDecay;
        double pExploreMin;

        ExponentialDecay(double initialExploration, int stepsToDecayToMinimum, double minimumExploration) :
                pExplore(initialExploration),
                exploreDecay(pow(minimumExploration/initialExploration, 1.0/stepsToDecayToMinimum)),
                pExploreMin(minimumExploration) {}

        bool operator()() {
            if (pExplore > pExploreMin) pExplore *= exploreDecay;
            return deselby::Random::nextBool(pExplore);
        }
    };

    struct NoExploration { bool operator()() { return false; } };

    class FixedExploration {
    public:
        double pExplore;
        FixedExploration(double pExplore): pExplore(pExplore) {}
        bool operator()() { return deselby::Random::nextBool(pExplore); }
    };
}

#endif //MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
