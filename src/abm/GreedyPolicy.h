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
    template<class ACTION>
    class GreedyPolicy {
    public:

        typedef ACTION action_type;

        std::function<bool()> explorationStrategy;

        //  d^steps = (minimum/initialExp)^{1/steps}
        explicit GreedyPolicy(std::function<bool()> explorationStrategy) : explorationStrategy(std::move(explorationStrategy)) {}


        template<class QVECTOR, DiscreteActionMask ACTIONMASK>
        ACTION sample(const QVECTOR &qValues, const ACTIONMASK &legalMoves) {
            size_t chosenMove = action_type::size;
            assert(legalMoves.count() >= 1);
            if (explorationStrategy()) {
                // choose a legal move at random
                chosenMove = sampleUniformly(legalMoves);
            } else {
                // choose a legal move with highest Q
                std::vector<size_t> indices = legalIndices(legalMoves);
                std::shuffle(indices.begin(), indices.end(), deselby::Random::gen); // ...in-case of multiple max values
                double bestQ = -std::numeric_limits<double>::infinity();
                for (size_t i : indices) {
                    double q = qValues[i];
                    assert(!isnan(q));
                    if (q > bestQ) {
                        bestQ = q;
                        chosenMove = i;
                    }
                }
            }
            assert(chosenMove < action_type::size);
            return static_cast<ACTION>(chosenMove);
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
