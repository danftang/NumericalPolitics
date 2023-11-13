//
// Created by daniel on 18/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
#define MULTIAGENTGOVERNMENT_GREEDYPOLICY_H

#include <bitset>
#include <concepts>
#include <functional>

#include "../../../DeselbyStd/random.h"
#include "../../Concepts.h"
#include "../ZeroIntelligence.h"

namespace abm::minds {
    class GreedyPolicy {
    public:
        std::function<bool()> explorationStrategy;

        /** exploration strategies can be found in abm::explorationStrategies */
        explicit GreedyPolicy(std::function<bool()> explorationStrategy) : explorationStrategy(std::move(explorationStrategy)) {}

        /** QVector must have operator[] and elements must have an ordering  */
        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        size_t sample(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            size_t chosenMove;
            assert(legalActs.count() >= 1);
            return explorationStrategy() ? minds::ZeroIntelligence::sampleUniformly(legalActs) : max(qValues, legalActs);
        }

        /** more efficient (on average) to use rejection sampling */
        template<GenericQVector QVECTOR, IntegralActionMask ACTIONMASK>
        static size_t max(const QVECTOR &qValues, const ACTIONMASK &legalActs) {
            // choose a legal move with highest Q
            auto indices = legalIndices(legalActs);
            std::shuffle(indices.begin(), indices.end(), deselby::random::gen); // ...in-case of multiple max values
            auto chosenMove = indices[0];
            for(size_t ii = 1; ii < indices.size(); ++ii) {
                auto act = indices[ii];
                if(qValues[chosenMove] < qValues[act]) chosenMove = act;
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

        LinearDecay(double initialExploration, size_t stepsToDecayToMinimum, double minimumExploration) :
                pExplore(initialExploration),
                exploreDecay((initialExploration - minimumExploration)/stepsToDecayToMinimum),
                pExploreMin(minimumExploration) {}

        bool operator()() {
            if (pExplore > pExploreMin) pExplore -= std::min(exploreDecay, pExplore - pExploreMin);
            return deselby::random::Bernoulli(pExplore);
        }
    };

    /** remain at maximum exploration for a number of steps, then decay linearly */
    class BurninThenLinearDecay {
    public:
        size_t burnin;
        double pExplore;
        double exploreDecay;
        double pExploreMin;

        BurninThenLinearDecay(size_t burnin, double initialExploration, size_t stepsToDecayToMinimum, double minimumExploration) :
                burnin(burnin),
                pExplore(initialExploration),
                exploreDecay((initialExploration - minimumExploration)/stepsToDecayToMinimum),
                pExploreMin(minimumExploration) {}

        bool operator()() {
            if(burnin > 0) {
                --burnin;
            } else if (pExplore > pExploreMin) {
                pExplore -= std::min(exploreDecay, pExplore - pExploreMin);
            }
            return deselby::random::Bernoulli(pExplore);
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
            return deselby::random::Bernoulli(pExplore);
        }
    };

    struct NoExploration { bool operator()() { return false; } };

    class FixedExploration {
    public:
        double pExplore;
        FixedExploration(double pExplore): pExplore(pExplore) {}
        bool operator()() { return deselby::random::Bernoulli(pExplore); }
    };
}

#endif //MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
