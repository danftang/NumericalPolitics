//
// Created by daniel on 07/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_QL_GREEDYPOLICY_H
#define MULTIAGENTGOVERNMENT_QL_GREEDYPOLICY_H

#include <functional>
#include "../../Body.h"
#include "../../../DeselbyStd/random.h"

namespace abm::minds::qLearning {

    /** A QFunction is a class that is callable on a BODY and returns a vector which
     * implements operator[] on size_t and returns something convertible to double
     */
    template<class BODY, class QFUNCTION>
    concept QCompatible = requires(QFUNCTION qFunc, BODY body, size_t i) {
        { qFunc(body)[i] } -> std::convertible_to<double>;
        { qFunc(body).size() } -> std::convertible_to<size_t>;
    };

    size_t sampleMaxQ(const auto &qValues, const auto &legalMoves) {
        size_t chosenMove = qValues.size();
        assert(legalMoves.count() >= 1);
        // choose a legal move with highest Q
        std::vector<size_t> indices = legalIndices(legalMoves);
        double bestQ = -std::numeric_limits<double>::infinity();
        size_t nMaxEntries = 1; // for randomly choosing between multiple maxQ
        for (size_t i : indices) {
            double q = qValues[i];
            assert(!isnan(q));
            if (q > bestQ) {
                bestQ = q;
                chosenMove = i;
                nMaxEntries = 1;
            } else if(q == bestQ) { // break tied max by drawing from uniform probability
                if(deselby::Random::nextBool(1.0/++nMaxEntries)) {
                    chosenMove = i;
                }
            }
        }
        assert(chosenMove < qValues.size());
        return chosenMove;
    }

    /** This class wraps a QFunction to provide a function from a Body to an act
     *
     * @tparam BODY         Body for which this policy is creating actions
     * @tparam QFUNCTION    A function from Body states to Q-values which implement operator[], returning
     *                      something convertible to double
     */
    template<class QFUNCTION>
    class GreedyPolicy: public QFUNCTION {
    public:
        std::function<bool()> explorationStrategy;

        explicit GreedyPolicy(std::function<bool()> explorationStrategy, QFUNCTION &&qFunc) :
        QFUNCTION(std::move(qFunc)),
        explorationStrategy(std::move(explorationStrategy)) {}

        template<class BODY> requires QCompatible<BODY,QFUNCTION>
        BODY::action_type operator()(const BODY &body) {
            return explorationStrategy()?ZeroIntelligence(body):static_cast<BODY::action_type>(sample(QFUNCTION(body), body.legalActs()));
        }
    };

    namespace explorationStrategies {


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

}


#endif //MULTIAGENTGOVERNMENT_QL_GREEDYPOLICY_H
