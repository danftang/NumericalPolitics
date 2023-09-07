//
// Created by daniel on 07/09/23.
//

#ifndef MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
#define MULTIAGENTGOVERNMENT_GREEDYPOLICY_H

#include <functional>
#include "../../Body.h"
#include "../../../DeselbyStd/random.h"

namespace abm::qLearning {

    /** A QFunction is a class that is callable on a BODY and returns a vector which
     * implements operator[] on size_t and returns something convertible to double
     */
    template<class T, class BODY>
    concept QFunction = requires(T f, BODY body, size_t i) {
        { f(body)[i] } -> std::convertible_to<double>;
    };

    /**
     *
     * @tparam BODY         Body for which this policy is creating actions
     * @tparam QFUNCTION    A function from Body states to Q-values which implement operator[], returning
     *                      something convertible to double
     */
    template<Body BODY, QFunction<BODY> QFUNCTION>
    class GreedyPolicy: public QFUNCTION {
    public:
        typedef BODY::action_type action_type;

        std::function<bool()> explorationStrategy;

        explicit GreedyPolicy(std::function<bool()> explorationStrategy, QFUNCTION &&qFunc) :
        QFUNCTION(std::move(qFunc)),
        explorationStrategy(std::move(explorationStrategy)) {}

        action_type operator()(const BODY &body) {
            return sample(QFUNCTION(body), body.legalActs());
        }

    protected:
        action_type sample(const auto &qValues, const auto &legalMoves) {
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
            return static_cast<action_type>(chosenMove);
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_GREEDYPOLICY_H
