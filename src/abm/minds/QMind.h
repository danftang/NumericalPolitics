//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QMIND_H
#define MULTIAGENTGOVERNMENT_QMIND_H

#include <bitset>
#include <optional>

namespace abm::events {
    template<class STATE>
    struct QLearningStep {
        STATE startState;
        size_t action;
        double reward;
        STATE endState;
    };
}

namespace abm::minds {

    /** A QMind is just a QFunction with a policy to create an act(body) function
     *
     * @tparam QFUNCTION    class that approximates a Q-function from body states to vectors of Q-values over acts
     * @tparam POLICY       class that converts a vector of Q-values and a legal-act mask to an act to perform.
     */
    template<class QFUNCTION, class POLICY>
    class QMind: public QFUNCTION { // A QMind is a QFunction with an act member so we inherit all event handlers
    public:
        POLICY      policy;

        QMind(QFUNCTION qfunction, POLICY policy): QFUNCTION(std::move(qfunction)), policy(std::move(policy)) { }

        template<class BODY>
        auto act(const BODY &body) {
            return policy.sample(this->QVector(body), body.legalActs());
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QMIND_H
