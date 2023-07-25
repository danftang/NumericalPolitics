//
// Created by daniel on 24/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_TRANSITION_H
#define MULTIAGENTGOVERNMENT_TRANSITION_H

namespace abm {
    template<class STATE>
    struct Transition {
        STATE startState;
        int action;
        double reward;
        STATE endState;
        bool isTerminal;
    };
}

#endif //MULTIAGENTGOVERNMENT_TRANSITION_H
