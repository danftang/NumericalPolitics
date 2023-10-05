//
// Created by daniel on 05/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_LAMBDAMIND_H
#define MULTIAGENTGOVERNMENT_LAMBDAMIND_H

#include <utility>

namespace abm::minds {
    template<class LAMBDA>
    class LambdaMind {
    public:
        LAMBDA function;

        LambdaMind(LAMBDA function): function(std::move(function)) {}

        template<class BODY>
        auto act(BODY &&body) {
            return function(std::forward<BODY>(body));
        }
    };
}


#endif //MULTIAGENTGOVERNMENT_LAMBDAMIND_H
