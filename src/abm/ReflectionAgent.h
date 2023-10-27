//
// Created by daniel on 26/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_REFLECTIONAGENT_H
#define MULTIAGENTGOVERNMENT_REFLECTIONAGENT_H

#include <utility>
#include <optional>
namespace abm {
    /** An agent that just reflects all messages */
    class ReflectionAgent {
    public:
        // A reflector cannot start an episode
         const std::nullopt_t &startEpisode() {
            return std::nullopt;
        }

        template<class MESSAGE>
        auto handleMessage(MESSAGE &&message) {
            return std::forward<MESSAGE>(message);
        }


    };
}


#endif //MULTIAGENTGOVERNMENT_REFLECTIONAGENT_H
