// Simple episodic agent for testing.
// An episode consists of the first player sending a Ping
// and the second player responding with a Pong.
//
// Created by daniel on 18/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_PINGPONGAGENT_H
#define MULTIAGENTGOVERNMENT_PINGPONGAGENT_H

#include <optional>
#include "../DeselbyStd/stlstream.h"

namespace abm {
    class PingPongAgent {
    public:
        enum message_type {
            ping,
            pong
        };

        message_type startEpisode() {
            return ping;
        }


        std::optional<message_type> handleMessage(message_type message) {
            std::optional<message_type> response;
            if(message == ping) response = pong;
            return response;
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_PINGPONGAGENT_H
