// Simple episodic agent for testing.
// An episode consists of the first player sending a Ping
// and the second player responding with a Pong.
//
// Created by daniel on 18/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_PINGPONGAGENT_H
#define MULTIAGENTGOVERNMENT_PINGPONGAGENT_H

namespace abm::agents {
    class PingPongAgent {
    public:
        enum message_type {
            ping,
            pong,
            close = -1
        };

        message_type startDialogue() {
            return ping;
        }

        static constexpr char *body = "";

        message_type reactTo(message_type message) {
            return message == ping ? pong : close;
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_PINGPONGAGENT_H
