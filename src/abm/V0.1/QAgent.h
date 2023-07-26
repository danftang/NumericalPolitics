// An agent that uses a Q function to make decisions
//
// Created by daniel on 17/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QAGENT_H
#define MULTIAGENTGOVERNMENT_QAGENT_H

#include "../DQN.h"
#include <bitset>
#include "../GreedyPolicy.h"

namespace abm {
    template<class BODY, class QFUNCTION, class POLICY>  requires(std::is_convertible_v<BODY, typename QFUNCTION::input_type>)
    class QAgent {
    public:
        typedef BODY body_type;
        typedef BODY::message_type message_type;
        typedef BODY::intent_type intent_type;


        body_type       body;
        QFUNCTION       qFunction;
        POLICY          policy;
        QFUNCTION::input_type       lastState;
        int             lastIntent;
        message_type    lastMessage;

        // qPolicy is a function from a vector of qValues and a legal-move mask to an action index
        QAgent(BODY body, QFUNCTION qfunction, POLICY qPolicy) {

        }


        QAgent(QFUNCTION qfunction, POLICY qPolicy = GreedyPolicy(0.5, 0.99997, 0.01)):
                qFunction(std::move(qfunction)),
                policy(std::move(qPolicy)),
                lastMessage(message_type::close) { }

        QAgent(POLICY qPolicy = GreedyPolicy(0.5, 0.99997, 0.01)):
                policy(std::move(qPolicy)),
                lastMessage(message_type::close) { }



        // --- Agent interface

        // called when this agent is first mover
        message_type startDialogue() {
            lastMessage = message_type::close;
            return chooseMessage( body.transition(message_type::close, message_type::close));
        }

        // called when handling an incoming message
        message_type reactTo(message_type incomingMessage) {
            message_type response;
            double reward = body.transition(lastMessage, incomingMessage);
            if(incomingMessage != message_type::close) {
                response = chooseMessage(reward);
            } else {
                endEpisode(reward);
                response = message_type::close;
            }
            return response;
        }

    protected:
        // ---- mind interface

        // The agent's body state has just changed
        // and we've received the given reward
        // At the beginning of an episode, reward is ignored
        message_type chooseMessage(const double &reward) {
            std::bitset<intent_type::size> legalIntentMask = body.legalActs();
            message_type nextMessage;
            if(legalIntentMask != 0) {
                // make a decision
                typename QFUNCTION::input_type newState = body;
                int nextIntent = policy.sample(qFunction.predict(newState), body.legalActs());
//                std::cout << "Sampled intent " << nextIntent << " nLegal moves = " << body.legalActs().count() << " qValues = " << qValues << " state = " << newStateMatrix <<  std::endl;
                nextMessage = body_type::intentToMessage(nextIntent);
                if(lastMessage != message_type::close) qFunction.train(lastState, lastIntent, reward, newState, false);
                lastState = std::move(newState);
                lastIntent = nextIntent;
            } else {
                // we're in a terminal state so end the episode, no need to encode the new state
                nextMessage = message_type::close;
                if(lastMessage != message_type::close) qFunction.train(lastState, lastIntent, reward, lastState, true);
            }
            lastMessage = nextMessage;
//            std::cout << "message: " << nextMessage << std::endl;
            return nextMessage;
        }

        // at the end of an episode, we want to train on the final action and reset the lastSentMessage register
        // we're not interested in the final state as by definition it has zero value
        void endEpisode(const double &reward) {
            qFunction.train(lastState, lastIntent, reward, lastState, true);
            lastMessage = message_type::close;
        }


    };
}

#endif //MULTIAGENTGOVERNMENT_QAGENT_H
