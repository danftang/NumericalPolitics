//
// Created by daniel on 21/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLE_H
#define MULTIAGENTGOVERNMENT_QTABLE_H

#include <array>
#include "QVector.h"
#include "../../Agent.h"
#include "../../../DeselbyStd/stlstream.h"

namespace abm::minds {
    template<int NSTATES, int NACTIONS, class QVALUE = ExponentiallyWeightedQValue<0.999>>
    class QTable {
    public:
        static constexpr int output_dimension = NACTIONS;

        std::array<QVector<NACTIONS, QVALUE>, NSTATES>  table;

        const double discount;     // exponential decay factor of future reward
        double reward;
        size_t lastBodyState;
        std::optional<size_t> lastAction;


        QTable(double discount) : discount(discount), reward(0.0) { }


        auto QVector(size_t body) const { return table[body]; }


        template<class BODY>
        void on(const events::AgentStartEpisode<BODY> & event) {
            // train on last step
            lastBodyState = event.body;
        }

        /** Remember last act, body state and reward */
        template<class BODY, class ACTION, class MESSAGE>
        void on(const events::OutgoingMessage<BODY, ACTION, MESSAGE> &outMessage) {
            lastAction = outMessage.act;
            reward = outMessage.reward;
        }


        /** Learn from last call/response step */
        template<class BODY, class MESSAGE>
        void on(const events::IncomingMessage<MESSAGE, BODY> &inMessage) {
            if(lastAction.has_value()) {
                reward += inMessage.reward;
                const double endStateQValue = *std::ranges::max_element(table[inMessage.body]); // assumes non-legal Q-values are never max
                const double forwardQ = reward + discount * endStateQValue;
//                std::cout << "Training on " << lastBodyState << " " << *lastAction << " " << reward << " " << forwardQ << std::endl;
                table[lastBodyState][*lastAction].addSample(forwardQ);
            }
            lastBodyState = inMessage.body;
        }


        /** learn from residual reward of end-game */
        template<class BODY>
        void on(const events::AgentEndEpisode<BODY> & /* event */) {
            // train on last step
            table[lastBodyState][*lastAction].addSample(reward);
            lastAction.reset();
        }

        friend std::ostream &operator <<(std::ostream &out, const QTable<NSTATES,NACTIONS,QVALUE> &qTable) {
            for(uint i=0; i<NSTATES; ++i) {
                out << i << " -> " << qTable.QVector(i).asArray() << '\n';
            }
            return out;
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QTABLE_H
