//
// Created by daniel on 21/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_QTABLE_H
#define MULTIAGENTGOVERNMENT_QTABLE_H

#include <array>
#include "QVector.h"
#include "../../Agent.h"
#include "../../../DeselbyStd/stlstream.h"
#include "QLearningStepMixin.h"

namespace abm::minds {
    template<int NSTATES, int NACTIONS, class QVALUE = ExponentiallyWeightedQValue<700>>
    class QTable : public QLearningStepMixin<size_t> {
    public:
        std::array<QVector<NACTIONS, QVALUE>, NSTATES>  table;
        const double discount;     // exponential decay factor of future reward

        using QLearningStepMixin<size_t>::on; // prevent our handler from hiding mixin


        QTable(double discount) : discount(discount) { }


        auto operator()(size_t body) const { return table[body]; }


        void on(const events::QLearningStep<size_t> &event) {
            if(event.isEndOfEpisode()) {
                table[*event.startStatePtr][event.action].addSample(event.reward);
            } else {
                const double endStateQValue = *std::ranges::max_element(table[*event.endStatePtr]); // assumes non-legal Q-values are never max
                const double forwardQ = event.reward + discount * endStateQValue;
                table[*event.startStatePtr][event.action].addSample(forwardQ);
            }
        }


        friend std::ostream &operator <<(std::ostream &out, const QTable<NSTATES,NACTIONS,QVALUE> &qTable) {
            for(uint i=0; i<NSTATES; ++i) {
                out << i << " -> " << qTable(i).asArray() << '\n';
            }
            return out;
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_QTABLE_H
