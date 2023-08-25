//
// Created by daniel on 08/08/23.
//

#ifndef MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
#define MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H

#include "../Body.h"
#include <cassert>
#include "../ActionMask.h"

namespace abm::minds {
    template<Body BODY>
    class ZeroIntelligence {
    public:
        typedef BODY                observation_type;
        typedef BODY::action_mask   action_mask;
        typedef double              reward_type;

        BODY::action_type act([[maybe_unused]] const observation_type &observation, const action_mask &legalMoves,
                              [[maybe_unused]] reward_type reward = 0.0) const {
            assert(legalMoves.size() == BODY::action_type::size);
            return static_cast<BODY::action_type>(sampleUniformly(legalMoves));
        }

        void endEpisode([[maybe_unused]] reward_type reward) { }

    };

}


#endif //MULTIAGENTGOVERNMENT_ZEROINTELLIGENCE_H
