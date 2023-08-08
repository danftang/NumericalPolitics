//
// Created by daniel on 28/07/23.
//

#include "../abm/minds/IncompleteInformationMCTS.h"
#include "../abm/bodies/SugarSpiceTradingBody.h"
#include "../DeselbyStd/random.h"

namespace tests {
    void incompleteInformationMCTSTest() {
        const int nRootSamples = 4;

        typedef abm::bodies::SugarSpiceTradingBody<false> body_type;
        abm::minds::IncompleteInformationMCTS<body_type> myMCTS(0, 0, std::function<std::function<const BODY &()>(
                const BODY &)>(), std::function<std::function<const BODY &()>(const BODY &)>());

        myMCTS.buildTree(std::function<const BODY &()>(), std::function<const BODY &()>());
    }
}