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
        abm::minds::IncompleteInformationMCTS<body_type> myMCTS(
                []() {
                    return body_type(deselby::Random::nextBool(), deselby::Random::nextBool(),
                                     deselby::Random::nextBool());
                }, 0, 0);

        myMCTS.buildTree();
    }
}