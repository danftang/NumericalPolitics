//
// Created by daniel on 28/07/23.
//

#include "../abm/minds/IncompleteInformationMCTS.h"
#include "../abm/bodies/SugarSpiceTradingBody.h"

namespace tests {
    void incompleteInformationMCTSTest() {
//        const int nRootSamples = 4;
//
        typedef abm::bodies::SugarSpiceTradingBody<true> body_type;
//        abm::minds::IncompleteInformationMCTS<body_type> myMCTS(0, 0, std::function<std::function<const BODY &()>(
//                const BODY &)>(), std::function<std::function<const BODY &()>(const BODY &)>());

        auto offTreeApproximator = []<class BODY>(const BODY & /* body */) -> arma::mat {
            return arma::zeros(BODY::action_type::size);
        };

        std::function<body_type(const body_type &)> bodyStateSampler = [](const body_type &myTrueState) {
            return body_type(!myTrueState.hasSugar(), !myTrueState.hasSpice(), deselby::random::uniform<bool>());
        };


        double discount = 1.0;
        uint nSamplesInATree = 1000000;

        auto myMCTS = abm::minds::IncompleteInformationMCTS(
                offTreeApproximator,
                bodyStateSampler,
                bodyStateSampler,
                discount,
                nSamplesInATree
        );

//        auto selfStateSampler = []() { return body_type(false, true, deselby::random::uniform<bool>()); };
//        auto otherStateSampler = []() { return body_type(true, false, deselby::random::uniform<bool>()); };
//        myMCTS.doSelfPlay<true>(selfStateSampler, otherStateSampler, 1);

        bool isFirstMover = true;
        bool hasSugar = false;
        bool hasSpice = true;
        bool prefersSugar = true;
        body_type startState(hasSugar, hasSpice,prefersSugar);
        myMCTS.on(abm::events::AgentStartEpisode<body_type>(startState, isFirstMover));

        std::cout << myMCTS(body_type(hasSugar, hasSpice, true)) << std::endl;
        std::cout << myMCTS(body_type(hasSugar, hasSpice, false)) << std::endl;
        std::cout << "Done" << std::endl;

    }
}