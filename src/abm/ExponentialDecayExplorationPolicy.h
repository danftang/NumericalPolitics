//
// Created by daniel on 16/05/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPONENTIALDECAYEXPLORATIONPOLICY_H
#define MULTIAGENTGOVERNMENT_EXPONENTIALDECAYEXPLORATIONPOLICY_H


class ExponentialDecayExplorationPolicy {
public:
    double explorationProb;
    const double decayRate;
    const double minimumExplorationProb;

    ExponentialDecayExplorationPolicy(double initialExplorationProb, double decayRate, double minimumExplorationProb):
        explorationProb(initialExplorationProb), decayRate(decayRate), minimumExplorationProb(minimumExplorationProb) {

    }

    double operator ()() {
        if(explorationProb>minimumExplorationProb) explorationProb *= decayRate;
        return explorationProb;
    }
};


#endif //MULTIAGENTGOVERNMENT_EXPONENTIALDECAYEXPLORATIONPOLICY_H
