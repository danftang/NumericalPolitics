// Wrapper to provide .action and ::size for use in
// mlpack environments.
//
// Created by daniel on 09/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_MLPACKACTION_H
#define MULTIAGENTGOVERNMENT_MLPACKACTION_H


namespace abm {
    template<class ACTION, int SIZE = ACTION::size>
    class MlPackAction {
    public:
        static constexpr int size = SIZE;
        ACTION action;

        MlPackAction(ACTION val) : action(val) {}
        MlPackAction() : action() {}

        operator ACTION() const { return action; }

        bool operator ==(const ACTION &otherAction) const { return action == otherAction; }

        MlPackAction &operator =(const ACTION &otherAction) {
            action = otherAction;
            return *this;
        }

    };
};

#endif //MULTIAGENTGOVERNMENT_MLPACKACTION_H
