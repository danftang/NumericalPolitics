//
// Created by daniel on 13/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
#define MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H

#include "../../DeselbyStd/typeutils.h"
#include "Concepts.h"
#include "../CallbackUtils.h"

namespace abm::events {
    template<class PARAM>
    struct ParameterUpdate {
        const PARAM &parameters;

        ParameterUpdate(const PARAM &params): parameters(params) {}
    };
}

namespace abm::approximators {

    /** A function that intercepts events and adapts to those events.
     * An instance of this class is the joining together of three objects:
     *   - A parameterised object that can update its own parameters given a LossFunction
     *   - A LossFunction describing a loss in the space of all functions
     *   - A training policy that defines when parameter updates should occur.
     */
    template<LossFunction LOSSFUNCTION, OptimisableFunction<LOSSFUNCTION> APPROXIMATOR, class TRAININGPOLICY>
//    template<LossFunction LOSSFUNCTION, class APPROXIMATOR, class TRAININGPOLICY>
    class AdaptiveFunction : public APPROXIMATOR {
    public:
        LOSSFUNCTION    lossFunction;
        TRAININGPOLICY  trainingPolicy;


        AdaptiveFunction(APPROXIMATOR approximator, LOSSFUNCTION lossFunction, TRAININGPOLICY trainingPolicy):
                APPROXIMATOR(approximator),
                lossFunction(std::move(lossFunction)),
                trainingPolicy(std::move(trainingPolicy))
        { }


        template<class EVENT> requires HasCallback<LOSSFUNCTION,EVENT> || std::is_invocable_v<TRAININGPOLICY,EVENT>
        void on(const EVENT &event) {
            callback(event, lossFunction);
            deselby::constexpr_if<std::is_invocable_v<TRAININGPOLICY,EVENT>>(
                    [&event, this](auto &policy) {
                        if(policy(event)) {
                            this->updateParameters(lossFunction);
                            callback(events::ParameterUpdate(this->parameters()), lossFunction);
                        }
                    }, trainingPolicy);
        }
    };


    /** A parameterised function and an update step, giving rise to an OptimisableFunction that can update
     * its parameters given a LossFunction */
    template<class UPDATESTEP, class APPROXIMATOR>
    class DifferentiableOptimisableFunction: public APPROXIMATOR {
    public:
        typedef std::remove_cvref_t<decltype(std::declval<APPROXIMATOR>().parameters())> MatType;

        UPDATESTEP                  updateStep;
        typename UPDATESTEP::template Policy<MatType,MatType> updatePolicy; // requires gradient to be same type as param type
        double                      stepSize;

        DifferentiableOptimisableFunction(APPROXIMATOR approximator, UPDATESTEP updatestep, double stepSize):
        APPROXIMATOR(std::move(approximator)),
        updateStep(std::move(updatestep)),
        updatePolicy(updateStep, this->parameters().n_rows, this->parameters().n_cols),
        stepSize(stepSize) {
        }

        template<LossFunction LOSS> requires DifferentiableParameterisedFunction<APPROXIMATOR,LOSS>
        void updateParameters(LOSS &loss) {
            updatePolicy.Update(this->parameters(), stepSize, this->gradientByParams(loss));
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
