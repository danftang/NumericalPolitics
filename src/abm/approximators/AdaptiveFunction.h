//
// Created by daniel on 13/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
#define MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H

#include "../../DeselbyStd/typeutils.h"
#include "Concepts.h"
#include "../CallbackUtils.h"

namespace abm::approximators {

    template<StochasticLossFunction STOCHASTICLOSSFUNCTION, OptimisableFunction<STOCHASTICLOSSFUNCTION> APPROXIMATOR, class TRAININGPOLICY>
    class AdaptiveFunction : public APPROXIMATOR {
    public:
        STOCHASTICLOSSFUNCTION  stochasticLossFunction;
        TRAININGPOLICY          trainingPolicy;


        AdaptiveFunction(APPROXIMATOR approximator, STOCHASTICLOSSFUNCTION stochasticLossFunction, TRAININGPOLICY trainingPolicy):
            APPROXIMATOR(approximator),
            stochasticLossFunction(std::move(stochasticLossFunction)),
            trainingPolicy(std::move(trainingPolicy))
        {
        }

        template<class EVENT> requires HasCallback<STOCHASTICLOSSFUNCTION,EVENT> || std::is_invocable_v<TRAININGPOLICY,EVENT>
        void on(const EVENT &event) {
            abm::callback(event, stochasticLossFunction);
            deselby::constexpr_if<std::is_invocable_v<TRAININGPOLICY,EVENT>>(
                    [&event, this](auto &policy) {
                        if(policy(event)) this->updateParameters(stochasticLossFunction.getNextLossFunction());

                    }, trainingPolicy);
        }

        // TODO: Should the function be able to
    };


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

        template<LossFunction LOSS> requires ParameterisedFunction<APPROXIMATOR,LOSS>
        void updateParameters(LOSS &loss) {
            updatePolicy.Update(this->parameters(), stepSize, this->parameterGradient(loss));
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
