//
// Created by daniel on 13/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
#define MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H

#include "../../DeselbyStd/typeutils.h"
#include "../Concepts.h"
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
     *   - A training policy that defines when parameter updates should occur and performs the actual updates
     */
    template<LossFunction LOSSFUNCTION, ParameterisedFunction APPROXIMATOR, class TRAININGPOLICY>
    class AdaptiveFunction : public APPROXIMATOR {
    public:
        LOSSFUNCTION    lossFunction;
        TRAININGPOLICY  trainingPolicy;


        AdaptiveFunction(APPROXIMATOR approximator, TRAININGPOLICY trainingPolicy, LOSSFUNCTION lossFunction):
                APPROXIMATOR(std::move(approximator)),
                lossFunction(std::move(lossFunction)),
                trainingPolicy(std::move(trainingPolicy))
        { }


        template<class EVENT>
        void on(const EVENT &event) {
            callback(event, lossFunction);
            bool didTrain = trainingPolicy.train(event, *this, lossFunction);
            if(didTrain) callback(events::ParameterUpdate(this->parameters()), lossFunction);
        }
    };


    template<class LOSSFUNCTION>
    struct UpdateOnLossFunctionEvent {
        template<IsEventHandledBy<LOSSFUNCTION> EVENT>
        bool operator()(const EVENT & /*event*/) { return true; }
    };


    /** Makes a trainable function, given
     *   - an approximator function whose outputs are differentiable w.r.t. the parameters.
     *   - a loss function whose loss is differentiable w.r.t. the function outputs.
     * and optionally:
     *   - An optimising parameter update step given the current gradient of loss w.r.t. params (default Adam Update).
     *   - A schedule defining exactly when to perform a parameter update step (default every time the loss function
     *     intercepts an event).
     */
    template<LossFunction LOSSFUNCTION, DifferentiableParameterisedFunction<LOSSFUNCTION> APPROXIMATOR,
            class UPDATESTEP = ens::AdamUpdate,
            class SCHEDULE = UpdateOnLossFunctionEvent<LOSSFUNCTION>>
    class DifferentiableAdaptiveFunction : public APPROXIMATOR {
    public:
        using APPROXIMATOR::parameters;
        using APPROXIMATOR::gradientByParams;

        LOSSFUNCTION    lossFunction;
        UPDATESTEP      update;
        double          stepSize;
        SCHEDULE        schedule;

        typedef std::remove_cvref_t<decltype(std::declval<APPROXIMATOR>().parameters())>                     param_type;
        typedef std::remove_cvref_t<decltype(std::declval<APPROXIMATOR>().gradientByParams(lossFunction))>   grad_type;
        typedef UPDATESTEP::template Policy<param_type, grad_type>              update_policy_type;

        update_policy_type updatePolicy; // requires gradient to be same type as param type


        DifferentiableAdaptiveFunction(
                APPROXIMATOR approximator,
                LOSSFUNCTION lossFunction,
                UPDATESTEP update = UPDATESTEP(),
                double stepSize = 0.001,
                SCHEDULE schedule = SCHEDULE())
                :
                APPROXIMATOR(std::move(approximator)),
                lossFunction(std::move(lossFunction)),
                update(std::move(update)),
                stepSize(stepSize),
                schedule(std::move(schedule)),
                updatePolicy(this->update, this->parameters().n_rows, this->parameters().n_cols)
        { }


        template<class EVENT>
        void on(const EVENT &event) {
            std::cout << "Intercepting event in DifferentialAdaptiveFunction" << std::endl;
            callback(event, lossFunction);
            if (deselby::invoke_or(schedule, false, event)) { // Update parameters
                updatePolicy.Update(parameters(), stepSize, gradientByParams(lossFunction));
                callback(events::ParameterUpdate(parameters()), lossFunction);
            }
        }

        template<class EVENT>
        static consteval bool doTrain(const EVENT & /*event */) {
            return abm::HasCallback<LOSSFUNCTION,EVENT>;
        }
    };
}

#endif //MULTIAGENTGOVERNMENT_ADAPTIVEFUNCTION_H
