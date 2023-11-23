//
// Created by daniel on 31/10/23.
//

#ifndef MULTIAGENTGOVERNMENT_ABM_CONCEPTS_H
#define MULTIAGENTGOVERNMENT_ABM_CONCEPTS_H

#include <armadillo>
#include "../DeselbyStd/typeutils.h"

namespace abm::events {
    template<class MESSAGE> struct OutgoingMessage;
    template<class MESSAGE> struct MessageReward;
    struct IsEndEpisodeMessage;
}

namespace abm {

    /** An indicator of which actions a body can deal with */
    template<class MASK, class ACTION>
    concept ActionMask = requires(MASK actionMask, ACTION action) {
  //      { actionMask.any() } -> std::convertible_to<bool>;      // true if there are any legal actions
        { actionMask[action] } -> std::convertible_to<bool>;    // true if action is legal
    };

    /** An IntegralActionMask is an ActionMask where the action domain is integral */
    template<class T>
    concept IntegralActionMask = requires(T actionMask) {
        { actionMask.size()  } -> std::integral; // the cardinality of the set of all actions.
        { actionMask.count() } -> std::integral; // the number of legal actions
        { actionMask[actionMask.size()] } -> std::convertible_to<bool>;    // true if action is legal
    };



    /** Takes a IntegralActionMask and returns a vector of indices of true bits
     * @return a vector containing the indices of each bit in legalActs that is true
     */
    template<IntegralActionMask MASK>
    static auto legalIndices(const MASK &legalActs) {
        typedef decltype(legalActs.size()) action_type;
        std::vector<action_type> indices;
        indices.reserve(legalActs.size());
        for(action_type i=0; i<legalActs.size(); ++i)
            if(legalActs[i]) indices.push_back(i);
        return indices;
    }


    template<class BODY, class MESSAGE>
    concept CanHandlesMessage = requires(BODY body, MESSAGE messageFromEnvironment) {
        { body.handleMessage(messageFromEnvironment) } -> std::same_as<double>; // updates body and returns a reward
    };

    /** Can these types communicate as BODY and MIND */
    template<class BODY, class MIND>
    concept BodyMindPair = requires(BODY body, MIND mind) {
        { body.handleAct(mind.act(body)) } -> deselby::IsUniquelyConvertibleToTemplate<events::OutgoingMessage>;
        { body.legalActs() } -> ActionMask<decltype(mind.act(body))>; // returns a mask of legal acts
    };

    /** A body/mind monad is one where the body doesn't send any messages out or handle any
     * incoming messages. So the communication is entirely act/reward between mind and body. */
    template<class BODY, class MIND>
    concept BodyMindMonad = requires(BODY body, MIND mind) {
        { body.handleAct(mind.act(body)) } -> std::convertible_to<events::MessageReward<events::IsEndEpisodeMessage>>;
        { body.legalActs() } -> ActionMask<decltype(mind.act(body))>; // returns a mask of legal acts
    };


    /** A loss function is a measure of how close an approximator function, F, is to a target function, G.
     * It depends only on evaluations of F at a vector of training-points, X = (X_1...X_n).
     * So the LossFunction is a function from a vector of 'predictions' (Y_1...Y_n), where Y_i = F(X_i),
     * to a real-valued distance.
     *
     * In particular, the gradient of this funciton.
     *
     * The loss function may represent a sum of log-probs.
     *
     */
    template<class T>
    concept LossFunction = requires(T obj, arma::mat &result, arma::mat predictions) {
        obj.batchSize();                  // number of points returned from obj.trainingSet(.)
        obj.trainingSet(result);  // inserts training points into result. One column per training-point
        obj.gradientByPrediction(predictions, result);  // sets result = dLoss/dPredictions,
                                                        // predictions should be one column per prediction,
                                                        // in the same order as given in the trainingMatrix
//        obj.loss(predictions, result); // sets result = Loss(predictions)
    };

    /** Just that */
    template<class T>
    concept ParameterisedFunction = requires(T obj) {
        { obj.parameters() } -> std::same_as<arma::mat &>;
    };

    /** A parameterised function that can do backpropogation with a loss function */
    template<class T, class LOSSFUNCTION>
    concept DifferentiableParameterisedFunction = requires(T obj, LOSSFUNCTION loss) {
        { obj.parameters() } -> std::same_as<arma::mat &>;
        { obj.gradientByParams(loss) } -> std::same_as<arma::mat>;
    };


/** Minimum requirements of a QVector :
     * must be a sized, indexable of ordered objects
     * */
    template<class T>
    concept GenericQVector = requires(T obj, size_t i) {
        { obj[i] < obj[i] } -> std::convertible_to<bool>;
        obj.size();
    };


}

#endif //MULTIAGENTGOVERNMENT_ABM_CONCEPTS_H
