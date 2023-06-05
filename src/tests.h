//
// Created by daniel on 01/06/23.
//

#ifndef MULTIAGENTGOVERNMENT_TESTS_H
#define MULTIAGENTGOVERNMENT_TESTS_H

#include <mlpack.hpp>

void feedForwardNeuralNetTest() {
    arma::mat trainData = {{1,1,0,0},
                           {1,0,1,0}};

    arma::mat trainTarget = {3.5, 2.5, 1.5, 0.5};

    std::cout << trainData << std::endl;
    std::cout << trainTarget << std::endl;

    mlpack::FFN<mlpack::MeanSquaredErrorType<>> network;
//    mlpack::FFN<mlpack::L1LossType<>> network;
//    network.Add<mlpack::Linear>(2);
//    network.Add<mlpack::Sigmoid>();
    network.Add<mlpack::Linear>(1);

    ens::Adam optimiser;
    optimiser.ResetPolicy() = false;
    for(int i=0; i<1000; ++i) {
        network.Train(trainData, trainTarget, optimiser);
    }


    arma::mat predictions;
    network.Predict(trainData, predictions);
    std::cout << predictions << std::endl;
}


void cartPoleDQNTest() {
    typedef mlpack::CartPole Environment;

    mlpack::TrainingConfig config;//(1, 1, 100, 200, 100, 0.01, 0.9, 1.0, false, true, false, 1, 1, 1, 1.0);
    config.StepSize() = 0.001;
    config.Discount() = 1.0;
    config.TargetNetworkSyncInterval() = 5;
    config.ExplorationSteps() = 100;
    config.DoubleQLearning() = false;
    config.StepLimit() = 200;

    mlpack::SimpleDQN<> qNetwork(100,50,2);
    mlpack::GreedyPolicy<Environment> policy(1.0, 100, 0.01, 0.97);
    mlpack::RandomReplay<Environment> replay(64,100000);
//    mlpack::PrioritizedReplay<Environment> replay(10, 10000, 0.6);
    ens::AdamUpdate updater;
    Environment environment;

    mlpack::QLearning<
            mlpack::CartPole,
            mlpack::SimpleDQN<>,
            ens::AdamUpdate,
            mlpack::GreedyPolicy<Environment>,
            mlpack::RandomReplay<Environment>> model(config, qNetwork, policy, replay, updater, environment);

    double smoothedReturn = 20.0;
    int iterations = 0;
    while(smoothedReturn < 199.999) {
        double episodeReturn = model.Episode();
        smoothedReturn = 0.95*smoothedReturn + 0.05*episodeReturn;
        std::cout << smoothedReturn << std::endl;
        ++iterations;
    }
    std::cout << "Learned to balance a pole in " << iterations << " episodes." << std::endl;
}


arma::Col<size_t> BestAction(const arma::mat& actionValues)
{
    // Take best possible action at a particular instance.
    arma::Col<size_t> bestActions(actionValues.n_cols);
    arma::rowvec maxActionValues = arma::max(actionValues, 0);
    for (size_t i = 0; i < actionValues.n_cols; ++i)
    {
        bestActions(i) = arma::as_scalar(
                arma::find(actionValues.col(i) == maxActionValues[i], 1));
    }
    return bestActions;
};


// given a set of replay data, a network and an optimisation algorithm, we
// can train the network.
// The learningNetwork is the one that should be updated, while the targetNetwork
// is the one from which we approximate future expected rewards from the end state
// of each step.
template <
        typename NetworkType,
        typename UpdatePolicyType,
        typename ReplayType,
        typename ConfigType
>
void QTrainAgent(const ReplayType &replayMethod, ConfigType config, UpdatePolicyType updatePolicy, NetworkType &targetNetwork,  NetworkType &learningNetwork)
{
    // Start experience replay.

    // Sample from previous experience.
    arma::mat sampledStates;
    std::vector<typename ReplayType::ActionType> sampledActions;
    arma::rowvec sampledRewards;
    arma::mat sampledNextStates;
    arma::irowvec isTerminal;

    replayMethod.Sample(sampledStates, sampledActions, sampledRewards,
                        sampledNextStates, isTerminal);

    // Compute action value for next state with target network.

    arma::mat nextActionValues;
    targetNetwork.Predict(sampledNextStates, nextActionValues);

    arma::Col<size_t> bestActions;
    if (config.DoubleQLearning())
    {
        // If use double Q-Learning, use learning network to select the best action.
        arma::mat nextActionValues;
        learningNetwork.Predict(sampledNextStates, nextActionValues);
        bestActions = BestAction(nextActionValues);
    }
    else
    {
        bestActions = BestAction(nextActionValues);
    }

    // Compute the update target.
    arma::mat target;
    learningNetwork.Forward(sampledStates, target);

    double discount = std::pow(config.Discount(), replayMethod.NSteps());

    /**
     * If the agent is at a terminal state, then we don't need to add the
     * discounted reward. At terminal state, the agent wont perform any
     * action.
     */
    for (size_t i = 0; i < sampledNextStates.n_cols; ++i)
    {
        target(sampledActions[i].action, i) = sampledRewards(i) + discount *
                                                                  nextActionValues(bestActions(i), i) * (1 - isTerminal[i]);
    }

    // Learn from experience.
    arma::mat gradients;
    learningNetwork.Backward(sampledStates, target, gradients);

    replayMethod.Update(target, sampledActions, nextActionValues, gradients);

#if ENS_VERSION_MAJOR == 1
    updater.Update(learningNetwork.Parameters(), config.StepSize(), gradients);
#else
    updatePolicy->Update(learningNetwork.Parameters(), config.StepSize(),
                         gradients);
#endif

    if (config.NoisyQLearning() == true)
    {
        learningNetwork.ResetNoise();
        targetNetwork.ResetNoise();
    }
    // Update target network.
//    if (totalSteps % config.TargetNetworkSyncInterval() == 0)
//        targetNetwork.Parameters() = learningNetwork.Parameters();

//    if (totalSteps > config.ExplorationSteps())
//        policy.Anneal();
}


#endif //MULTIAGENTGOVERNMENT_TESTS_H
