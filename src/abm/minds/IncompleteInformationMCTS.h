// Implementation of Monte Carlo Tree Search for two agents that have
// incomplete information about the other's internal state, but
// each agent knows the reward function of both agents and the
// state transitions for
//
// At any point we only care about other's state to the extent that it
// affects other's reward structure...so state is an input to reward.
// Reward of any agent is a function of
// (state-before-last-decision, channel/message, state-before-next-decision)
//
// We assume that both agents share a prior distribution over their own and other's state
// and that an agent's belief about the other's state is simply the conditional distribution
// given the acts observed so far [esp, this doesn't depend on the believer's state].
//
// BODY must provide:
//  - state transition function: newState(oldState, outMessage, inMessage)
//  - reward function: reward(oldState, outMessage, newState)
// AGENTSTATEDISTRIBUTION must provide:
//  - bayesianUpdate(likelihoodFunction)
//  - map(stateTransitionFunction) which maps the distribution to a different domain
//
// NB: If an agent's policy is completely random. it tells us nothing about its state
// but as the Q values change, so too do the state distributions and so the weight of the
// samples. However, MCTS takes the current Q values for the next sample, glossing over this
// change (should we exponentially weight down?).
//
// A change in policy at the ancestors of a tree node affects the posterior there through
// changes in the likelihood function, so we need to, at least periodically, update the
// posteriors [this is bad because a change in policy at the root node changes all likelihoods
// ].
//
// Perhaps agents should fixate on the opponent hidden states that maximise probability times reward
// for each action, as these are the greatest contributors to expected reward...
//
// Or we choose the playout that will give us the most information about the best next action given
// the current state of the tree. We can calculate this by considering the playout as drawn from the
// distribution of a fictitious pair of agents, and then use importance sampling for all states on the
// same path of public information.
//
// Or play out on categories of states, grouped by posterior probability.
//
// If we could sample the trajectories with a known probability (approximately) proportional to
// the cumulative reward at the root times the current probability of the trajectory, then
// we could approximate Q values at the root very efficiently.
//
// TODO: dealing with multiple equilibria (language learning). If we use a Q-approximator for
//  off-tree decisions (and perhaps bias new tree nodes towards the Q-approximation?) then the Q-approximator
//  defines which equilibrium we end up in (as well as potentially making the algorithm more efficient).
//  If the Q-approximator is from body state to Q-vector, then there is flexibility on how much of the
//  episode history is included in the body state.
//  The Q-approximation is not changed during self-play. When I make a real move, my Q-approximation
//  learns from the Q-values of the root of the tree on my current body state [also the other body states in the root-node give
//  us information, should we use these?]. On your move, I observe your actual move but don't know your body-state.
//  However, I have samples from your body distribution given past moves in the episode.
//  If there are one or more body states on the tree-root that would have chosen the observed move, choose one
//  at random and learn from its Q-values. If no body states would choose the observed move, then choose the
//  one that requires the minimal zero-sum perturbation to its Q-values to make the observed action, and learn from
//  the minimally perturbed Q-values [or choose weighted by the size of the perturbation? or just don't learn?].
//  [N.B. we should properly use the hindcast body-states at the end of the episode]
//  No batching? (Adam update)
//  N.B. The test of the whole algorithm is whether we end up with a society that ends up converging to social norms.
//  (e.g. can it converge to language use by convention?)
//
// Created by daniel on 09/07/23.
//

#ifndef MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
#define MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H

#include <vector>
#include <map>
#include <functional>
#include <cassert>
#include <algorithm>
#include <random>
#include <bitset>
#include <ranges>
#include "../Agent.h"
#include "../GreedyPolicy.h"
#include "../bodies/MessageRecorder.h"
#include "../societies/RandomEncounterSociety.h"
#include "../QVector.h"
#include "../UpperConfidencePolicy.h"
#include "ZeroIntelligence.h"
#include "../../DeselbyStd/stlstream.h"
#include "../episodes/SimpleEpisode.h"
#include "../../approximators/FeedForwardNeuralNet.h"
#include "../../approximators/Concepts.h"
#include "../../observations/InputOutput.h"

namespace abm::minds {
    /**
     *
     * @tparam BODY The body with which we should play out in order to build this tree
     */
    template<Body BODY,
            class OffTreeQFunction, // Function from <body,legalMove> to action trainable on I/O pairs
            class SelfPlayPolicy = abm::UpperConfidencePolicy<typename BODY::action_type>>
    class IncompleteInformationMCTS {
    public:

        typedef BODY observation_type;
        typedef BODY::action_mask action_mask;
        typedef double reward_type;
        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible
        typedef BODY body_type;
        typedef const episodes::SimpleEpisode<BODY,BODY> & init_type;

        inline static std::default_random_engine randomGenerator = std::default_random_engine();

        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
        class TreeNode {
        public:
            typedef std::map<BODY, QVector<action_type::size>>::iterator iterator;

            std::map<BODY, QVector<action_type::size>> qEntries;   // QVectors by agent state for this node
            std::map<BODY,uint>     otherPlayerDistribution; // counts of other player states by body state

        private:
            std::array<TreeNode *, static_cast<size_t>(message_type::size)> children;   // ...indexed by actId. nullptr if child not present.
        public:

            TreeNode() { children.fill(nullptr); }
            ~TreeNode() { for (TreeNode *child: children) delete (child); }

            TreeNode *getChildOrCreate(message_type message);
            TreeNode *getChildOrNull(message_type message) { return children[static_cast<size_t>(message)]; }
            TreeNode *unlinkChild(message_type message);
            iterator addQEntry(const BODY &agent);
            std::function<BODY()> createNextMoverSampler();
            std::function<BODY()> createOtherPlayerSampler();
            size_t nSamples();
            const BODY *sampleActorBodyGivenMessage(message_type actorMessage);
        };

        /** A SelfPlayMind is a Mind used to build the tree using self-play.
         * It can be thought of as a smart pointer into the tree that moves from the root
         * to a leaf then back-propagates to update Q-values.
         * During self-play, two SelfPlayMinds simultaneously navigate the same tree.
         * Each mind can add at most a single qEntry on their turn to act.
         * While a Mind hasn't yet added a qEntry, it may also create new TreeNodes in which
         * to put the new qEntry.
         * [So, there are three states, onTree/notAddedQEntry, onTree/AddedQentry. offTree]
         * @tparam POLICY for choosing an action given a QVector
         */
        class SelfPlayMind {
        public:
            typedef BODY observation_type;
            typedef BODY::action_mask action_mask;
            typedef double reward_type;

            IncompleteInformationMCTS<BODY,OffTreeQFunction, SelfPlayPolicy> &           tree;
            TreeNode *                                  treeNode;// current treeNodes for player's experience, null if off the tree
            std::vector<QValue *>                       qValues; // Q values at choice points of the player
            std::vector<double>                         rewards; // reward between choice points of the player
            bool                                        hasAddedQEntry;

//            SelfPlayMind(TreeNode *treeNode, double discount,const POLICY &policy, OffTreeMind &offTreeMind):
//                    treeNode(treeNode), discount(discount), onTreePolicy(policy), offTreeMind(offTreeMind), hasAddedQEntry(false) {}

            SelfPlayMind(IncompleteInformationMCTS<BODY,OffTreeQFunction, SelfPlayPolicy> &tree):
                    tree(tree), treeNode(tree.rootNode), hasAddedQEntry(false) {}

            action_type act(const observation_type &body, action_mask legalActs, reward_type rewardFromLastAct);
            // train(Reward);
            void endEpisode(double rewardFromFinalAct);
            void onOutgoingMessage(message_type outgoingMessage, const BODY &body);
//            void halfStepObservationHook(const observation_type &body);
            void onIncomingMessage(message_type incomingMessage, const BODY &body);
            void onInit() {
                treeNode = tree.rootNode;
                qValues.clear();
                rewards.clear();
                hasAddedQEntry = false;
            }
        };


        TreeNode *rootNode;                                 // points to the rootNode. nullptr signifies no acts this episode yet.
//        std::function<BODY()>     firstMoverPrior;
//        std::function<BODY()>     secondMoverPrior;
        size_t nSamplesPerTree;                             // number of samples taken to build the tree before a decision is made
        double discount;                                    // discount of rewards into the future
        GreedyPolicy<action_type> finalDecisionPolicy;      // policy used to make final decision once the tree is built
        SelfPlayPolicy  selfPlayPolicy; // policy used when building tree
        OffTreeQFunction     offTreeQMind;    // mind to decide acts during self-play when off the tree.

        IncompleteInformationMCTS(
                size_t nSamplesPerTree,
                double discount,
                OffTreeQFunction offTreeQFunction = approximators::FeedForwardNeuralNet<BODY::dimension, action_type::size, mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>>(mlpack::SimpleDQN<mlpack::MeanSquaredError, mlpack::HeInitialization>(100,50, action_type::size))
                );
        ~IncompleteInformationMCTS() { delete (rootNode); }

        // ----- Mind interface -----

        action_type act(const observation_type &body, action_mask legalActs, [[maybe_unused]] reward_type rewardFromLastAct);

        void onIncomingMessage(message_type incomingMessage, const BODY &body) {
            assert(rootNode != nullptr);
            // learn from opponent move
            const BODY *sampledOpponentBodyPtr = rootNode->sampleActorBodyGivenMessage(incomingMessage);
            if(sampledOpponentBodyPtr != nullptr) {
                QVector<action_type::size> qVec = rootNode->qEntries[*sampledOpponentBodyPtr];
                offTreeQMind.train(observations::InputOutput((const BODY &)*sampledOpponentBodyPtr, qVec.toVector()));
            }
            shiftRoot(incomingMessage);
        }

        void onOutgoingMessage(message_type outgoingMessage, const BODY &body) {
            assert(rootNode != nullptr);
            shiftRoot(outgoingMessage);
        }

        void endEpisode([[maybe_unused]] double rewardFromFinalAct) {
            delete(rootNode);
            rootNode = nullptr;
        }

        void onInit(init_type initialDistribution) {
            rootNode = new TreeNode();
            buildTree(initialDistribution);
        }

    protected:
        void buildTree(const episodes::SimpleEpisode<BODY> &episode);
        void shiftRoot(message_type message);

    };

    /** Given that the current actor in this node produced a given message, sample from the posterior
     *
     * @tparam BODY
     * @tparam OffTreeQFunction
     * @tparam SelfPlayPolicy
     * @param actorMessage
     * @return
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    const BODY *
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::sampleActorBodyGivenMessage(message_type actorMessage) {
        double totalWeight = 0.0;
        std::vector<double>         cumulativeWeights;
        std::vector<const BODY *>   states;
        GreedyPolicy<action_type> maxQPolicy(explorationStrategies::NoExploration());
        for(const auto &[body, qVec]: qEntries) {
            action_type maxQAct = maxQPolicy.sample(qEntries, body.legalMoves());
            double w = body.actToMessageProb(maxQAct, actorMessage);
            totalWeight += w;
            cumulativeWeights.push_back(totalWeight);
            states.push_back(&body);
        }
        if(totalWeight == 0.0) { /* no state in this mode could have produced this message */
            return nullptr;
        }
        // sample an element from cumulativeWeights with weighted prob
        double rand = deselby::Random::nextDouble(0.0, totalWeight);
        auto chosenIt = std::ranges::upper_bound(cumulativeWeights, rand);
        assert(chosenIt != cumulativeWeights.end());
        return states[chosenIt - cumulativeWeights.begin()];
    }


    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY,OffTreeQFunction,SelfPlayPolicy>::action_type
    IncompleteInformationMCTS<BODY,OffTreeQFunction,SelfPlayPolicy>::act(const observation_type &body, action_mask legalActs,
                                                     IncompleteInformationMCTS::reward_type rewardFromLastAct) {
        // onTree and can add entry so try emplace
//            std::cout << "Available bodies \n" << (rootNode->qEntries | std::views::keys) << std::endl;
//            std::cout << "Search body \n" << body << std::endl;
//            std::cout << "Equality vector\n" << (rootNode->qEntries | std::views::keys | std::views::transform([&body](auto &bod) { return bod == body; })) << std::endl;
//            std::cout << "Size of qEntry map " << rootNode->qEntries.size() << std::endl;
//            std::cout << "Size of other  map " << rootNode->otherPlayerDistribution.size() << std::endl;
//            std::cout << "Nsamples " << rootNode->nSamples() << std::endl;

        assert(rootNode != nullptr);
        buildTree(episodes::SimpleEpisode(rootNode->createNextMoverSampler(), rootNode->createOtherPlayerSampler()));

        QVector<action_type::size> qVec = rootNode->qEntries.at(body);
//            std::cout << "QVector = " << qVec << std::endl;
        // Train off-tree function on tree Q-values
        offTreeQMind.train(observations::InputOutput(body, qVec.toVector()));

        action_type action = finalDecisionPolicy.sample(qVec, legalActs);
        return action;
    }

//    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
//    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::halfStepObservationHook(const observation_type &body) {
//        if(treeNode != nullptr && !hasAddedQEntry) treeNode->otherPlayerDistribution[body]++;
//    }

    /**
     *
     * @param message identifies the child to unlink
     * @return the unlinked child
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode *IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::unlinkChild(message_type message) {
        const size_t childIndex = static_cast<size_t>(message);
        TreeNode *child = children[childIndex];
        children[childIndex] = nullptr;
        return child;
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::onIncomingMessage(message_type incomingMessage, const BODY &body) {
        if(treeNode != nullptr) {
            if(hasAddedQEntry)
                treeNode = treeNode->getChildOrNull(incomingMessage);
            else
                treeNode = treeNode->getChildOrCreate(incomingMessage);
        }
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::onOutgoingMessage(message_type outgoingMessage, const BODY &body) {
        if(treeNode != nullptr) {
            if(hasAddedQEntry) {
                treeNode = treeNode->getChildOrNull(outgoingMessage);
            } else {
                treeNode = treeNode->getChildOrCreate(outgoingMessage);
                treeNode->otherPlayerDistribution[body]++;
            }
        }
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::endEpisode(double rewardFromFinalAct) { // back propagate rewards
        double cumulativeReward = rewardFromFinalAct;
        while(rewards.size() > qValues.size()) {
            cumulativeReward = cumulativeReward*tree.discount + rewards.back();
            rewards.pop_back();
        }
        while(!rewards.empty()) {
            qValues.back()->addSample(cumulativeReward);
            cumulativeReward = cumulativeReward*tree.discount + rewards.back();
            rewards.pop_back();
            qValues.pop_back();
        }
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::action_type
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::SelfPlayMind::act(const observation_type &body, action_mask legalActs,
                                                               IncompleteInformationMCTS::reward_type rewardFromLastAct) {
//        std::cout << "rewardFromLastAct " << rewardFromLastAct << std::endl;
        QVector<action_type::size> *qEntry = nullptr; // if null, choose at random
        if(treeNode != nullptr) {
            if (hasAddedQEntry) {
                // onTree but can't add new qEntry, so look for existing entry or null
                auto qIt = treeNode->qEntries.find(body);
                if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second);
            } else {
                auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
                if (insertedNewEntry) {
//                    std::cout << "Added QEntry " << std::endl;
                    hasAddedQEntry = true;
                }
                qEntry = &(qIt->second);
            }
        }
        action_type action;
        if(qEntry != nullptr) { // on tree
            action = tree.selfPlayPolicy.sample(*qEntry, legalActs);
            rewards.push_back(rewardFromLastAct);
            qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]);
        } else { // off-tree
            action = tree.offTreeQMind(body);
            // static_cast<action_type>(sampleUniformly(legalActs));
            rewards.push_back(rewardFromLastAct);
        }
        return action;
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    std::function<BODY()> IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::createOtherPlayerSampler() {
        std::vector<BODY>   states;
        std::vector<double> weights;
        if(otherPlayerDistribution.size() == 0) return {}; // undefined if no data
//        assert(otherPlayerDistribution.size() > 0);
        states.reserve(otherPlayerDistribution.size());
        weights.reserve(otherPlayerDistribution.size());
        for(auto &entry: otherPlayerDistribution) {
            states.emplace_back(entry.first);
            weights.emplace_back(entry.second);
        }
        auto distribution = std::discrete_distribution(weights.begin(), weights.end());

        return [distribution, states, &gen = randomGenerator]() mutable {
            return states[distribution(gen)];
        };
    }


    /**
     * @return the total number of samples that have passed this tree node.
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    size_t IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::nSamples() {
        size_t sum = 0;
        for(const auto &entry: qEntries) sum += entry.second.totalSamples();
        return sum;
    }


    /**
     * @return A function that returns a sample from the body states with a probability
     * proportional to the total number of samples in each qEntry of this node.
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    std::function<BODY()> IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::createNextMoverSampler() {
        std::vector<BODY>   states;
        std::vector<double> weights;
        assert(qEntries.size() > 0);
        states.reserve(qEntries.size());
        weights.reserve(qEntries.size());
        for(auto &entry: qEntries) {
            states.emplace_back(entry.first);
            weights.emplace_back(entry.second.totalSamples());
        }
        auto distribution = std::discrete_distribution(weights.begin(), weights.end());

        return [distribution, states, &gen = randomGenerator]() mutable {
            return states[distribution(gen)];
        };
    }

    /** When either agent actually moves, we need to move the root node down the tree
     * and delete tree nodes that are above the new root.
     *
     * @param message identifies the child of the rootNode that will become the new root
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::shiftRoot(message_type message) {
        TreeNode *newRoot = rootNode->unlinkChild(message);
        assert(newRoot != nullptr);
        delete(rootNode);
        rootNode = newRoot;
    }



    /** Build a tree from the
     *
     * @param nSamples
     * @param discount
     */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    void IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::buildTree(const episodes::SimpleEpisode<BODY> &episode) {
        for (size_t nSamples = rootNode->nSamples(); nSamples < nSamplesPerTree; ++nSamples) {
            Agent<BODY,SelfPlayMind> player1(episode.sampleFirstMoverBody(), SelfPlayMind(*this));
            Agent<BODY,SelfPlayMind> player2(episode.sampleSecondMoverBody(), SelfPlayMind(*this));
            episodes::episode(player1,player2);
        }
    }

    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::IncompleteInformationMCTS(size_t nSamplesPerTree, double discount, OffTreeQFunction offTreeMind) :
            rootNode(nullptr),
            nSamplesPerTree(nSamplesPerTree),
            discount(discount),
            finalDecisionPolicy(explorationStrategies::NoExploration()),
            offTreeQMind(std::move(offTreeMind)) { }


    /** Make an entry for a new body state.
    *
    * @param agent
    * @return
    */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::iterator
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::addQEntry(const BODY &agent) {
        auto [newEntry, wasInserted] = qEntries.try_emplace(
                agent); // insert uniform quality with zero sample count
        assert(wasInserted);
        return newEntry;
    }


    /** Returns the child of this node corresponding to the given act,
    * creating a new one if necessary
    *
    * @param act the act that identifies the child
    * @return a pointer to the child
    */
    template<Body BODY, class OffTreeQFunction, class SelfPlayPolicy>
    IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode *IncompleteInformationMCTS<BODY, OffTreeQFunction, SelfPlayPolicy>::TreeNode::getChildOrCreate(message_type message) {
        TreeNode *childNode = children[static_cast<size_t>(message)];
        if (childNode == nullptr) {
            childNode = new TreeNode();
            children[static_cast<size_t>(message)] = childNode;
        }
        return childNode;
    }
}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
