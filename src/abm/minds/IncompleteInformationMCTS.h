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
//
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
#include "../utils.h"
#include "../QVector.h"
#include "../UpperConfidencePolicy.h"
#include "ZeroIntelligence.h"

namespace abm::minds {

    /**
     *
     * @tparam BODY The body with which we should play out in order to build this tree
     */
    template<Body BODY>
    class IncompleteInformationMCTS {
    public:

        typedef BODY observation_type;
        typedef BODY::action_mask action_mask;
        typedef double reward_type;

        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible
        typedef BODY body_type;

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
            std::function<BODY()> createNextPlayerSampler();
            std::function<BODY()> createOtherPlayerSampler();
            size_t nSamples();
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
        template<class POLICY>
        class SelfPlayMind {
        public:
            typedef BODY observation_type;
            typedef BODY::action_mask action_mask;
            typedef double reward_type;

            TreeNode *                                  treeNode;// current treeNodes for player's experience, null if off the tree
            std::vector<QValue *>                       qValues; // Q values at choice points of the player
            std::vector<double>                         rewards; // reward between choice points of the player
            double                                      discount;
            POLICY                                      policy;
            bool                                        hasAddedQEntry;

            SelfPlayMind(TreeNode *treeNode, POLICY policy, double discount):
            treeNode(treeNode), discount(discount), policy(policy), hasAddedQEntry(false) {}

            action_type act(const observation_type &body, action_mask legalActs, reward_type rewardFromLastAct);
            void endEpisode(double rewardFromFinalAct);
            void outgoingMessageHook(message_type outgoingMessage);
            void halfStepObservationHook(const observation_type &body);
            void incomingMessageHook(message_type incomingMessage);
        };


        TreeNode *rootNode;                                 // points to the rootNode. nullptr signifies no acts this episode yet.
        std::function<std::function<BODY()>(const BODY &)>     thisSamplerFactory;
        std::function<std::function<BODY()>(const BODY &)>     otherSamplerFactory;
        size_t nSamplesPerTree;                             // number of samples taken to build the tree before a decision is made
        double discount;                                    // discount of rewards into the future
        GreedyPolicy<action_type> policy;                   // fixed policy for now (greedy, no exploration)

        explicit IncompleteInformationMCTS(size_t nSamplesPerTree, double discount,
                                           std::function<std::function<BODY()>(
                                                   const BODY &)> thisSamplerFactory,
                                           std::function<std::function<BODY()>(
                                                   const BODY &)> otherSamplerFactory);
        ~IncompleteInformationMCTS() { delete (rootNode); }

        // ----- Mind interface -----

        action_type act(const observation_type &body, action_mask legalActs, [[maybe_unused]] reward_type rewardFromLastAct) {
            if(rootNode == nullptr) { // must be first move
                rootNode = new TreeNode();
                buildTree(thisSamplerFactory(body), otherSamplerFactory(body));
            } else {
                buildTree(rootNode->createNextPlayerSampler(), rootNode->createOtherPlayerSampler());
            }
            QVector<action_type::size> qVec = rootNode->qEntries.at(body);
            action_type action = policy.sample(qVec, legalActs);
            return action;
        }

        void incomingMessageHook(message_type incomingMessage) { shiftRoot(incomingMessage); }

        void outgoingMessageHook(message_type outgoingMessage) { shiftRoot(outgoingMessage); }

        void endEpisode([[maybe_unused]] double rewardFromFinalAct) {
            delete(rootNode);
            rootNode = nullptr;
        }

    protected:
        void buildTree(std::function<BODY()> thisBodySampler, std::function<BODY()> otherBodySampler);
        void shiftRoot(message_type message);

    };

    template<Body BODY>
    template<class POLICY>
    void IncompleteInformationMCTS<BODY>::SelfPlayMind<POLICY>::halfStepObservationHook(const observation_type &body) {
        if(treeNode != nullptr && !hasAddedQEntry) treeNode->otherPlayerDistribution[body]++;
    }

    /**
     *
     * @param message identifies the child to unlink
     * @return the unlinked child
     */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::TreeNode *IncompleteInformationMCTS<BODY>::TreeNode::unlinkChild(message_type message) {
        const size_t childIndex = static_cast<size_t>(message);
        TreeNode *child = children[childIndex];
        children[childIndex] = nullptr;
        return child;
    }

    template<Body BODY>
    template<class POLICY>
    void IncompleteInformationMCTS<BODY>::SelfPlayMind<POLICY>::incomingMessageHook(message_type incomingMessage) {
        if(treeNode != nullptr) {
            if(hasAddedQEntry)
                treeNode = treeNode->getChildOrNull(incomingMessage);
            else
                treeNode = treeNode->getChildOrCreate(incomingMessage);
        }
    }

    template<Body BODY>
    template<class POLICY>
    void IncompleteInformationMCTS<BODY>::SelfPlayMind<POLICY>::outgoingMessageHook(message_type outgoingMessage) {
        if(treeNode != nullptr) {
            if(hasAddedQEntry)
                treeNode = treeNode->getChildOrNull(outgoingMessage);
            else
                treeNode = treeNode->getChildOrCreate(outgoingMessage);
        }
    }

    template<Body BODY>
    template<class POLICY>
    void IncompleteInformationMCTS<BODY>::SelfPlayMind<POLICY>::endEpisode(double rewardFromFinalAct) { // back propagate rewards
        double cumulativeReward = rewardFromFinalAct;
        while(rewards.size() > qValues.size()) {
            cumulativeReward = cumulativeReward*discount + rewards.back();
            rewards.pop_back();
        }
        while(!rewards.empty()) {
            cumulativeReward = cumulativeReward*discount + rewards.back();
            qValues.back()->addSample(cumulativeReward);
            rewards.pop_back();
            qValues.pop_back();
        }
    }

    template<Body BODY>
    template<class POLICY>
    IncompleteInformationMCTS<BODY>::action_type
    IncompleteInformationMCTS<BODY>::SelfPlayMind<POLICY>::act(const observation_type &body, action_mask legalActs,
                                                               IncompleteInformationMCTS::reward_type rewardFromLastAct) {
        QVector<action_type::size> *qEntry = nullptr; // if null, choose at random
        if(treeNode != nullptr) {
            if (hasAddedQEntry) {
                // onTree but can't add new qEntry, so look for existing entry or null
                auto qIt = treeNode->qEntries.find(body);
                if(qIt != treeNode->qEntries.end()) qEntry = &(qIt->second);
            } else {
                // onTree and can add entry so try emplace
                auto [qIt, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
                if (insertedNewEntry) hasAddedQEntry = true;
                qEntry = &(qIt->second);
            }
        }
        action_type action;
        if(qEntry != nullptr) {
            action = policy.sample(*qEntry, legalActs);
            rewards.push_back(rewardFromLastAct);
            qValues.push_back(&(*qEntry)[static_cast<size_t>(action)]);
        } else { // off-tree
            action = static_cast<action_type>(sampleUniformly(legalActs));
            rewards.push_back(rewardFromLastAct);
        }
        return action;
    }

    template<Body BODY>
    std::function<BODY()> IncompleteInformationMCTS<BODY>::TreeNode::createOtherPlayerSampler() {
        std::vector<BODY>   states;
        std::vector<double> weights;
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
    template<Body BODY>
    size_t IncompleteInformationMCTS<BODY>::TreeNode::nSamples() {
        size_t sum = 0;
        for(const auto &entry: otherPlayerDistribution) sum += entry.second;
        return sum;
    }


    /**
     * @return A function that returns a sample from the body states with a probability
     * proportional to the total number of samples in each qEntry of this node.
     */
    template<Body BODY>
    std::function<BODY()> IncompleteInformationMCTS<BODY>::TreeNode::createNextPlayerSampler() {
        std::vector<BODY>   states;
        std::vector<double> weights;
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
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::shiftRoot(message_type message) {
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
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::buildTree(std::function<BODY()> thisBodySampler,
                                                    std::function<BODY()> otherBodySampler) {
        for (size_t nSamples = rootNode->nSamples(); nSamples < nSamplesPerTree; ++nSamples) {
            auto player1 = Agent(thisBodySampler(),
                                 SelfPlayMind(rootNode,
                                              UpperConfidencePolicy<action_type>(),
                                              discount)
            );
            auto player2 = Agent(otherBodySampler(),
                                 SelfPlayMind(rootNode,
                                              UpperConfidencePolicy<action_type>(),
                                              discount)
            );
            episode(player1,player2);
        }
    }

    template<Body BODY>
    IncompleteInformationMCTS<BODY>::IncompleteInformationMCTS(size_t nSamplesPerTree, double discount,
                                                               std::function<std::function<BODY()>(
                                                                       const BODY &)> thisSamplerFactory,
                                                               std::function<std::function<BODY()>(
                                                                       const BODY &)> otherSamplerFactory) :
            rootNode(nullptr),
            thisSamplerFactory(thisSamplerFactory),
            otherSamplerFactory(otherSamplerFactory),
            nSamplesPerTree(nSamplesPerTree),
            discount(discount),
            policy(NoExploration()) { }


    /** Make an entry for a new body state.
    *
    * @param agent
    * @return
    */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::TreeNode::iterator
    IncompleteInformationMCTS<BODY>::TreeNode::addQEntry(const BODY &agent) {
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
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::TreeNode *IncompleteInformationMCTS<BODY>::TreeNode::getChildOrCreate(message_type message) {
        TreeNode *childNode = children[static_cast<size_t>(message)];
        if (childNode == nullptr) {
            childNode = new TreeNode();
            children[static_cast<size_t>(message)] = childNode;
        }
        return childNode;
    }
}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
