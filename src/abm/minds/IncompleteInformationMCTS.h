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

namespace abm::minds {
/**
 *
 * @tparam BODY The body with which we should play out in order to build this tree
 */
    template<Body BODY>
    class IncompleteInformationMCTS {
    public:

        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type; // in and out messages must be the same for self-play to be possible

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
            std::array<TreeNode *, message_type::size> children;   // ...indexed by actId. nullptr if child not present.

            TreeNode() { children.fill(nullptr); }
            ~TreeNode() { for (TreeNode *child: children) delete (child); }

            TreeNode *getChildOrCreate(int messageId);
            iterator addQEntry(const BODY &agent);
            std::function<const BODY &()> createNextPlayerSampler();
            std::function<const BODY &()> createOtherPlayerSampler();
            size_t nSamples();
        };

        class SelfPlayBody {
        public:
            TreeNode *                                  treeNode;   // current treeNodes for player's experience, null if in random playout
            typename TreeNode::QVector *                qVec;       // qVector for the next choice point (lazily set)
            BODY                                        body;       // current state of player

            SelfPlayBody(TreeNode *rootNode, BODY body): treeNode(rootNode), body(std::move(body)) { }

            void reset(TreeNode *rootNode, BODY Body) {
                treeNode = rootNode;
                body = std::move(Body);
            }

            const QVector<action_type::size> &qVector() {
                if(qVec == nullptr) {
                    auto [qEntry, insertedNewEntry] = treeNode->qEntries.try_emplace(body);
                    if(insertedNewEntry) treeNode = nullptr; // start random playout
                    qVec = &(qEntry->second);
                }
                return *qVec;
            }

            message_type actToMessage(action_type act) {
                message_type message = body.actToMessage(act);
                if(isOnTree()) {
                    treeNode = treeNode->getChildOrCreate(message);
                    qVec = nullptr;
                }
                return message;
            }

            double messageToReward(message_type incomingMessage) {
                treeNode->otherPlayerDistribution[body]++;
                if(treeNode != nullptr) {
                    treeNode = treeNode->children[incomingMessage];
                    assert(treeNode != nullptr); // already generated by other player
                }
                double reward = body.messageToReward(incomingMessage);
                return reward;
            }

            BODY::mask_type legalActs() { return body.legalActs(); }

            [[nodiscard]] bool isOnTree() const { return treeNode != nullptr; }
        };

        template<class POLICY>
        class SelfPlayMind {
        public:
            typedef SelfPlayBody observation_type;
            typedef BODY::action_mask action_mask;
            typedef double reward_type;

            std::vector<typename TreeNode::QValue *>    qValues; // Q values at choice points of the player
            std::vector<double>                         rewards; // reward between choice points of the player
            double                                      discount;
            POLICY                                      policy;

            SelfPlayMind(POLICY policy, double discount): discount(discount), policy(policy) {}

            action_type act(const observation_type &body, action_mask legalActs, reward_type rewardFromLastAct) {
                action_type action;
                if(body.isOnTree()) {
                    action = policy.sample(body.qVec, legalActs);
                    rewards.push_back(rewardFromLastAct);
                    qValues.push_back(body.qVec->operator[](action));
                } else {
                    action = sampleUniformly(legalActs);
                    rewards.push_back(rewardFromLastAct);
                }
                return action;
            }

            void endEpisode(double rewardFromFinalAct) { // back propagate rewards
                double cumulativeReward = rewardFromFinalAct;
                while(rewards.size() > qValues.size()) {
                    cumulativeReward = cumulativeReward*discount + rewards.back();
                    rewards.pop_back();
                }
                while(!rewards.empty()) {
                    cumulativeReward = cumulativeReward*discount + rewards.back();
                    qValues.back().addSample(cumulativeReward);
                    rewards.pop_back();
                    qValues.pop_back();
                }
            }
        };


        TreeNode *rootNode;
        std::function<const BODY &()> nextPlayerSampler;      // sampler for the current root node (player to move next, state before decision)
        std::function<const BODY &()> otherPlayerSampler;   // sampler for the other player, state just before incoming message is received
        size_t nSamplesPerTree;                             // number of samples taken to build the tree before a decision is made
        double discount;                                    // discount of rewards into the future
        Agent<SelfPlayBody,SelfPlayMind<GreedyPolicy<action_type>>>        realPlayAgent;

        explicit IncompleteInformationMCTS(BODY initialBodyState, std::function<const BODY &()> priorBodySampler,
                                           size_t nSamplesPerTree, double discount);
        ~IncompleteInformationMCTS() { delete (rootNode); }


        // ----- Agent interface ------
        message_type startEpisode() {
            delete(rootNode);
            rootNode = new TreeNode();
            realPlayAgent.body.treeNode = rootNode;
            buildTree();
            message_type initialMessage = realPlayAgent.startEpisode();
            shiftRoot(initialMessage);
            return initialMessage;
        }

        message_type handleMessage(message_type incomingMessage) {
            buildTree();
            message_type response = realPlayAgent.handleMessage(incomingMessage);
            shiftRoot(incomingMessage);
            return response;
        }

    protected:
        void buildTree();
        void shiftRoot(message_type message);

    };

    template<Body BODY>
    std::function<const BODY &()> IncompleteInformationMCTS<BODY>::TreeNode::createOtherPlayerSampler() {
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
    std::function<const BODY &()> IncompleteInformationMCTS<BODY>::TreeNode::createNextPlayerSampler() {
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
        TreeNode *newRoot = rootNode->children[message];
        assert(newRoot != nullptr);
        rootNode->children[message] == nullptr;
        delete(rootNode);
        rootNode = newRoot;
        nextPlayerSampler = newRoot->createNextPlayerSampler();
        otherPlayerSampler = newRoot->createOtherPlayerSampler(); // TODO: implement this!
    }



    /** Build a tree from the
     *
     * @param nSamples
     * @param discount
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::buildTree() {
        Agent player1(
                SelfPlayBody(rootNode, nextPlayerSampler()),
                SelfPlayMind(UpperConfidencePolicy<action_type>(), discount)
        );
        Agent player2(
                SelfPlayBody(rootNode, otherPlayerSampler()),
                SelfPlayMind(UpperConfidencePolicy<action_type>(), discount)
        );
        for(size_t nSamples = rootNode->nSamples(); nSamples < nSamplesPerTree; ++nSamples) {
            player1.body.
            episode(player1, player2);
        }
    }

    template<Body BODY>
    IncompleteInformationMCTS<BODY>::IncompleteInformationMCTS(BODY initialBodyState, std::function<const BODY &()> priorBodySampler,
                                                               size_t nSamplesPerTree, double discount) :
            rootNode(nullptr),
            nextPlayerSampler(priorBodySampler),
            otherPlayerSampler(priorBodySampler),
            nSamplesPerTree(nSamplesPerTree),
            discount(discount),
            realPlayAgent(SelfPlayBody(nullptr, std::move(initialBodyState)), SelfPlayMind(GreedyPolicy<action_type>(), discount)) {

            }



    /** Choose a random body state from this node
    * @return iterator into the body->QValue map.
    */
//    template<Body BODY>
//    IncompleteInformationMCTS<BODY>::TreeNode::iterator IncompleteInformationMCTS<BODY>::TreeNode::randomEntry() {
//        auto uniformDist = std::uniform_int_distribution<int>(0, qEntries.size() - 1);
//        auto it = qEntries.begin();
//        std::advance(it, uniformDist(randomGenerator));
//        return it;
//    }

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
    IncompleteInformationMCTS<BODY>::TreeNode *IncompleteInformationMCTS<BODY>::TreeNode::getChildOrCreate(int messageId) {
        TreeNode *childNode = children[messageId];
        if (childNode == nullptr) {
            childNode = new TreeNode();
            children[messageId] = childNode;
        }
        return childNode;
    }



    //    /**
//     *
//     * @tparam BODY
//     * @return
//     */
//    template<Body BODY>
//    IncompleteInformationMCTS<BODY>::action_type IncompleteInformationMCTS<BODY>::Trajectory::Player::chooseAct() {
//        assert(treeNode != nullptr);
//        assert(treeNode->qEntries.contains(body));
//        typename TreeNode::QVector &qVec = treeNode->qEntries[body];
//        auto legalActs = body.legalActs();
//        action_type act = qVec.getUTCAction(body.legalActs());
//        qValues.push_back(qVec[act]);
//        return act;
//    }

//    /** Update each qEntry in this trajectory by incrementing the sample count and updating the Qvalue
//     * in-line with the cumulative reward to the end of the episode.
//     * Deletes the values in rewards as it goes.
//     *
//     * @param discount
//     */
//    template<Body BODY>
//    void IncompleteInformationMCTS<BODY>::Trajectory::backProp(double discount) {
//        for(int player = 0; player < 2; ++player) {
//            Player &currentPlayer = players[player];
//            double cumulativeReward = 0.0;
//            while(currentPlayer.rewards.size() > 0) {
//                cumulativeReward = currentPlayer.rewards.back() + discount * cumulativeReward;
//                currentPlayer.qValues.back().addSample(cumulativeReward);
//                currentPlayer.rewards.pop_back();
//                currentPlayer.qValues.pop_back();
//            }
//        }
//    }

//    /** Play out from the current trajectory state to the end of the episode
//     * using random moves,
//     * updates the final entries of the rewards for each player to be the cumulative
//     * rewards to the end of the episode.
//     *
//     * @param discount
//     */
//    template<Body BODY>
//    void IncompleteInformationMCTS<BODY>::Trajectory::playOut(double discount) {
////        std::uniform_int_distribution<int> uniformActDist(0, BODY::Action::size - 1);
//
//        BODY playerBodies[2] = {this->players[0].body, this->players[1].body};
//        double cumulativeRewards[2] = {0.0, 0.0};
//        double totalDiscount[2] = {1.0,1.0};
//        int currentPlayerIndex = nextPlayerIndex();
//        typename BODY::message_type message;
//        do {
//            // current player acts and generates message
//            BODY &currentPlayer = playerBodies[currentPlayerIndex];
//            const std::bitset<action_type::size> &legalActs = currentPlayer.legalActs();
//            if(legalActs.count() > 0) {
//                int nextAct = GreedyPolicy::sampleUniformly(legalActs);
//                message = currentPlayer.actToMessage(nextAct);
//            } else {
//                message = BODY::message_type::close;
//            }
//
//            // deliver message to other player and record reward
//            currentPlayerIndex = currentPlayerIndex ^ 1;
//            BODY &otherPlayer = playerBodies[currentPlayerIndex];
//            cumulativeRewards[currentPlayerIndex] = otherPlayer.messageToReward(message) * totalDiscount[currentPlayerIndex];
//            totalDiscount[currentPlayerIndex] *= discount;
//
//        } while (message != BODY::message_type::close);
//        players[0].rewards.push_back(cumulativeRewards[0]);
//        players[1].rewards.push_back(cumulativeRewards[1]);
//    }

//    template<Body BODY>
//    bool IncompleteInformationMCTS<BODY>::Trajectory::isLeaf() {
//        const Player & nextPlayer = players[nextPlayerIndex()];
//        return nextPlayer.treeNode == nullptr || !nextPlayer.treeNode->qEntries.contains(nextPlayer.body);
//    }

//    /** Adds a new QEntry for a state that isn't currently stored.
//     * This may or may not require a new TreeNode to be created.
//     * Assumes we're currently at a leaf. i.e.
//     */
//    template<Body BODY>
//    void IncompleteInformationMCTS<BODY>::Trajectory::expandTreeOneMove() {
//        assert(isLeaf());
//        int currentPlayerIndex = nextPlayerIndex();
//        Player &currentPlayer = players[currentPlayerIndex];
//        Player &otherPlayer = players[currentPlayerIndex^1];
//
//        if(currentPlayer.treeNode == nullptr) {
//            currentPlayer.treeNode = new TreeNode();
//            otherPlayer.treeNode->children[lastMessage] = currentPlayer.treeNode;
//        }
//        currentPlayer.treeNode->qEntries.try_emplace(currentPlayer.body);
//    }

//    /** The next player chooses his move and the trajectory moves forward to the
//     * next player's choice point.
//     *
//     * @return true if were NOT at a leaf.
//     */
//    template<Body BODY>
//    bool IncompleteInformationMCTS<BODY>::Trajectory::addMove() {
//        if(isLeaf()) return false;
//        int currentPlayerIndex = nextPlayerIndex();
//        Player &currentPlayer = players[currentPlayerIndex];
//        Player &otherPlayer = players[currentPlayerIndex^1];
//
//        typename TreeNode::QVector &currentPlayerQVec = currentPlayer.treeNode->qEntries[currentPlayer.body];
//        auto legalActs = currentPlayer.body.legalActs();
//        if(legalActs.count() > 0) {
//            action_type lastAct = currentPlayerQVec.getUTCAction(legalActs);
//            currentPlayer.qValues.push_back(currentPlayerQVec[lastAct]);
//            lastMessage = currentPlayer.body.actToMessage(lastAct);
//        }
//        double otherPlayerReward = otherPlayer.body.messageToReward(lastMessage);
//        otherPlayer.rewards.push_back(otherPlayerReward);
//        otherPlayer.treeNode = currentPlayer.treeNode->children[lastMessage];
//        return true;
//    }

//    template<Body BODY>
//    IncompleteInformationMCTS<BODY>::Trajectory::Trajectory(IncompleteInformationMCTS::TreeNode *rootNode, BODY player0body, BODY player1body) {
////        typename TreeNode::iterator firstMoverQEntry  = rootNode->randomEntry();
////        typename TreeNode::iterator secondMoverQEntry = rootNode->randomEntry();
//        players[0].body = std::move(player0body);
//        players[1].body = std::move(player1body);
//        players[0].treeNode = rootNode;
//        players[1].treeNode = nullptr;
////        action_type lastAct = players[0].act(); // opening move
////        players[0].qValues.push_back(&((firstMoverQEntry->second)[lastAct]));
////        lastMessage = players[0].body.actToMessage(lastAct);
////        players[0].treeNode = rootNode;
////        players[1].treeNode = rootNode->children[lastAct];
////        players[1].body.messageToReward(lastMessage); // ignore any initial reward
//
//    }

/**
 * Calculates the best action according to UCT. i.e. the action with maximum
 * meanQ + S*sqrt(ln(N))
 * where
 *  meanQ = the mean of all samples
 *  S = standard error in the mean
 *  N = the total number of samples
 *
 * @return the chosen act
 */
//    template<Body BODY>
//    IncompleteInformationMCTS<BODY>::action_type IncompleteInformationMCTS<BODY>::TreeNode::QVector::getUCTAction(const action_mask &legalMoves) {
//        double nStandardErrors = sqrt(log(totalSamples()));
//        int bestActId;
//        double bestQ = -std::numeric_limits<double>::infinity();
//        for (int actId = 0; actId < action_type::size; ++actId) {
//            if(legalMoves[actId]) {
//                const QValue &qVal = (*this)[actId];
//                if (qVal.sampleCount == 0) {
//                    bestActId = actId;
//                    break; // if there is an unexplored option, always explore.
//                }
//                double upperConfidenceQ = qVal.mean() + nStandardErrors * sqrt(qVal.standardVarianceInMean());
//                if (upperConfidenceQ >= bestQ) {
//                    bestQ = upperConfidenceQ;
//                    bestActId = actId;
//                }
//            }
//        }
//        assert(bestQ > -std::numeric_limits<double>::infinity()); // make sure we found an act
//        return static_cast<action_type>(bestActId);
//    }

}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
