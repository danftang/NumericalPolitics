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

namespace abm::minds {
/**
 *
 * @tparam BODY The body with which we should play out in order to build this tree
 */
    template<Body BODY>
    class IncompleteInformationMCTS {
    public:

        typedef BODY::action_type action_type;
        typedef BODY::message_type message_type;
        typedef BODY::action_mask action_mask;
        typedef BODY observation_type;



        inline static std::default_random_engine randomGenerator = std::default_random_engine();

        /** A single node in the tree. Represents Q-values of all hidden states with a given observable history.
         *  For each hidden state, there is an associated "QEntry" (a map<BODY,QVector>::iterator) which
         *  identifies the body state and the Q-values for all acts from this state.
         **/
        class TreeNode {
        public:

            /** A QValue stores statistical information about the Q-Value of a single (state,action) pair */
            class QValue {
            public:
                double  sumOfQ      = 0.0; // Sum of Q-values of all samples so far by action
                double  sumOfQSq    = 0.0; // Sum of squares of Q-values of all samples so far by action
                uint    sampleCount = 0;   // number of samples by action

                void addSample(double cumulativeReward);
                double mean() const { return  sumOfQ / sampleCount; };
                double standardVarianceInMean() const { return(sumOfQSq - pow(sumOfQ, 2))/pow(sampleCount, 3); }
            };


            /** A QVector stores QValues for all acts from a single state */
            class QVector: public std::array<QValue, action_type::size> {
            public:
                int         totalSamples();
                action_type getUCTAction(const action_mask &legalMoves);
            };


            typedef std::map<BODY, QVector>::iterator iterator;

            std::map<BODY, QVector> qEntries;   // QVectors by agent state for this node
            std::map<BODY,uint>     otherPlayerDistribution; // counts of other player states by body state
            std::array<TreeNode *, message_type::size> children;   // ...indexed by actId. nullptr if child not present.

            TreeNode() { children.fill(nullptr); }
            ~TreeNode() { for (TreeNode *child: children) delete (child); }

            TreeNode *getChildOrCreate(int actId);
            iterator addQEntry(const BODY &agent);
            iterator randomEntry();
            std::function<const BODY &()> createStateSampler();
        };


        /** Represents a path through the tree. This includes, in addition to the
         * tree nodes, a record of which body states (qEntries) within each node were visited,
         * the rewards that were received and a final act beyond the endNode.
         */
        class Trajectory {
        public:
\
            class SelfPlayBody {
            public:
                TreeNode *                                  treeNode;   // current treeNodes for player's experience
                BODY                                        body;       // current state of player

                message_type actToMessage(action_type act) {
                    message_type message = body.actToMessage(act);
                    if(treeNode != nullptr) {
                        treeNode = treeNode->children[message];
                        if(treeNode == nullptr) {
                            // TODO: add child and increment other player counter
                        } else {
                            treeNode->otherPlayerDistribution[body]++;
                        }
                    }
                    return message;
                }

                double messageToReward(message_type incomingMessage) {
                    if(treeNode != nullptr) {
                        treeNode = treeNode->children[incomingMessage];
                        assert(treeNode != nullptr); // already generated by other player
                    }
                    return body.messageToReward(incomingMessage);
                }
            };

            class SelfPlayMind {
            public:
                typedef SelfPlayBody observation_type;

                std::vector<typename TreeNode::QValue *>    qValues; // Q values at choice points of the player
                std::vector<double>                         rewards; // reward between choice points of the player

                action_type act(const observation_type &body, action_mask legalActs, double rewardFromLastAct) {
                    typename TreeNode::QVector qVec = body.treeNode->qEntries[body.body];
                    action_type action = qVec.getUTCAction(legalActs);
                    rewards.push_back(rewardFromLastAct);
                    qValues.push_back(qVec[action]);
                    return action;
                }

                void endEpisode(double rewardFromFinalAct) {
                    // TODO: do backprop
                }
            };


            // TODO: make the player an agent (or perhaps a Mind)
            class Player {
            public:
                TreeNode *                                  treeNode; // current treeNodes for player's experience
                std::vector<typename TreeNode::QValue *>    qValues; // Q values at choice points of the player
                std::vector<double>                         rewards; // reward between choice points of the player
                BODY                                        body; // current state of player

                action_type chooseAct();
            };

            Player                                      players[2]; // current state of players
            message_type                                lastMessage;    // needed for leaf conditions

            explicit Trajectory(TreeNode *rootNode, BODY player0body, BODY player1body);

            int nextPlayerIndex() { return players[0].rewards.size() == players[1].rewards.size(); }
            bool addMove();
            void expandTreeOneMove();
            // Selects a leaf node to expand by traversing from root using the Q values
            void moveToLeaf() { while (addMove()); }
            bool isLeaf();
            void playOut(double discount);
            void backProp(double discount);
        };

        TreeNode *rootNode;
        std::function<const BODY &()> rootNodeSampler;      // sampler for the current root node (player to move next, state before decision)
        std::function<const BODY &()> otherPlayerSampler;   // sampler for the other player, state just before incoming message is received

        explicit IncompleteInformationMCTS(std::function<BODY()> priorAgentSampler);
        ~IncompleteInformationMCTS() { delete (rootNode); }


        // ----- Mind interface ------
        action_type act(message_type observation, action_mask actMask);
//            { mind.train(observation, act, 0.0, observation, true) }; // TODO: send reward only? More flexible? or add messages? or be more flexible with observation?

    protected:
        void buildTree(int nSamples, double discount);
        void shiftRoot(message_type message);

    };

    /**
     * @return A function that returns a sample from the body states with a probability
     * proportional to the total number of samples in each qEntry of this node.
     */
    template<Body BODY>
    std::function<const BODY &()> IncompleteInformationMCTS<BODY>::TreeNode::createStateSampler() {
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

    /** When either agent actually moves, we need to leap-frog the root node for that agent
     * and delete tree nodes that above the new root.
     *
     * @param message
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::shiftRoot(message_type message) {
        TreeNode *newRoot = rootNode->children[message];
        assert(newRoot != nullptr);
        rootNode->children[message] == nullptr;
        delete(rootNode);
        rootNode = newRoot;
        otherPlayerSampler = std::move(rootNodeSampler); // need to save half-step states on play-out! (or only allow body state change on message receipt)
        // TODO: is it reasonable to say message/response defines body state change?
        //   if not, then there is a hidden change, this is perfectly possible.
        //   So, is it easy to sample from the possible hidden transitions given the message/response?
        //   not necessarily easy.
        rootNodeSampler = newRoot->createStateSampler();
    }

    /** Implementation of Mind interface for the whole tree.
     *
     * @param incomingMessage
     * @param legalActs
     * @return
     */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::action_type
    IncompleteInformationMCTS<BODY>::act(message_type incomingMessage, action_mask legalActs) {

    }


    /** Build a tree from the
     *
     * @param nSamples
     * @param discount
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::buildTree(int nSamples, double discount) {
        for (int s = 0; s < nSamples; ++s) {
            Trajectory sample(rootNode);
            sample.moveToLeaf();
            sample.expandTreeOneMove();
            sample.playOut(discount);
            sample.backProp(discount);
        }
    }

    template<Body BODY>
    IncompleteInformationMCTS<BODY>::IncompleteInformationMCTS(std::function<const BODY &()> priorAgentSampler) :
            rootNode(new TreeNode()),
            rootNodeSampler(priorAgentSampler),
            otherPlayerSampler(priorAgentSampler) {

            }


    /**
     *
     * @tparam BODY
     * @return
     */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::action_type IncompleteInformationMCTS<BODY>::Trajectory::Player::chooseAct() {
        assert(treeNode != nullptr);
        assert(treeNode->qEntries.contains(body));
        typename TreeNode::QVector &qVec = treeNode->qEntries[body];
        auto legalActs = body.legalActs();
        action_type act = qVec.getUTCAction(body.legalActs());
        qValues.push_back(qVec[act]);
        return act;
    }

    /** Update each qEntry in this trajectory by incrementing the sample count and updating the Qvalue
     * in-line with the cumulative reward to the end of the episode.
     * Deletes the values in rewards as it goes.
     *
     * @param discount
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::Trajectory::backProp(double discount) {
        for(int player = 0; player < 2; ++player) {
            Player &currentPlayer = players[player];
            double cumulativeReward = 0.0;
            while(currentPlayer.rewards.size() > 0) {
                cumulativeReward = currentPlayer.rewards.back() + discount * cumulativeReward;
                currentPlayer.qValues.back().addSample(cumulativeReward);
                currentPlayer.rewards.pop_back();
                currentPlayer.qValues.pop_back();
            }
        }
    }

    /** Play out from the current trajectory state to the end of the episode
     * using random moves,
     * updates the final entries of the rewards for each player to be the cumulative
     * rewards to the end of the episode.
     *
     * @param discount
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::Trajectory::playOut(double discount) {
//        std::uniform_int_distribution<int> uniformActDist(0, BODY::Action::size - 1);

        BODY playerBodies[2] = {this->players[0].body, this->players[1].body};
        double cumulativeRewards[2] = {0.0, 0.0};
        double totalDiscount[2] = {1.0,1.0};
        int currentPlayerIndex = nextPlayerIndex();
        typename BODY::message_type message;
        do {
            // current player acts and generates message
            BODY &currentPlayer = playerBodies[currentPlayerIndex];
            const std::bitset<action_type::size> &legalActs = currentPlayer.legalActs();
            if(legalActs.count() > 0) {
                int nextAct = GreedyPolicy::sampleUniformly(legalActs);
                message = currentPlayer.actToMessage(nextAct);
            } else {
                message = BODY::message_type::close;
            }

            // deliver message to other player and record reward
            currentPlayerIndex = currentPlayerIndex ^ 1;
            BODY &otherPlayer = playerBodies[currentPlayerIndex];
            cumulativeRewards[currentPlayerIndex] = otherPlayer.messageToReward(message) * totalDiscount[currentPlayerIndex];
            totalDiscount[currentPlayerIndex] *= discount;

        } while (message != BODY::message_type::close);
        players[0].rewards.push_back(cumulativeRewards[0]);
        players[1].rewards.push_back(cumulativeRewards[1]);
    }

    template<Body BODY>
    bool IncompleteInformationMCTS<BODY>::Trajectory::isLeaf() {
        const Player & nextPlayer = players[nextPlayerIndex()];
        return nextPlayer.treeNode == nullptr || !nextPlayer.treeNode->qEntries.contains(nextPlayer.body);
    }

    /** Adds a new QEntry for a state that isn't currently stored.
     * This may or may not require a new TreeNode to be created.
     * Assumes we're currently at a leaf. i.e.
     */
    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::Trajectory::expandTreeOneMove() {
        assert(isLeaf());
        int currentPlayerIndex = nextPlayerIndex();
        Player &currentPlayer = players[currentPlayerIndex];
        Player &otherPlayer = players[currentPlayerIndex^1];

        if(currentPlayer.treeNode == nullptr) {
            currentPlayer.treeNode = new TreeNode();
            otherPlayer.treeNode->children[lastMessage] = currentPlayer.treeNode;
        }
        currentPlayer.treeNode->qEntries.try_emplace(currentPlayer.body);
    }

    /** The next player chooses his move and the trajectory moves forward to the
     * next player's choice point.
     *
     * @return true if were NOT at a leaf.
     */
    template<Body BODY>
    bool IncompleteInformationMCTS<BODY>::Trajectory::addMove() {
        if(isLeaf()) return false;
        int currentPlayerIndex = nextPlayerIndex();
        Player &currentPlayer = players[currentPlayerIndex];
        Player &otherPlayer = players[currentPlayerIndex^1];

        typename TreeNode::QVector &currentPlayerQVec = currentPlayer.treeNode->qEntries[currentPlayer.body];
        auto legalActs = currentPlayer.body.legalActs();
        if(legalActs.count() > 0) {
            action_type lastAct = currentPlayerQVec.getUTCAction(legalActs);
            currentPlayer.qValues.push_back(currentPlayerQVec[lastAct]);
            lastMessage = currentPlayer.body.actToMessage(lastAct);
        }
        double otherPlayerReward = otherPlayer.body.messageToReward(lastMessage);
        otherPlayer.rewards.push_back(otherPlayerReward);
        otherPlayer.treeNode = currentPlayer.treeNode->children[lastMessage];
        return true;
    }

    template<Body BODY>
    IncompleteInformationMCTS<BODY>::Trajectory::Trajectory(IncompleteInformationMCTS::TreeNode *rootNode, BODY player0body, BODY player1body) {
//        typename TreeNode::iterator firstMoverQEntry  = rootNode->randomEntry();
//        typename TreeNode::iterator secondMoverQEntry = rootNode->randomEntry();
        players[0].body = std::move(player0body);
        players[1].body = std::move(player1body);
        players[0].treeNode = rootNode;
        players[1].treeNode = nullptr;
//        action_type lastAct = players[0].act(); // opening move
//        players[0].qValues.push_back(&((firstMoverQEntry->second)[lastAct]));
//        lastMessage = players[0].body.actToMessage(lastAct);
//        players[0].treeNode = rootNode;
//        players[1].treeNode = rootNode->children[lastAct];
//        players[1].body.messageToReward(lastMessage); // ignore any initial reward

    }

    /** Choose a random body state from this node
    * @return iterator into the body->QValue map.
    */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::TreeNode::iterator IncompleteInformationMCTS<BODY>::TreeNode::randomEntry() {
        auto uniformDist = std::uniform_int_distribution<int>(0, qEntries.size() - 1);
        auto it = qEntries.begin();
        std::advance(it, uniformDist(randomGenerator));
        return it;
    }

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

    template<Body BODY>
    int IncompleteInformationMCTS<BODY>::TreeNode::QVector::totalSamples() {
        return std::reduce(begin(*this), end(*this), QValue(), [](const QValue &a, const QValue &b) {
            return a.sampleCount + b.sampleCount;
        });
    }

    /** Returns the child of this node corresponding to the given act,
    * creating a new one if necessary
    *
    * @param act the act that identifies the child
    * @return a pointer to the child
    */
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::TreeNode *IncompleteInformationMCTS<BODY>::TreeNode::getChildOrCreate(int actId) {
        TreeNode *childNode = children[actId];
        if (childNode == nullptr) {
            childNode = new TreeNode(this, actId);
            children[actId] = childNode;
        }
        assert(childNode->lastAct == actId);
        return childNode;
    }

    template<Body BODY>
    void IncompleteInformationMCTS<BODY>::TreeNode::QValue::addSample(double cumulativeReward) {
        sumOfQ += cumulativeReward;
        sumOfQSq += cumulativeReward * cumulativeReward;
        ++sampleCount;
    }

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
    template<Body BODY>
    IncompleteInformationMCTS<BODY>::action_type IncompleteInformationMCTS<BODY>::TreeNode::QVector::getUCTAction(const action_mask &legalMoves) {
        double nStandardErrors = sqrt(log(totalSamples()));
        int bestActId;
        double bestQ = -std::numeric_limits<double>::infinity();
        for (int actId = 0; actId < action_type::size; ++actId) {
            if(legalMoves[actId]) {
                const QValue &qVal = (*this)[actId];
                if (qVal.sampleCount == 0) {
                    bestActId = actId;
                    break; // if there is an unexplored option, always explore.
                }
                double upperConfidenceQ = qVal.mean() + nStandardErrors * sqrt(qVal.standardVarianceInMean());
                if (upperConfidenceQ >= bestQ) {
                    bestQ = upperConfidenceQ;
                    bestActId = actId;
                }
            }
        }
        assert(bestQ > -std::numeric_limits<double>::infinity()); // make sure we found an act
        return static_cast<action_type>(bestActId);
    }

}
#endif //MULTIAGENTGOVERNMENT_INCOMPLETEINFORMATIONMCTS_H
