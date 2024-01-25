//  A Deterministic Particle Monte-Carlo Tree Search consists of the following:
//      - A Q-approximator (neural net) going from self/other body-state pairs to Q-vectors [the quality
//      also depends on episode history as this defines other's belief in my state, also my belief about other's
//      state may be wrong...however, a Q-vector is well defined for any amount of information, it's simply the
//      expectation of the posterior over reward given that information...so...the key is capturing the most
//      important information. We can consider a Q-approximator that takes a self/other body-state pair and an
//      episode history - although in this case, what exactly does other body-state bring us? ].
//      - A set of particles representing the distribution of other-state given the current episode history
//      - A set of particles representing (other's belief in) self-state given the current episode history
//      - (Notionally) A set of Monte-Carlo search trees over self/other body states, one for each self/other pair
//          from the self and other particles.
//  The algorithm assumes that Agent state transition is deterministic given a message. Given a set of self/other
//  particles, we can express the Q-vectors for each particle as a set of simultaneous equations involving
//  perfect information Q-vectors.
//
//  What if the off-tree Q-network takes a self/other body-state pair, and generates an expected Q-vector?
//  -----------------------------------------------------------------------------------------------------------
//  Given a known self-body state, a set of (possibly weighted) samples of our belief over other's body state,
//  and a Q-function over body-state pairs, we can caluclate the expected Q-vector as a (weighted) average over
//  all possible body state pairs (given other's belief about my state).
//  We can build a Monte-Carlo tree over the off-tree Q-function in the normal way, and this becomes an improved
//  Q-function over body-state pairs which we can use to train the off-tree function [but this must either marginalise
//  over episode histories, or assume complete knowledge. However, if we assume a deterministic state transition given
//  a message, then perfect knowledge at time t implies perfect knowledge at time t+1].
//  Given the off-tree Q-function, a Q-policy, a distribution over other's body state and a distribution
//  over (other's belief about) self-state (derivable from episode messages) we can define the probability of observing
//  an incoming message as the weighted sum of probabilities of each other-state. This can be differentiated w.r.t.
//  the Q-function for training.
//  So, we can define the loss function as the RMS loss over a set of I/O pairs of the off-tree Q-function taken from the
//  root of each MCTS lookup, plus the negative log probability of observing incoming messages given a self/other body
//  state distribution. This can be unified as a negative log probability of observing a given episode history,
//  given a self start-state (and a prior over start state pairs). These will come in two flavours depending on
//  whether self is first mover or not (although maybe not worth doing this unification).
//
// Created by daniel on 27/12/23.
//

#ifndef PARTICLEMCTS_H
#define PARTICLEMCTS_H



class ParticleMCTS {
public:

};



#endif //PARTICLEMCTS_H
