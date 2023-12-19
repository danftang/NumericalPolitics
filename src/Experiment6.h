// Question:
//   Under what conditions can agents form a shared language? i.e. attach a shared semantics to an arbitrary
//   set of symbols.
// Hypothesis:
//    - Pairs of Q-learning agents can form a language, but that language may not be shared (each agent
//      must learn the semantics of the other).
//    - Q-learning agents in a random-encounter society can form a shared language, and the existence of
//      an established society allows new-born agents to learn the shared language faster than in the
//      two-agent case.
//    - Pure Monte-Carlo tree search agents cannot reliably form a shared language in either paired or
//      social situations.
//    - Agents who have the ability to copy the behaviour of other agents can form a shared language faster
//      than Q-learning agents in both the two-agent and random-encounter society scenarios.
//
// Method:
//   Agents play a turns-based game of "guess the number". A first mover is chosen at random
//   and initialised with an number in 1...N, unknown to the other agent. The first mover must then pass
//   a symbol in a language, L. The second mover must then guess which number the first mover was given.
//   If the second mover guesses correctly both agents get reward 1, otherwise both get no reward.
//
// Discussion:
//   If there are N symbols in the language, there are N!^2 optimal 2-agent strategies (each agent can have one of N!
//   semantic interpretations of the language) but only N! of these are classed as languages (i.e. both agents share
//   the same semantics). Pure Monte-Carlo tree search would allow each agent to
//   find a joint optimal strategy, but two Monte-Carlo tree search agents wouldn't be able to align their strategies,
//   let alone find a shared langauge.
//   Q-learners would find a joint optimal strategy, but not necessarily a shared language.
//
// Created by daniel on 19/12/23.
//

#ifndef MULTIAGENTGOVERNMENT_EXPERIMENT6_H
#define MULTIAGENTGOVERNMENT_EXPERIMENT6_H

#endif //MULTIAGENTGOVERNMENT_EXPERIMENT6_H
