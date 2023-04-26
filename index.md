Numerical politics is a new approach to the study of politics in which we perform numerical experiments on simulated societies in order to gain an understanding of organised, collective behaviour. This repo is intended to provide the software necessary to allow anyone to set up a numerical "laboratory" and start studying simulated societies. In these pages we'll describe numerical experiements and discuss how the practice of numerical politics can contribute to our understanding of how to effectively structure and govern real-world societies.

## The framework: thinking clearly about collective behaviour

The subject matter of numerical politics is the collective behaviour of many interacting agents. Agents interact by passing messages between eachother. Each agent has a number of ``channels'' for receiving messages. When an agent receives a message in one of its channels, it can respond by updating its internal state and/or sending yet more messages.

So, the behaviour of an agent can be defined as a set of probability distributions (one for each channel) $P(a|m,\psi)$ which gives the probability of an agent performing action $a$ in response to the receipt of message $m$ in the channel given that the agent's internal state is $\psi$. The action $a$ consists of a new internal state and/or a set of messages passed to other agent's channels at some time after receipt of the message.

If we also have a measure of individual wellbeing $$W(\psi)$$ based on the state, $$\psi$$, of each agent then we can define collective wellbeing as the sum of the wellbeing measures of all agents, $$\Omega = \sum_{\psi \in S} W(\psi)$$. Our particular interest here is to make predictive statements about collective wellbeing in terms of the behaviour of the agents, the wellbeing function and the states of the agents. Although we call $$W$$ "wellbeing", it is just a function that supplies us with an objective function. The assumption is that there is something we're trying to maximise, although it should be the subject of much debate exactly what this measure ought to be.

Notice that in this definition there is no mention of government. A goverment, if there is one, is encoded within the behaviours of the agents and is part of the model, as opposed to it being an exogenous actor imposing "interventions" on the agents. In this way, a government is best thought of as an emergent property of the agents' behaviours. We choose to make government endogenous because our interest here is *not* to simulate specific policy interventions but to understand the fundamental principles of organised, collective behaviour.

# Sugar and spice world

We begin with a very simple world where agents farm sugar and spice. However, some agents are better at farming sugar and others are better at farming spice. An agent's wellbeing is dependent on the amount of sugar/spice it eats per unit time but the tastiest (and/or healthiest) sugar/spice mix will generally not be aligned with the agent's ability to farm for its own consumption. So, collective wellbeing is maximised if the agents farm the crop they're good at farming, then trade sugar for spice and vice-versa.


## Sugar and spice world version 1

Suppose half the agents can farm only sugar and half can farm only spice, each at a rate of one unit per unit time. Suppose the agents randomly encouter eachother, whereupon each agent can either offer to trade or try to steal the other agent's crop. If both agents offer to trade then half the crop of each agent is swapped, however, if one agent offers to trade and the other tries to steal then the stealing agent gets half the others crop and the other is left with only half its original crop. If both try to steal then they are both unsuccessful and no food is transferred. This is the classic prisoner's dilemma, and can be summarised in the following agent wellbeing function

| Me  | Other | My wellbein|g |
| --- | --- | --- |
| trade | trade | 3 |
| trade | steal | 0 |
| steal | trade | 4 |
| steal | steal | 1 |

This world is simple enough that we can see immediately that the optimal collective wellbeing is when all agents trade. In this case, the average wellbeing of all agents is 3. However, under what circumstances will agents reach this optimum?

If we assume the agents are Q-learning then we can ask what kinds of society do the agents create for themselves using their learning. More mathematically we can look at the society as a dynamic system and ask about the distribution of wellbeing on the attractors. In the special case where the attractor is a point, we have a stable society where no amount of learning from further encounters will change any agent's policy.

### Zero memory agents

If agents have no memory of previous encounters then each encounter is a simple prisoner's dilemma situation. The state of a Q-learning agent is just the Q-values of trade, $$Q_t$$ and steal, $$Q_s$$. Equilibrium is when
$$
Q_t = 3*P(t) + r*\max(Q_t, Q_s)
$$
$$
Q_s = 4*P(t) + 1*P(s) + r*\max(Q_t, Q_s)
$$
but
$$
Q_s - Q_t = P(t) + P(s) = 1
$$
so $$Q_s > Q_t$$ irrespective of the other person's behaviour so a zero memory Q-learning agent will always learn to steal, leading to a society where all agents try to steal and every agent is worse off than a trading society.

So, memoryless Q-learning agents get stuck in an equilibrium that is far from optimal both collectively and individually.

What needs to change in order to improve these agent's lives?

### One step memory agents

If we give the agents the ability to remember the last encounter they had with another agent (if this isn't the first encounter) then the dynamics of the society gets much more interesting.

#### Experiment 1

We start with the simple case of a society of just two agents. Experiment 1 in the accompanying repo shows that there is more than one attractor in this world (i.e. the society is non-ergodic) and that all attractors are points in policy space. The society where both agents always try to steal from eachother is still an attractor, but their memory of the previous encounter allows other attractors.

The agents can also learn to both adopt the policy of always trying to steal unless we both traded last time. In this society the agents spend most of the time trying to steal from eachother, but if they both happen to explore the trade option at the same time, they will enter a period of sustained trading until one of them explores the steal option. So, average wellbeing is slightly better than the always-steal case, but still far from optimal.

Interestingly, the agents can learn to both adopt the following policy:

My last move | your last move | my move | human trait
| --- | --- | --- | --- |
| trade | trade |  trade |	 mutual-benefit |
| trade | steal |  steal |	 revenge |
| steal | trade |  steal |	 exploitation |
| steal | steal |  trade |	 ? |

The first three entries in the table above have analogues in human behaviour, but the last is a little unintuitive. If we both tried to steal from eachother last time, then this time I'll try to trade. This is key to the success of the policy as it means that whatever state the agents get into they quickly revert to mutual trading. As the exploration probability tends to zero, this society tends to the optimum of always trading, while remaining unexploitable (if I always try to steal from an agent with this policy, we'll flip between mutual stealing and me stealing from the agent, but my average wellbeing will be 2.5, less than if I take on the same policy as the other agent).

Note that the agents do not learn the tit-for-tat policy: if you tried to steal from me last time, I'll try to steal from you this time, but if you traded with me last time, I'll try to trade with you again. [what attractor does mutual tit-for-tat lead to?]

#### Experiment 2

Experiment 2 shows what happens as this society gorws.

