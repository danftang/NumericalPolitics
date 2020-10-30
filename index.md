Numerical politics is a new approach to the study of politics in which we perform numerical experiments on simulated societies in order to gain an understanding of organised, collective behaviour. This repo is intended to provide the software necessary to allow anyone to set up a numerical "laboratory" and start studying simulated societies to this end. In these pages we'll describe how to perform numerical experiements and discuss how the practice of numerical politics can contribute to our understanding of how to effectively govern real-world societies.

## The framework

The subject matter of numerical politics is a number of interacting agents whose behaviour is defined by a distribution $$P(a\|O,\psi)$$ that gives the probability that an agent in state $$\psi$$, having made observations $$O$$ of its environment, will perform action $$a$$. An action can modify the state of the acting agent, terminate the acting agent's existance or create a new agent. Without loss of generality we assume all agents have the same behaviour; heterogeneity among agents can be modelled by absorbing it into the state of the agent. Suppose we also have a measure of individual wellbeing $$W(\psi)$$ based on the state of each agent (agents that represent objects in the environment - e.g. chairs, tables etc. - can be given constatnt, zero wellbeing). Define collective wellbeing as the sum of the wellbeing measures of all agents, $$\Omega = \sum_\psi W(\psi)$$. The focus of numerical politics is to make predictive statements about collective wellbeing in terms of the behaviour of the agents, the wellbeing function and the states of the agents.

Notice that in this definition there is no mention of government. A goverment, if there is one, is encoded within the behaviours of the agents and is part of the model, as opposed to it being an exogenous actor imposing "interventions" on the agents. In this way, a government is best thought of as an emergent property of the agents' behaviours. We choose to make government endogenous because our interest here is *not* to simulate specific potential real-world policies but to understand the fundamental principles of organised, collective behaviour.

### Is reality contained within this framework?

In a trivial sense, reality is contained within this framework since we can arbitrarily carve up the quantum state of the world into separate "agents" whose state defines the quantum state within a certain volume of spacetime, whose "observations" consist of information about the neighbouring "agents" and whose behaviour encodes the dynamics of the quantum state.

Less trivially, most people would be willing to accept that the world is made up of people and objects (agents), that people act in ways that are influenced by their observations of the world around them and that objects act in ways defined by their physical properties and their interactions with other objects and with people.

## How can a numerical experiment tell us something about the real world?

There are many ways that a numerical experiment can give us knowledge to make predictive statements about the real world. 

### Categorical inference

We may be able to use numerical inference to prove an implication of the form
$$
A(P,W) \rightarrow B(\Omega) 
$$
where $$A$$ and $$B$$ are predicates, $$P$$ is agent behaviour, $$W$$ is agent wellbeing and $$\Omega$$ is collective wellbeing. Given this, if we are willing to accept that $$A$$ is true of reality, then we must also accept that $$B$$ is true of collective wellbeing. Also, if $$A$$ is not true of reality, and nor is $$B$$ of $$\Omega$$ it means that we can bring about $$B(\Omega)$$ by changing behaviour such that $$A(P,W)$$ is true.

Notice that this doesn't require us to have full knowledge of $$P$$, only whether $$A(P,W)$$ is true or not. This is important because we are unlikely to ever have full knowledge of $$P$$ in reality.

### Probabilistic inference

Given some obervations of the real world, $$o$$, we can use numerical inference techniques to put a posterior probability distribution on $$P$$ and on the state of the world at some time $$\psi_t$$. Given this, we can also put a posterior probability distribution on the rate of change of $$\Omega$$ at time $t$.
