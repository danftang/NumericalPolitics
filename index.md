Numerical politics is a new approach to the study of politics in which we perform numerical experiments on simulated societies in order to gain an understanding of organised, collective behaviour. This repo is intended to provide the software necessary to allow anyone to set up a numerical "laboratory" and start studying simulated societies. In these pages we'll describe how to perform numerical experiements and discuss how the practice of numerical politics can contribute to our understanding of how to effectively govern real-world societies.

## The framework

The subject matter of numerical politics is a number of interacting agents whose probability of performing some action, $$a$$ at time $$t$$ can be given by $$P(a\|O,\psi)$$ where $$\psi$$ is some internal state of the agent, and $$O$$ are the agent's current observations of its environment. An action can modify the state of the acting agent, terminate the acting agent's existance or create some number of new agents. Without loss of generality we assume all agents have the same behaviour; heterogeneity among agents can be modelled by absorbing it into the state of the agent. Suppose we also have a measure of individual wellbeing $$W(\psi)$$ based on the state of each agent (agents that represent objects in the environment - e.g. chairs, tables etc. - can be given constatnt, zero wellbeing). Define collective wellbeing as the sum of the wellbeing measures of all agents, $$\Omega = \sum_\psi W(\psi)$$. The focus of numerical politics is to make predictive statements about collective wellbeing in terms of the behaviour of the agents, the wellbeing function and the states of the agents. Although we call $$W$$ "wellbeing", it is just a function that supplies us with an objective function. The assumption is that there is something we're trying to maximise, although it should be the subject of much debate exactly what this measure ought to be.

Notice that in this definition there is no mention of government. A goverment, if there is one, is encoded within the behaviours of the agents and is part of the model, as opposed to it being an exogenous actor imposing "interventions" on the agents. In this way, a government is best thought of as an emergent property of the agents' behaviours. We choose to make government endogenous because our interest here is *not* to simulate specific policy interventions but to understand the fundamental principles of organised, collective behaviour.

### Is reality contained within this framework?

In a trivial sense, reality is contained within this framework since we can arbitrarily carve up the quantum state of the world into separate "agents" whose state defines the quantum state within a certain volume of spacetime, whose "observations" consist of information about the neighbouring "agents" and whose behaviour encodes the dynamics of the quantum state.

Less trivially, most people would be willing to accept that the world is made up of people and objects (agents), that people act in ways that are influenced by their observations of the world around them and that objects act in ways defined by their physical properties and their interactions with other objects and with people.

## How can a numerical experiment tell us something about the real world?

The problem we face is that, on the one hand, the agent's state, $$\psi$$, needs to be sufficiently detailed to to give a believable wellbeing function, $$W$$. The detail captured in $$\psi$$ defines the detail that needs to be captured in agent actions $$a$$ and agent observations $$O$$; however, this makes, $$P(a\|O,\psi)$$, very difficult to define in a way that is provably correct for the level of detail contained in $$a$$, $$O$$ and $$\psi$$. That is to say, it's hard to model human behaviour.

In order to deal with this, we will develop inference techniques that do not require an exact specification of $$P$$. That is we wish to infer something about $$\Omega$$ given incomplete information about $$P$$, $$\psi$$, $$O$$, $$a$$ and $$W$$.

### Tractable societies

One approach to this is to develop "tractable societies", for which we can prove implications of the form

$$
\forall P,W: A(P,W) \rightarrow B(\Omega)
$$

where $$A$$ and $$B$$ are predicates.

Given this, if we are willing to accept that $$A$$ is true of reality, then we must also accept that $$B$$ is true of real collective wellbeing.

Notice that this doesn't require us to have full knowledge of $$P$$, only whether $$A(P,W)$$ is true or not. This is important because we are unlikely to ever have full knowledge of $$P$$ in reality. However, we do have a problem of semantics; in order to judge whether $$A$$ is true in reality, we need to relate the behaviour $$P$$ to the behaviour of real agents.



### Probabilistic inference

Given some obervations of the real world, $$o$$, we can use numerical inference techniques to put a posterior probability distribution on $$P$$ and on the state of the world at some time $$\psi_t$$. Given this, we can also put a posterior probability distribution on the rate of change of $$\Omega$$ at time $$t$$. From here, we'd be interested in calculating the change in this distribution with changes in behaviour, $$P$$, so we know what behavioural changes would improve the expectation of collective wellbeing.

However, in full generality, doing this is computationally intractable so we'll need to make some approximations. Possible approximations include:

* Abstraction over the number of agents.
* Abstraction over agent state
* Abstraction over acts
* Abstraction over observations

The key to numerical politics is to find the correct abstraction given the data (and the data itself can only be given meaning in relation to an abstraction, so the abstractions we make are nothing more or less than abstractions of the semantics of the data and/or the semantics of the model).
