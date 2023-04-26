## Model semantics

Given a model, which consists of a policy $$\pi$$, an agent wellbeing $$W$$ and a probability distribution over states,
[should we have universal probabilistic agents with policy also absorbed into state? We can then work entirely in a Fock space, and consider the dynamic system within this phase space.]

### The power of ideas: The politics of information

If it is the case that the agents are bounded rational, then **understanding** becomes an intervention. By making some subset of agents aware of some information this would change their behaviour, which would change the dynamics of the system.

## How can a numerical experiment tell us something about the real world?

The problem we face is that, on the one hand, the agent's state, $$\psi$$, needs to be sufficiently detailed to to give a believable wellbeing function, $$W$$. The detail captured in $$\psi$$ defines the detail that needs to be captured in agent actions $$a$$ and agent observations $$O$$; however, this makes, $$P(a\|O,\psi)$$, very difficult to define in a way that is provably correct for the level of detail contained in $$a$$, $$O$$ and $$\psi$$. That is to say, it's hard to model human behaviour in enough detail to have a deterministic wellbeing function.

One way to deal with this is to abstract the model and make the wellbeing function probabilistic (given the abstracted state), but abstracting the state has the effect of loss of information and widening of the dynamics...so the probability of the abstract state given the abstract model is not the probability of the abstract state given the correct model.

[Ordinary model abstraction is equivalent to making the abstract state a distribution over partitions of the concrete domain. This introduces widening by introducing states in partitions that properly have zero probability. Abstract domains from static analysis can probably improve on this.]

[Perhaps what we're doing here is saying that the abstraction in which we define the model dynamics is different from the abstraction in which we define the wellbeing function. Our contribution is to explain how to do this and to provide tools. However, we're also allowing uncertainty with respect to the dynamics of the model. By providing a probability distribution over actions given a state/observation then we're expressing oru uncertainty over model dynamics. Under this interpretation, we cannot assume that the distribution over acts is independent of previous decisions to act, given the current state, so it's not correct to simply draw from the distribution at each timestep, we must make a more subtle inference.

We can express all this as different kinds of knowledge about the system. The aim is to make inferences given these disparate pieces of knowledge.]

In order to deal with this, we will develop inference techniques that do not require an exact specification of $$P$$. That is we wish to infer something about $$\Omega$$ given incomplete information about $$P$$, $$\psi$$, $$O$$, $$a$$ and $$W$$.

### Tractable societies

One approach to this is to develop "tractable societies", for which we can prove implications of the form

$$
\forall P,W: A(P,W) \rightarrow B(\Omega)
$$

where $$A$$ and $$B$$ are predicates.

Given this, if we are willing to accept that $$A$$ is true of reality then we must also accept that $$B$$ is true of real collective wellbeing.

Notice that this doesn't require us to have full knowledge of $$P$$, only whether $$A(P,W)$$ is true or not. This is important because we are unlikely to ever have full knowledge of $$P$$ in reality. However, we do have a problem of semantics; in order to judge whether $$A$$ is true in reality, we need to relate the behaviour $$P$$ to the behaviour of real agents.



### Probabilistic inference

Given some obervations of the real world, $$o$$, we can use numerical inference techniques to put a posterior probability distribution on $$P$$ and on the state of the world at some time $$\psi_t$$. Given this, we can also put a posterior probability distribution on the rate of change of $$\Omega$$ at time $$t$$. From here, we'd be interested in calculating the change in this distribution with changes in behaviour, $$P$$, so we know what behavioural changes would improve the expectation of collective wellbeing.

However, in full generality, doing this is computationally intractable so we'll need to make some approximations. Possible approximations include:

* Abstraction over the number of agents.
* Abstraction over agent state
* Abstraction over acts
* Abstraction over observations

The key to numerical politics is to find the correct abstraction given the data (and the data itself can only be given meaning in relation to an abstraction, so the abstractions we make are nothing more or less than abstractions of the semantics of the data and/or the semantics of the model).
