# RL Properties

## General 
- Non-deterministic (Uncertainty because actions might result in multiple successor states)
- Sequential decision making, decisions affect future decisions
- Difference to other ML: rewards as delayed feedback, time matters, actions affect data/outcomes
- Experience replay:   
  - Incremental methods have strongly correlated updates that break the i.i.d. assumptions, are not sample efficient and may rapidly forget rare experiences that would be useful later on
  - solution: Reuse experience/data and replay them --> Store experience in an experience replay buffer with all experiences are equally important or important transitions/samples more frequently

## MDP
- full MDP: (s, a, T(s,a,s'), R(s,a,s'))
  - s: states
  - a: actions
  - T(s,a,s'): probability that state s’ is reached, if action a is executed in state s (sometimes also, as P(s’|s,a))
  - R(s,a,s'): reward at each step
- For MDPs, “Markov” means action outcomes depend only on the current state
- Utility = sum of rewards
- policy π: S-->A defines the agent’s action for each state: π(s)=a (deterministic) or π(s)=p(a) (probability distribution, Non-deterministic/Stochastic)
- optimal policy π* (highest expected utility)
- utility of agent = sum of discounted rewards for environment history $U[s_o, s_1, …] = R(s_o) + γR(s_1) + γ^2*R(s_2) + …+ γ^n*R(s_n)$
- Solve an MDP:
  - value function: 
    - V(s) = value/utility of a state
    - V*(s) = expected value/utility starting in s, optimally onwards = $max_a Q^*(s,a)$
    - $V_\pi(s)= E_\pi[R_{t+1}+\gamma^1R_{t+2}+...]$
  - Q-state: (s,a) being in state s and having taken action a
    - $Q^*(s,a) = \sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V^*(s')]$ expected value/utility starting with a from state s, optimally onwards
    - $π^*(s) = argmax _aQ^*(s,a) = argmax_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V^*(s')]$ (optimal action from state s)
  - keeps a look-up table of all states
  - value Iteration algorithm:
    - start with arbitrary initial values for the utilities
    - each iteration k+1, for all states s:
      - Update $V_{k+1}(s)$ based on $V_k(s)$ --> Bellman update
        - $V_{k+1}(s) \leftarrow max_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V_k(s')]$
    - Repeat until convergence
    - Extract policy $\pi ^*(s) = argmax_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V_k(s')] (= argmax _aQ^*(s,a) )$ 
  - Problems with value-iteration: try to find optimal values --> value convergence takes too long
  - Policy iteration algorithm: 
    - policy evaluation --> calculate values/utilities for fixed policy π  with recursion until convergence
      - $V_0^\pi (s) = 0$
      - $V^\pi _{k+1} (s) \leftarrow \sum_{s'}T(s,\pi (s),s')[R(s,\pi (s),s')+\gamma V^\pi _k(s')]$
      - Fully shares information between states, but requires T and R! 
    - policy extraction --> update/extract policy using one-step look-ahead with resulting converged utilities as future values
      - $\pi _{i+1}(s) = argmax_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V^{\pi _i}(s')]$ 

## Reinforcement Learning
- Don't have T and R anymore like with MDPs
- Solve MDP problems when we don't now MDP
- Online vs Offline/Batch learning
  - Learn while exploring the world, or learn from fixed batch of data
- Active vs. Passive Learning
  - Does the learner actively choose actions to gather experience? or, is a fixed policy provided?
  
- Model-based vs. Model-free Learning
  - Do we estimate T(s,a,s’) and R(s,a,s’), or just learn values/policy directly?

### Model-based learning

- Learn an approximate model of T, R based on experiences/data and solve the MDP based on the learned T, R
- keeps a look-up table of all states
- Step 1: learn MDP model 
  - Count outcomes s’ for each q-state (s, a)
  - Normalize to give an estimate of the learned T(s,a,s')
  - Discover each  R(s,a,s') estimate when we experience (s,a,s’)
- Step 2: Solve learned MDP 
  - Value or policy iteration approach 
- Pros: Very simple and intuitive, Remarkably effective
- Cons: Sufficient (training) experience is required, Maintaining all these counts is expensive

### Direct evaluation learning 
- offline/batch, model-free and passive 
- keeps a look-up table of all states
- Monte-Carlo policy evaluation with experience from full episodes/ monte carlo sampling
- Goal: Compute values for each state under a given policy π
- Algorithm:
  - Step 1: Act according to π
  - Step 2: Every time you visit a state s, keep track of its utility (sum of discounted rewards) as well as of the number of visits 
      - $U[s_o, s_1, …] = R(s_o) + γR(s_1) + γ^2*R(s_2) + …+ γ^n*R(s_n)$
  - Step 3: Average those samples to get the estimated value of s by summing up the utilities for each state and divide by number of visits
- Pros: Easy to understand, Doesn’t require any knowledge of T, R, It eventually computes the correct average values, using just sample transitions
- Cons: takes a long time to learn; Each state must be learned separately --> No share of information between states, it wastes information about state connections (transitions between states); The method needs (complete) episodes (Monte Carlo sampling)

### Temporal difference (TD) learning

- online, model-free and passive
- keeps a look-up table of all states
- on-policy learning following a policy and not use information from policy dependent sampling of value function immediatly to improve policy
- Shares information between states by updating V(s) each time we get a new sample/transition
- Likely outcomes s’ will contribute updates more often
- algorithm
  - Step 1: Estimate value of V(s) based on sample: (s, a, s’, r)
    - $sample = R(s,\pi(s),s')+\gamma V^\pi(s')$
  - Step 2: Update: Move current value estimate V(s) values toward value of whatever successor occurs (s’)
    - $V^\pi(s) \leftarrow (1- \alpha) V^\pi(s) + (\alpha)sample$
    - learning rate $\alpha$ controls how to combine our current estimate with the new sampled estimate, 0≤$\alpha$≤ 1
    - older samples contribute exponentially less to $V^π(s)$, which is great since these samples are based on old (hence worse) versions of $V^π(s)$

### TD Q-leaning
- just replace Step 2 updating values of states by updating q-values of states:
- $sample = R(s,a,s') + \gamma max_{a'}Q(s',a') $
- $Q(s,a) \leftarrow (1- \alpha) Q(s,a) + (\alpha)[sample]$
- easy converting to policy: $π^*(s) = argmax _aQ^*(s,a) $

### Q-learning

- model-free and active 
- keeps a look-up table of all states
- off-policy: converges to optimal policy, even if
acting suboptimally
- Exploitation: Make best decision given current information
- Exploration: Gather more information that might lead us to better decisions in the future
-ε-greedy policies
    - Explore with probability ε a random action a and exploit with probability 1-ε, 0≤ε ≤1 using current policy and $a = argmax_a Q(s,a)$
    - Problem: You do eventually explore the space, but keep thrashing around once learning is done
    - Solutions:
      - lower ε over time
      - exploration functions
- exploration functions:
  - explore areas whose badness is not (yet) established, eventually stop exploring
  - Q-value considers frequency of visiting a particular state
  - $sample = R(s,a,s') + \gamma max_{a'}f(s',a') $
  - $Q(s,a) \leftarrow (1- \alpha) Q(s,a) + (\alpha)[sample]$
  - exploration function: $f(s,a) = Q(s,a) + \frac{k}{N(s,a)}$ with N(s,a) frequency of visiting a particular (s,a) state and k predefined value
- compare ε-greedy and exploration function by regret, which measures your total mistake cost
  - difference between the cumulative reward of the optimal policy and that gathered by π


### Approximate Q-learning
- In realistic situations, we cannot possibly learn about every single state
- Estimate value function with value function approximation and generalize from seen states to unseen states
- Feature-based state representation
  - hand-crafted features (traditional ML)
    - Distance to closest ghost, Distance to closest dot,...
    - For q-states: e.g. action moves closer to food
  - learned features (deep learning)
    - Very useful for e.g., images, text etc
    - E.g., CNNs extract most useful features for classifying images
    - Approaches:
      - States s to v-values, Q-states (s,a) to q-values, States s to q-values
- Function approximator to estimate the action-value function
  - Typically differentiable function approximators like linear combination of features and NNs
- Challenges: Data is no-stationary and not i.i.d.
- Sample estimate: $ sample = R(s,a,s') + \gamma max_{a'}Q(s',a') $
- Approximate Q-learning: 
  - $w_i \leftarrow w_i + \alpha (sample-Q(s,a)) f_i(s,a)$
  - $Q(s,a) = w_1f_1(s,a)+...+w_nf_n(s,a)$
  - Tune weights of features instead of maintaining values