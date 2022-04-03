# Study Questions
## In general:
1. When does LOF characterize a point as an outlier?
        - outliers have much lower density than its neighbors and are points with the largest LOF values
        - LOF > 1: point is an outlier
2. How does dimensionality affect inconsistency?
        - the lower the dimensionality on a subspace, the higher the inconsistency due to monotonicity and more inconsistent data samples to consider
3. How can we mitigate overfitting of decision trees?
     - Prepruning: Stop growing tree, when it does not provide much information
          - techniques: minimum number of instances per node threshold or statistical significance test (chi-squared, infogain)
      - Postpruning: Discard subtrees that do not contribute much to classification
4. The V-value of a state equals?
    - V*(s) = expected utility starting in s and acting optimally from that point onwards  = $max_a Q^*(s,a)$
    - $V_\pi(s) =$ expected value of the state =  $E_\pi[R_{t+1}+\gamma^1R_{t+2}+...]$
5. The Q-value of a state equals?
    - Q*(s,a) = expected utility starting out having taken action a from state s and acting optimally from that point onwards = $\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V^*(s')]$ 
6. The policy of a state equals?
    - π*(s) = $argmax_a Q^*(s,a)$
7. What does Markov wrt. MDPs mean?
   - action outcomes depend only on the current state
8. Why do we discount when calculating the utility of an agent?
   - Sooner rewards probably do have higher utility than later rewards
   - Helps with the infinite sequences problem
9.  How do you solve a MDP problem?
    - Value iteration algorithm 
    - Policy Iteration algorithm
10. What is the Bellman update?
       - Step within the value iteration algorithm to solve an MDP
       - Update Vk+1(s) based on Vk(s)
       - $V_{k+1}(s) \leftarrow max_a\sum_{s'}T(s,a,s')[R(s,a,s')+\gamma V_k(s')]$
11. Under which conditions is policy iteration faster than value iteration when solving an MDP?
    - In practice, value iteration is much faster per iteration, but policy iteration takes fewer iterations, since  value iteration has to run through all possible actions at once to find the maximum action value
12. How can information be shared between different states within RL? Which types of RL share?
    - Over state connections (transitions between states)
    - All that do Bellman update do share: TD-learning, Q-learning
    - No share of information: Monte carlo MC-learning/Direct evaluation
13. Does online RL always share information between states and offline RL always doesn't?
    - Online: Learn while exploring the world, with each experience/sample -> always shares information
    - Offline/Batch learning: learn from fixed batch of data
     Explain the ageing schema of Hoeffding trees
15. Explain the ageing schema in the CluStream algorithm
    - Online step maintains only a fixed number of q micro-clusters over time via snapshots, which is a damped window using a fading function that assigns all points weight 0 and snapshots weight 1
    - offline step inds the active micro-clusters during horizon therefore using sliding window model
16. How can you handle outliers for stream clustering?
    - Denstream: maintains outlier micro-clusters
17. In Hoeffding bound, explain the meaning of n and δ and their relation-
ship to each other.
       - n is the count of independent observations of the random variable r, or sample size
       - δ is the desired confidence level of the calculated mean of the random variable → maximum error in choosing the wrong split attribute
     - If n or δ increases, the Hoeffding bound shrinks. This means, either having much data or a small confidence level leads to a small Hoeffding bound.

17. How does the VFDT deal with concept drifts?
       - VFDT consider all observations and have no forgetting mechanism. Therefore, they can only handle static concepts, not drifting concepts.
18. What type of ageing function is used in DenStream algorithm? Explain
what the γ parameter is and what happens when the value of γ is high.
      - Data are subject to ageing according to the exponential ageing function (damped window model). λ(λ > 0) is the decay rate which determines the importance of historical data. The higher the value of λ, the lower the importance of old data.
19. What is redundancy and irrelevance in the sense of high dimensional data? 
    - not useful for the learning task, addition “destroys” the class separation
    - Redundand:relevant feature may be redundant in the presence of another relevant feature with
    - individually irrelevant features, might be relevant together which it is strongly correlated
20. The IC of a subspace can only get smaller or bigger? Why?
    - Bigger, because the have less features that have by requirement first be the same for several points before we calculate IC and have less correlation to label
21. When is it better to do feature selection and when dimensionality reduction?
    - selection: Always when there are irrelevant/redundant features
    - reduction: If all features seem relevant and nessecary
22. What is the difference between partitioning and hierarchical? 
      - Partitioning algorithms typically have global objectives, e.g., k-Means
      - Hierarchical clustering algorithms typically have local objectives



## Compare:
1. What is the the role of raw training instances for each: KNN, SVM, Decision-Tree, KMeans, Q-Learning?
    - Used to train construct model: SVM, DT, Kmeans
    - Stored and used when new data points arrive: Knn
    - Q-learning: summarized and used when new points arrive
2. Compare policy and value iteration.
    - Both algorithms implicitly update the policy and state value function in each iteration.
    - The policy iteration algorithm updates the policy. one phase evaluates the policy, and the another one improves it.
    - The value iteration algorithm iterates over the value function instead. It takes the maximum over the utility function for all possible actions. It runs through all possible actions at once to find the maximum action value
3. Name similarities and differences between density-based clustering and grid-based clustering.
   - Both: Use density-measure to find cluster
   - density-based: density as distance of point to its k neighbours
   - Grid: density as number of points within a cell
4. Compare TD-learning to Q-learning based on (model-free or model-based, passive or active RL, batch learning or learning from every experience, learns V or Q-values, on-policy or off-policy, Information shared between states (yes/no))
    - TD: model-free, passive(fixed policy provided), online (learn from every experience), learn v-values, on-policy, shares information between states
    - Q: model-free, active (actively choose actions to gather experience), online (learn from every experience), learn Q-values, off-policy, shares information between states
5. What is a disadvantage of using approximate Q-learning instead of the standard Q-learning?
   - Disadvantages: Requires intervention by the designer to add domain-specific knowledge with features. If reward/discount are not balanced right, the agent might prefer accumulating the small rewards to actually solving the problem. Doesn't reduce the size of the Q-table.
6. Compare Kmeans and KNN.
    - Both: Use a distance metric to their k nearest neighbours 
    - knn: classification
    - kmeans: clustering
7. Compare Naive Bayes and SVM.
    - Both: Classification 
    - Naive Bayes: treats features as independent
    - SVM:looks at the interactions between features to a certain degree, as long as you're using a non-linear kernel (Gaussian, rbf, poly etc)
8. Compare Holdout and prequential evaluation method for streams.
    - Both: evaluation methods
    - Holdout: two separate datasets one for Dynamic training one for (static or dynamic) testing
    - prequential: One dynmaic dataset for training and testing
9.  Compare Incremental clustering methods and stream clustering methods.
    - Both: Both save some representation of the data
    - Incremental: require random access to the raw data to update the old clustering based on new instances
    - stream: not assume random access to the data, but rather summaries of data
10. Compare the feature selection methods Forward Selection, Backward Elimination and k-dimensional subspace projections.
    - All: 
    - Forward Selection:
    - Backward Elimination:
    - k-dimensional subspace projections
11. Compare feature selection and dimensionality reduction.
    - Selection: new feature space F' consists of a subset of the original features, useless” features from F have been removed, F’ is interpretable, mostly supervised (need class labels)
    - reduction: new feature space F’ consists of “artificial” variables/features, often not interpretable, typically unsupervised
12. Compare Autoencodes and PCA.
      - Autoencoders: Complex non-linear functions, Features might be correlated, Computationally expensive
      - PCA: Linear transformation, Uncorrelated features, Faster
13. Compare outlier detection methods: Local Outlier Factor (LoF)
and Kth Nearest Neighbor Distance (KthNN).

14. Compare EM to k-Means. 
      - Both: assign each object to a cluster, compute cluster centroids, 
      - EM: soft clustering method
          - assigned to a cluster with a probability,
          - computation of the mean also considers the fact that each object belong to a distribution with a certain probability
          - cluster is represented viaa probability distribution (Gaussian)
      - kmeans: hard clustering method
          - hard assignment, 
          - to compute of the mean, hard assignments of points are considered
          - cluster is represented via a centroid
15. Compare Soft and hierarchial clustering:
      - hierarchial: 
          - instance belongs to more than one clusters in the hierarchy, still this is a hard assignment
      - soft:
          - instance belongs to all clusters with some probability. Flat clustering
16. Compare k-medoids (PAM) and kMeans.
      - kmeans: clusters represented via ``virtual’’ centers
      - PAM: clusters represented via real instances, more robust to outliers 