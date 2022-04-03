# Clustering

## General
- Use:
  - As a stand-alone tool to get insight into data distribution
  - As a preprocessing step for other algorithms
- Application: Marketing, Telecommunications, Land use, City-planning, Bioinformatics, Web
- Goal: Group objects into groups so that the objects belonging in the same group are similar (high intra-cluster similarity), whereas objects in different groups are different (low inter-cluster similarity)
- Cluster labeling: describe them in a human interpretable way
  - Extensive description (enumerate cluster members)
  - Intensive description/cluster labeling (a more abstract description of the properties of the cluster members)
    - depends on data types (e.g., numerical vs categorical) and extra information not used for clustering (like class labels)
      - numerical data: Center and radius
- Requirements on clustering algorithms: 
  - Discovery of clusters with arbitrary shape, 
  - Minimal requirements for domain knowledge to determine input parameters, 
  - Able to deal with noise and outliers, 
  - Incorporation of user-specified constraints, 
  - Interpretability and usability, 
  - Insensitive to the order of input records, 
  - Scalability, 
  - Ability to deal with different types of attributes, 
  - Ability to handle dynamic data, High dimensionality, ...

## Partitioning based clustering
- Construct a partition of D into a set of k clusters
- Each object belongs to exactly one cluster (hard or crisp clustering)
- The number of clusters k is given in advance
- partition should optimize the chosen partitioning criterion, i.e., minimize the intra-cluster distance
- lobal optimal: exhaustively enumerate all partitions
- Heuristic methods: k-Means and k-Medoids

## kMeans, kMedoids
- kMeans
  - Assign each point to closest cluster center and update center of each cluster based on new point(s) until convergence
  - convergence: cluster centers do not change, cost is not improved significantly, a max number of iterations t is reached
  - clusters represented via ``virtual’’ centers
  - Finds a local optimum
  - initial centers can have a large influence in the results
  - sensitive to outliers
  - Not suitable to discover clusters with non-convex shapes
  - Complexity:
    - O(tkn), n: # objects, k: # clusters, t: # iterations.
- kMedoids
  - clusters represented via real instances
  - starts from an initial set of k medoids and iteratively replaces one of the medoids by one of the non-medoid points iff such a replacement improves the total clustering cost
  - complexity:
    - O(k(n-k)^2) for each iteration, n is # of data, k is # of clusters
  - Efficiently for small data sets but does not scale well for large data sets.

## Selecting number of clusters
- Silhouette coefficient of an object i:
  - $i \in Cluster A$
  - a(i) the distance of i to A (the so-called best first cluster distance)
- The Silhouette coefficient of a cluster is the avg silhouette of all its objects
  - Is a measure of how tightly grouped all the data in the cluster are.
  - > 0,7: strong structure, > 0,5: usable structure
- Silhouette coefficient of a clustering is the avg silhouette of all objects
  - measure of how appropriately the dataset has been clustered

## Hierarchical clustering
- Produces a set of nested clusters organized as a hierarchical tree visualized also as a dendrogram
- instance can belong to multiple clusters, the assignement is still hard!
- dendrogram of clusters may correspond to meaningful taxonomies
- use a similarity or distance matrix to decide on which cluster to split/merge next
- Hierarchical clustering methods:
  - Agglomerative or AGNES (Agglomerative Nesting):
    - Bottom-up approach
    - Start with the points as individual clusters
    - At each step, merge the closest pair of clusters until only one cluster (or k clusters) left
  - Divisive or DIANA (Divisive analysis):
    - Top-down approach
    - Start with one, all-inclusive cluster
    - At each step, split a cluster until each cluster contains a single point (or there are k clusters)
- No knowledge on the number of cluster
- Produces a hierarchy of clusters, not a flat clustering
- No backtracking: Merging decisions are final
- Lack of a global objective function (Decisions are local, at each step --> no objective function is directly minimized)
- Different schemes have problems with one or more of the following:
  - Sensitivity to noise and outliers
  - Breaking large clusters
  - Difficulty handling different sized clusters and convex shapes
  - Inefficiency, especially for large datasets
- Complexity:
  - O(n^2) space to store the proximity matrix
  - O(n^3) time in most of the cases
  - reduced to O(n2 log(n) ) time for some approaches using appropriate data structures
- Agglomerative approaches
  - agglomerative clustering algorithm
    - Compute the proximity matrix
    - Let each data point be a cluster
    - Repeat:
      - Merge the two closest clusters
      - Update the proximity matrix
    - Until only a single cluster remains
  - Similarity measures between clusters:
    - Centroid-link distance
    - (Group) Average-link distance
    - Complete link distance
    - Single link distance

## Bisecting kMeans
- combines k-Means and hierarchical clustering
- first split the set of points into two clusters, select one of these clusters for further splitting,
and so on, until k clusters remain

## Density-based clustering
- Clusters are regions of high density surrounded by regions of low density (noise)
- Directly density-reachable: A point p is directly density-reachable from a point q w.r.t. Eps, MinPts if: p belongs to NEps(q) and q is a core point, i.e.,: |NEps(q)| >= MinPts; not a symmetric relation
- Density-reachable: A point p is density-reachable from a point q w.r.t. Eps, MinPts if there is a chain of points p1, …, pn, p1 = q, pn = p such that pi+1 is directly density-reachable from pi, not a symmetric relation
- Density-connected: A point p is density-connected to a point q w.r.t. Eps, MinPts if there is a point o such that both, p and q are density-reachable from o w.r.t. Eps and MinPts, Density-connectedness is symmetric
- 
- DBSCAN
  - pros: Resistant to noise, handle clusters of different shapes and sizes
  - cons: fails to identify clusters of varying densities, high-dimensional data due to curse of dimensionality
  - determining Eps and MinPts:
    - calculate, the distance of every point to its k nearest neighbor. The value of k will be specified by the user and corresponds to MinPts
    - determine the “knee”, which corresponds to the optimal eps parameter.
      - sharp change occurs along the k-distance curve
  - complexity
    - n points, the time complexity of DBSCAN is worst case O(n^2)
    - In low-dimensional spaces O(nlogn)

## Grid-based clustering
- Density = number of points within each cell
- cluster is a set of connected dense cells
- Steps:  
  - Dense cells are identified
  - Neighboring dense cells form clusters
  - cluster is a maximal set of connected dense cells
- algorithms: CLIQUE, STING, WaveCluster,...
- pros:  No assumption on the number of clusters, Discovering clusters of arbitrary shapes, Ability to handle outliers
- cons: result depends on the grid parameters (cell size and cell density, which are typically global)

## Cluster evaluation
- Cluster validation goals:
  - Determining the clustering tendency of a dataset
  - Comparing the results of a cluster analysis to externally known results
  - Evaluating how well the results of a cluster analysis fit the data without reference to external information.
  - Comparing the results of two different sets of cluster analyses to determine which is better.
  - Determining the ‘correct’ number of clusters (and other input parameters).
  - evaluate the entire clustering or just individual clusters?
- Internal measures
  - Rely on cluster-member characteristics, no external information is available
  - Cluster cohesion is the sum of the weight of all links within a cluster.
    - How closely related are objects in a cluster
  - Cluster separation is the sum of the weights between nodes in the cluster and nodes outside the cluster.
    - How well-separated a cluster is from other clusters
  - Silhouette Coefficient
    - combines ideas of cohesion and separation, for individual points, as well as for clusters and clusterings
- External measures
  - match externally supplied class labels
  - entropy, purity

## Soft clustering
- allows instances to belong to more than one clusters
- membership probabilities must sum to 1.0
- Mixture models are a probabilistically-grounded way of doing soft clustering
- Each cluster corresponds to a probability distribution (Gaussian or multinomial)

## Expectation Maximization (EM)
- Soft clustering to find parameter of distribution
- Steps:
  - Initialize: Start with two randomly placed Gaussians (μc ,Σc)
  - Two alternating steps:
    - E-step (“Expectation”): re-estimate the cluster assignments under the current estimate of the model
    - M-step (“Maximization”): re-estimate the model parameter under the current assignment
  - until convergence