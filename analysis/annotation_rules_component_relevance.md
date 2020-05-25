# Annotation Rules
Use these as guidelines to manually annotate component relevance.

Task: Given a component (i.e. a class in the target API, Scikit-Learn)
as a query, retrieve the top 10 "nearest" components (for some distance metric).

We solve this task using two approaches:
  * Compute cosine similarity between query component and API classes using
  pre-trained embeddings
  * Compute relevance using BM25
  * Randomly sample from the API (independent of the query component),
  which we use to baseline both approaches.


# Determining relevance
Given that the relevance of a retrieved component with respect to the query component
can be subjective, we formulate a set of rules we use in our annotation to
systematically assess relevance.

For the rules, we refer to the query component as Q and a retrieved component
as R. For R to be relevant, we require the following to be true:

* R must be functionally equivalent to Q. In our case, functional equivalence
means we can replace Q with R in a pipeline without raising any errors or
performing a fundamentally different operation. For example:
  - A classifier can replace another classifier
  - A value normalizing operation can replace another value normalizing operation
  - A decomposition algorithm can replace another decomposition algorithm
* R must respect output shape constraints. In particular, if Q is a multi-task
regressor/classifier, R must be a multi-task regressor/classifier. If Q
is a single task classifier/regressor, R may be either single-task or
multi-task (as R can still replace Q).
* If Q is a linear classifier/regressor, R must be linear.
* If Q is a non-linear classifier/regressor, R must be non-linear. However, if Q is non-linear due to composition of piece-wise linear (e.g. neural network), then R may be linear.
* If Q can be both linear or non-linear (depending on a hyper-parameter choice, e.g. SVM with linear vs RBF kernel), then R can be both linear or non-linear.
* If Q is not ensemble-based, R may be ensemble-based if the individual models used in R
are relevant for Q (for example, Q is a decision tree, then R may be a random forest model), otherwise R must not be ensemble-based.
* If Q is ensemble-based, R may be non-ensemble-based if the individual models used in Q
are relevant to R (for example, Q is a random forest, then R may be a decision tree), otherwise R must be ensemble-based.
