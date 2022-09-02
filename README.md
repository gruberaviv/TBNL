# TBNL
Targeted Bayesian Network Learning
The TBNL builds a BN classifier and saves the resultsted network onto a file which can be viewed by GeNIe framework.
Generally speaking, Bayesian network learning has two learning stages: structure learning (the graph) and parameter learning (the probability distributions).
GeNIe is a third party app. There are several ways for licensing GeNIe and this is on the user's responsibility.
Using GeNIe is not mandatory, but then the user would have to modify the code to their needs, as the parameter learning stage is tailored for GeNIe.
The special value of the TBNL is that it places the target node at the bottom of the network, and gradually selects the most influential parents of it, by maximizing the total information about it, constrained by graphical constraints and informational constraints.
The same routine is then applied to each one of the parents in turn, resulting in a quasi-cuasal explanatory classifier.
