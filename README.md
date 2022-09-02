# TBNL
## Targeted Bayesian Network Learning
### The TBNL builds a BN classifier and saves the resulted network onto a file which can be viewed by GeNIe framework.
### Generally speaking, Bayesian network learning has two learning stages: structure learning (the graph) and parameter learning (the probability distributions).
### GeNIe is a third party app. There are several ways for licensing GeNIe and this is on the user's responsibility.
### Using GeNIe is not mandatory, but then the user would have to modify the code to their needs, as the parameter learning stage is tailored for GeNIe.
### The special value of the TBNL is that it places the target node at the bottom of the network, and gradually selects the most influential parents of it, by maximizing the total information about it, constrained by graphical constraints and informational constraints.
### The same routine is then applied to each one of the parents in turn, resulting in a quasi-cuasal explanatory classifier.
#### GeNIe: https://www.bayesfusion.com/genie/
#### TBNL: Gruber, A., & Ben-Gal, I. (2019). A targeted Bayesian network learning for classification. Quality Technology and Quantitative Management, 16(3), 243-261. https://doi.org/10.1080/16843703.2017.1395109
