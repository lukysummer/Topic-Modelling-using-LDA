# Topic Modelling using Latent Drichlet Allocation (LDA)

<p align="center"><img src="images/LDA.png" height = "256"></p>

This is my implementation of Topic Modelling of News Headlines with LDA, which:


* pre-processes all texts of the headlines by cleaning, stemming, and tokenizing,
* predicts underlying topics of a group of texts, then
* outputs the probabilities that a body of text belongs to each of the predicted topics


This model includes two Drichlet distributions:


1. **distribution of Topics per each Body of Text** (alpha)

2. **distribution of Words per each Topic** (eta)



## Sources

I referenced Udacity's Natural Language Processing Nanodegree's workspace.

