# Topic Modelling using Latent Drichlet Allocation (LDA)

<p align="center"><img src="images/news.png", width = 30%></p>

This is my implementation of Topic Modelling of News Headlines with LDA, which:


* pre-processes all texts of the headlines by cleaning, stemming, and tokenizing,
* predicts underlying topics of a group of texts, then
* outputs the probabilities that a body of text belongs to each of the predicted topics


## Results:
The model predicted 10 different underlying topics among 300,000 headlines, with 3 keywords:

1. (government, fund, service)     : could be a government-related topic
2. (kill, iraq, attack)            : could be a terrorist attack-related topic
3. (court, accuse, drug)           : could be a drug-related topic
4. (water, farmer, safety)         : could be an agriculture-related topic
5. (worker, drought, price)        : could be a natural disaster-related topic
6. (power, plan, nuclear)          : could be a nuclear-related topic
7. (world, play, final)            : could be a sports-related topic
8. (council, minister, elect)      : could be an election-related topic
9. (coast, blaze, firefighter)     : could be a fire-related topic
10. (police, crash, investigate)   : could be an accident-related topic



## Model

This model includes two Drichlet distributions:


1. **distribution of Topics per each Body of Text** (alpha)

2. **distribution of Words per each Topic** (eta)



## Sources

I referenced Udacity's Natural Language Processing Nanodegree's workspace.([Course Page](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892))
