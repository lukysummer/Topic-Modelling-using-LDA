# ABC News Headline Topic Modelling using Latent Drichlet Allocation (LDA)

<p align="center"><img src="images/news.png", width = 40%></p>

This is my implementation of Topic Modelling of News Headlines with LDA, which:


* pre-processes all texts of the news headlines by cleaning, stemming, and tokenizing,
* predicts underlying topics of a group of texts, then
* outputs the probabilities that a body of text belongs to each of the predicted topics


## Results:
The model predicted 10 different underlying topics among 300,000 headlines, with **top 5 keywords**:

1. ("government", "fund", "urgent", "service", "warn") --- could be a **government**-related topic
2. ("kill", "iraq", "closer", "attack", "troop") --- could be a **terrorist attack**-related topic
3. ("charge", "court", "face", "accuse", "drug") --- could be a **drug crime**-related topic
4. ("water", "miss", "search", "break", "polic") --- could be a **missing body or crime**-related topic
5. ("worker", "drought", "price", "reject", "hous") --- could be an **agricultural disaster**-related topic
6. ("plan", "power", "rise", "rule", "nuclear") --- could be a **nuclear power**-related topic
7. ("open", "test", "lead", "world", "win") --- could be a **sports**-related topic
8. ("council", "plan", "minist", "back", "elect") --- could be an **election**-related topic
9. ("continu", "coast", "blaze", "market", "firefight") --- could be a **fire**-related topic
10. ("polic", "crash", "investig", "jail", "death") --- could be an **accident or crime investigation**-related topic

The detected topics such as different types of crimes and natural disasters make sense to appear frequnetly in news headlines.



## Model

This model includes two Drichlet distributions:


1. **distribution of Topics per each Body of Text** (alpha)

2. **distribution of Words per each Topic** (eta)



## Sources

I referenced Udacity's Natural Language Processing Nanodegree's workspace.([Course Page](https://www.udacity.com/course/natural-language-processing-nanodegree--nd892))
