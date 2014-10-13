# Topic separator

### Objectives

This python script provides an easy topic separator - given a set of documents, and a number of topics, as well as two ready to use vizualisation tools.

### Use

The class **TopicSeparator** is initialized with a set of documents.
The function **categorize()** enables the classification by topic, given a number of topics. For each topic found, the user is asked to enter labels given the ten most defining words of the topic (the algorithm uses NMF with the tdfidf version of the documents). 
![alt text]()


The two functions **plot_topics_split()** and **plot_words_importance()** enable to visualize the categorisation:

#### plot_topics_split()
![alt text]()
    
#### plot_words_importance()
![alt text]()
