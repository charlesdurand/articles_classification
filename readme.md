# Article classification

### Objectives

This python script provides an easy topic separator - given a set of documents, and a number of topics, as well as two ready to use vizualisation tools.

### Use

The class **TopicSeparator** is initialized with a set of documents.
The function **categorize()** enables the classification by topic, given a number of topics. For each topic found, the user is asked to enter labels given the ten most defining words of the topic (the algorithm uses NMF with the tdfidf version of the documents):
![alt text](https://github.com/charlesdurand/articles_classification/blob/master/images/topic_definition.png)

The two functions **plot_topics_split()** and **plot_words_importance()** enable to visualize the categorisation:

##### plot_topics_split()
![alt text](https://github.com/charlesdurand/articles_classification/blob/master/images/topic_split.png)
    
##### plot_words_importance()
![alt text](https://github.com/charlesdurand/articles_classification/blob/master/images/word_importance.png)
