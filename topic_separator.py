import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
from nltk.corpus import stopwords
from sklearn.decomposition import NMF
import seaborn
import matplotlib.pyplot as plt


class TopicSeparator():
    def __init__(self, bag_of_articles):
        self.articles = bag_of_articles
        self.nmf = None
        self.tfidf = None
        self.vectorialized_articles = None
        self.n_topics = None
        self.topics = {}
    
    def _vectorialize_articles(self):
        stop = stopwords.words('english')
        self.tfidf = TfidfVectorizer(stop_words = stop, max_df = .8)
        self.vectorialized_articles = self.tfidf.fit_transform(self.articles)
           
    def _nmf_articles(self, n_topics):
        self.n_topics = n_topics
        nmf = NMF(n_components=self.n_topics)
        self.nmf = nmf.fit(self.vectorialized_articles)
        
    def _list_main_words(self, topic):
        wordlist = ''
        for word in [self.feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]:
            wordlist += ', '+word
        return wordlist[2:]
    
    def _get_subject(self, wordlist):
        return raw_input("Please enter a topic name defining the 10 following words : "+ wordlist+'      ')
        
    def _label_topics(self):
        self.feature_names = self.tfidf.get_feature_names()
        n_top_words = 10
        for topic_idx, topic in enumerate(self.nmf.components_):
            self.topics[topic_idx] = self._get_subject(self._list_main_words(topic))
            
    def categorize(self, n_topics):
        self._vectorialize_articles()
        self._nmf_articles(n_topics)
        self._label_topics()
            
    def plot_topics_split(self):
        article_counts = {}
        topic_counter = Counter([np.argsort(ts.nmf.transform(ts.vectorialized_articles)[k,:])[::-1][0] for k in range(ts.vectorialized_articles.shape[0])])
        reversed_topics = {v:k for k,v in self.topics.items()}
        for topic in reversed_topics.keys():
            article_counts[topic] = topic_counter[reversed_topics[topic]]
        fig, ax = plt.subplots(1,1)
        ax.bar(range(len(article_counts)), article_counts.values())
        ax.set_xticks(linspace(0.4, len(article_counts)-0.6, len(article_counts)))
        ax.set_xticklabels(article_counts.keys(), rotation = 40)
        plt.tight_layout()
        ax.set_title('Article classification - number of articles per topic', fontweight = 'bold')
        plt.show()
        fig.savefig('Article classification - number of articles per topic.png', dpi=300);
    
    def plot_words_importance(self):
        n = self.n_topics / 2 + self.n_topics % 2
        fig, axes = plt.subplots(n, 2, figsize=(15,5*n))
        for i in range(2):
            if n > 1:
                for j in range(n):
                    if n*i+j < self.n_topics:
                        axes[j, i].bar(range(10), np.sort(self.nmf.components_[n*i+j])[::-1][:10], alpha = 0.8)
                        axes[j, i].set_title('Main words for ' + self.topics[n*i+j], fontsize = 20, fontweight = 'bold')
                        axes[j, i].set_xlim(0,10)
                        axes[j, i].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
                        axes[j, i].set_xticks(np.linspace(0.3, 9.3, 10))
                        axes[j, i].set_xticklabels([self.feature_names[k] for k in 
                            self.nmf.components_[n*i+j].argsort()[:-10 - 1:-1]], rotation = 40)
                        axes[j, i].set_ylim(0,1.1*np.sort(self.nmf.components_[n*i+j])[::-1][0])
                        axes[j, i].set_ylabel('Word importance', fontweight = 'bold')
                        axes[j, i].set_yticks(np.linspace(0, np.sort(self.nmf.components_[n*i+j])[::-1][0], 6))  
            else:
                axes[i].bar(range(10), np.sort(self.nmf.components_[n*i])[::-1][:10], alpha = 0.8)
                axes[i].set_title(self.topics[n*i], fontsize = 20, fontweight = 'bold')
                axes[i].set_xlim(0,10)
                axes[i].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
                axes[i].set_xticks(np.linspace(0.3, 9.3, 10))
                axes[i].set_xticklabels([self.feature_names[k] for k in 
                    self.nmf.components_[n*i].argsort()[:-10 - 1:-1]], rotation = 40)
                axes[i].set_ylim(0,1.1*np.sort(self.nmf.components_[n*i])[::-1][0])
                axes[i].set_ylabel('Word importance', fontweight = 'bold')
                axes[i].set_yticks(np.linspace(0, np.sort(self.nmf.components_[n*i])[::-1][0], 6)) 
        plt.tight_layout()
        plt.subplots_adjust(wspace =.2, top = .9)
        fig.suptitle('Article classification - important words per topic', fontsize = 26, fontweight = 'bold')
        plt.show()
        fig.savefig('Article classification - important words per topic.png', dpi=300);