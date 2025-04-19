# In case you don't feel like scrolling through the notebook to find the responses

# NLP Disaster Tweets Kaggle Mini Project

This challenge was an entry level task for prospective data scientists to get some valuable experience with Natural Language Processing. 

## Problem:

X has become an important communication channel in times of emergency. Smartphones enables people to announce an emergency they're observing in real-time. Because of this, more agencies are interested in programtically monitoring X. The Problem is it isn't always clear whether a person's words are actually announcing a disaster or if they are using language that is just popular to describe something else. The goal of this challenge is to build a learning model that predicts which posts are about real disasters and which one's aren't.

## Data

The dataset contains over 10,000 tweets that were hand classified for this project. In the test and train dataset, the data contains the text of the tweet, a keywords from the text, and the location the tweet was sent from. The goal is to use these datasets to predict whether a given tweet is about a disaster which will be labeled as a 1 or not which would be labeled as a 0.

## Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data:

This dataset had a good amount of work that needed to be done to get things prepared for models. The data included values such as location and keywords, but both of these variables were allowed to be left blank. This many missing variable needed to be addressed in the EDA to properly have the dataset cleaned before visualizing the data could provide any accurate information. The solution that makes the most sense would be to replace the NaN value with a "No_Keyword" so that the data isn't removed, but it has a better value than null.

Visualizing the data we can see how many all of the meta features have information about target as well, but some of them are not good enough such as url_count, hashtag_count and mention_count.

## Model Architecture

As stated in the grading rubric, we didn't learn any specific NLP- techniques such as word embeddings in the lectures, so I had to watch a guide and use what they were able to do for this part of the project to understand and speed up the process of this project. 

I used GloVe for the pre-trained embedding. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

Using this tool without direction is a bad decision so building a vocabulary to assist GloVe is necessary so that valuable information is not lost.

## Results and Analysis

DisasterDetector is a wrapper that incorporates the cross-validation and metrics stated above.

The tokenization of input text is performed with the FullTokenizer class from tensorflow/models/official/nlp/bert/tokenization.py. max_seq_length parameter can be used for tuning the sequence length of text.

Parameters such as lr, epochs and batch_size can be used for controlling the learning process. There are no dense or pooling layers added after last layer of BERT. SGD is used as optimizer since others have hard time while converging.

plot_learning_curve plots Accuracy, Precision, Recall and F1 Score (for validation set) stored after every epoch alongside with training/validation loss curve. This helps to see which metric fluctuates most while training.

## Discussion / Conclusion

This was the first time I was working on a Natural Language Processing project. I had a lot of fun following guides and doing my best to understand more on the topic while working on it as the information from the lectures wasn't enough for me to feel like I was gaining a grasp on the skill. 

I had a lot of bugs trying to get the code to work, and the model also took a whole day to compile. But I was happy with the results and I plan on playing around with it more to explore the results further.

There is a lot that can be done for my learning going forward. This is supposedly a basic project to help people begin learning NLPs so I would love to work on more advanced projects to push what I am capable of doing. 

### References

* Firoj Alam, Hassan Sajjad, Muhammad Imran and Ferda Ofli, CrisisBench: Benchmarking Crisis-related Social Media Datasets for Humanitarian Information Processing, In ICWSM, 2021. [Bibtex]
* Firoj Alam, Ferda Ofli and Muhammad Imran. CrisisMMD: Multimodal Twitter Datasets from Natural Disasters. In Proceedings of the International AAAI Conference on Web and Social Media (ICWSM), 2018, Stanford, California, USA.
* Muhammad Imran, Prasenjit Mitra, and Carlos Castillo: Twitter as a Lifeline: Human-annotated Twitter Corpora for NLP of Crisis-related Messages. In Proceedings of the 10th Language Resources and Evaluation Conference (LREC), pp. 1638-1643. May 2016, Portorož, Slovenia.
* A. Olteanu, S. Vieweg, C. Castillo. 2015. What to Expect When the Unexpected Happens: Social Media Communications Across Crises. In Proceedings of the ACM 2015 Conference on Computer Supported Cooperative Work and Social Computing (CSCW '15). ACM, Vancouver, BC, Canada.
* A. Olteanu, C. Castillo, F. Diaz, S. Vieweg. 2014. CrisisLex: A Lexicon for Collecting and Filtering Microblogged Communications in Crises. In Proceedings of the AAAI Conference on Weblogs and Social Media (ICWSM'14). AAAI Press, Ann Arbor, MI, USA.
* Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Extracting Information Nuggets from Disaster-Related Messages in Social Media. In Proceedings of the 10th International Conference on Information Systems for Crisis Response and Management (ISCRAM), May 2013, Baden-Baden, Germany.
* Muhammad Imran, Shady Elbassuoni, Carlos Castillo, Fernando Diaz and Patrick Meier. Practical Extraction of Disaster-Relevant Information from Social Media. In Social Web for Disaster Management (SWDM'13) - Co-located with WWW, May 2013, Rio de Janeiro, Brazil.
* https://appen.com/datasets/combined- disaster-response-data/
* https://data.world/crowdflower/disasters- on-social-media
