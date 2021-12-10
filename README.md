# movie_summarization_capstone

![cover_photo](./files/cover_photo.jpg)

*Over the past few years Natural Language Processing (NLP) abilities and applications have seen a lot of growth, and text summarization is a big part of that. Text summarization is generating intelligent, accurate and coherent summaries for long pieces of text. There are two fundamental approaches to text summarization: extractive and abstractive. The extractive approach takes exact words, phrases and sentences from the original text to create a summary. The abstractive approach learns the internal representation of the text to generate new sentences for the summary.*

*Our goal is to look at a few models for both extractive and abstractive techniques and compare which models perform best, as well as doing a guided walkthrough of setting up a recurrent neural network. One objective measure we will use to compare these approaches is cosine similarity scores between the generated summary and the actual summary. Another is their rouge score. Unfortunately, these do not tell us what is most important, which is how coherent and accurate the generated summaries are. At this point humans are simply better at evaluating summaries. Thus, we will need to use our subjective opinions on the accuracy and coherence of the text produced. *

## 1. Data

This dataset comes from the Kaggle website. To view the original Kaggle data or to import it using the Kaggle API click on the link below:

> * [Kaggle Dataset](https://www.kaggle.com/jrobischon/wikipedia-movie-plots)


## 2. Data Cleaning 

Our data consists of 2 csv files, wiki_movie_plots_deduped.csv and movies_metadata.csv. We merged the two and ultimitely only kept the columns of 'Release Year', 'Title', 'Plot', and 'Overview.' Additionally, we constructed columns for 'cleaned_plot' and 'cleaned_overview.' We dropped duplicate rows and rows that don't contain both 'Plot' and 'Overview'

* **Problem 1:** Some rows had missing values for the plot and some for the summary. **Solution:** This is why we merged the 2 tables. For movies that had no summary or plot in one of the tables, we simply filled in the missing info from the other table if possible. Any rows left that had missing values were dropped . 

* **Problem 2:** Tensorflow and CUDA toolkit do not have versions compatable with this computer's GPU  **Solution:** I had to use the CPU and drastically cut down on the number of movies, which made it nearly impossible for the neural network to produce quality results. It is hard to learn something as complex as the English language from fewer than 2,000 movie summaries. This resulted in the neural network becoming a guided walkthrough, rather than something we expect quality results to come from that can be evaluated on their own.  

* **Problem 3:** There is a lot of variability in the plots and overviews.  **Solution:** We choose those movies whose plot length was less than 2 standard deviations higher than the mean and less tan 2 standard deviation below the mean. We also chose movies whose overview length was between 10 and 55 words. 

* **Problem 4:** NLP and neural networks require a lot of pre-processing **Solution:** We converted everything to lowercase, removed HTML tags, did contraction mapping, removed text inside parenthesis, eliminated punctuation and special characters and removed stopwords.


## 3. EDA

* Being an NLP project, there wasn't much EDA. We did do word counts to guide our cutoffs for plot and overview lengths 

![](./files/word_len_dist.png)

## 4. Models and Librariries

I used a few different models from a few libraries. We used the SpaCy library to build an extractive model that simply weighs each sentence based on the frequency of the token present in each sentence, and the top n sentences are used to create the summary. From the sumy library we used 2 different models for extractive summarization: LSA, which is an unsupervised method that combines term frequency techniques with SVD; Luhn, which is based on TF-IDF and scores sentences based on frequency of the most important words. For abstractive summarization we built our own seq2seq RNN as a guided walkthrough of building a neural network for text summarization and also used the T5 and Bart models from the transformers library. T5 uses relative scalar embeddings and is pre-trained on a multi-task mixture of unsupervised and supervised tasks, for which each task is converted into a text-to-text format. Bart is a denoising autoencoder that uses both BERT (bidirectional encoder) and GPT architecture that was trained on the CNN/Daily Mail data set.


**WINNER: BART**

First, lets look at the subjective measure. For brevity, we will only evaluate the generated summaries of one movie. It should be fairly clear that the abstractive generated summaries perform much better and are much more coherent. The sentences flow nicely into each and it seems we get a summary that looks real, compared to the awkwardly placed sentences from the extractive models.  

![](./files/subj_eval.png)

Now let's look at the objective measures. 

![](./files/cosine_sim.png)

![](./files/rouge_scores.png)

## 5. Summary

* As we can see, the extractive approaches do fairly similar to the abstractive approaches with cosine similarity. This is likely because the extractive approach literally lifts full sentences from the original text. With the ROUGE scores, the clear winner is the Bart model. It has by far the highest precision and f1 scores for all three ROUGE scores. Spacy does notably well with recall, but poorly in precision and f1.

However, the real test is the human subjective test. With the extractive approaches, it is pretty hit or miss if the generated summary is actually able to capture the essence of the original plot and because they use fully lifted sentences, most of the sentences of the generated summaries do not flow into one another naturally. They all seem out of place. This is where the abstractive methods really shine. They are not perfect, but they actually do really well to capture the essence of the plot in a concise manner where sentences flow naturally. Again, this is subjective, but almost anyone would agree the abstractive methods are the clear winners. It is perhaps hard to choose between T5 and Bart by comparing summaries alone, but along with the objective scores, it seems Bart is the clear winner here. 
