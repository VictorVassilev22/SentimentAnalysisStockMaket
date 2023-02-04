import tweepy

# configure your paths and env

rule = '($TSLA tesla) lang:en'
search_rules = [tweepy.StreamRule(rule)]
# Example: tweepy.StreamRule('(trump OR musk) lang:en -is:reply -is:retweet')
topics = ['TWITTER']  # Set topic to stock exchange listed company e.g. TSLA, GOOG, AAPL
bootstrap_servers = ['localhost:9092']

stops = ['myself', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours',
         'yourself', 'yourselves', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', "it's", 'its',
         'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that',
         "that'll", 'these', 'those', 'are', 'was', 'were', 'been', 'being', 'have', 'has', 'had', 'having', 'does',
         'did', 'doing', 'the', 'and', 'but', 'because', 'until', 'while', 'for', 'with', 'about', 'between',
         'into', 'through', 'during', 'before', 'after', 'from', 'then', 'once', 'here', 'there', 'when', 'where',
         'why', 'how', 'any', 'both', 'each' 'other', 'some', 'same', 'than', 'will', 'just', 'now',
         "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u",
         "v", "w", "x", "y", "z"]

# full dataset path
full_dataset_path = '/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/dataset/csv/combined_dataset33.csv'
prep_dataset_path = '/home/victor/PycharmProjects/sentiment/dev/dataset/preprocessed_dataset33.csv'
transform_pipeline_path = "/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/model/pipelines/transform"
nb_model_path = "/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/model/pipelines/pipeline_model_nb_experimental_cv_"
lr_model_path = "/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/model/pipelines/pipeline_model_lr_experimental_cv_"
nb_best_model_path = "/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/model/pipelines/pipeline_model_nb_experimental_cv_33/bestModel"
lr_best_model_path = "/home/victor/Desktop/SentimentAnalysisStockMarket/SentimentAnalysisStockMaket/model/pipelines/pipeline_model_lr_experimental_cv_33/bestModel"
