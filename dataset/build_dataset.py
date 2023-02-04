import logging

from preprocess.preprocess import preprocess, lemmatize_entry
from spark.sparkConfig import spark
import pandas as pd
from config.props import full_dataset_path

from pyspark.sql.functions import when
from pyspark.sql.types import IntegerType

logging.basicConfig(filename='SCNLP_results.log', encoding='UTF-8', level=logging.INFO)
logger = logging.getLogger(__name__)


# Takes the combined dataset, preprocesses it and saves it
def savePreprocessedData(file_name):
    df = pd.read_csv(full_dataset_path, index_col=False, error_bad_lines=False).dropna()
    # preprocess
    df = preprocess(df, 'text', 'clean_text')
    # lemmatize
    df['clean_text'] = df['clean_text'].apply(lemmatize_entry)
    df = df[['clean_text', 'sentiment']]
    # save
    df.to_csv(file_name, index=False)
    print('Data saved!')


# produces a new dataset by combining other datasets
# set path to your data and edit the code to change it accordingly
def extract_df(binary=True):
    # Load the text data into a PySpark dataframe //
    # df0 = spark.read.csv("path", header=True)
    df1 = spark.read.csv("path", header=True)
    # df2 = spark.read.csv("path", header=True).limit(8300)
    # df3 = spark.read.csv("path", header=True)
    df4 = spark.read.csv("path", header=True)
    df5 = spark.read.csv("path", header=True)
    # Select only the text and sentiment columns
    # df0 = df0.select("headline", "sentiment")
    df1 = df1.select("text", "Sentiment")
    # df2 = df2.select("Sentence", "Sentiment")
    # df3 = df3.select("Text", "Sentiment")
    df4 = df4.select("text", "text_sentiment")
    df5 = df5.select("text", "Sentiment")
    # Rename the column "headline" to "text"
    # df0 = df0.withColumnRenamed("headline", "text")
    df1 = df1.withColumnRenamed("Sentiment", "sentiment")
    # df2 = df2.withColumnRenamed("Sentence", "text")
    # df2 = df2.withColumnRenamed("Sentiment", "sentiment")
    # df3 = df3.withColumnRenamed("Text", "text")
    # df3 = df3.withColumnRenamed("Sentiment", "sentiment")
    df4 = df4.withColumnRenamed("text_sentiment", "sentiment")
    df5 = df5.withColumnRenamed("Sentiment", "sentiment")

    # filter
    # df0 = df0.filter((df0["sentiment"] == 1) | (df0["sentiment"] == 0))
    # if not binary:
    #     df0 = df0.withColumn("sentiment", when(df0["sentiment"] == 0, -1).otherwise(df0["sentiment"]))

    if binary:
        df1 = df1.filter((df1["sentiment"] == 1) | (df1["sentiment"] == -1))
        df1 = df1.withColumn("sentiment", when(df1["sentiment"] == -1, 0).otherwise(df1["sentiment"]))

    df1 = df1.filter((df1["sentiment"] == 1) | (df1["sentiment"] == 0) | (df1["sentiment"] == -1))
    # df2 = df2.filter((df2["sentiment"] == 1) | (df2["sentiment"] == 0))

    # if not binary:
    #     df2 = df2.withColumn("sentiment", when(df2["sentiment"] == 0, -1).otherwise(df2["sentiment"]))

    # if binary:
    #     df3 = df3.withColumn("sentiment", when(df3["sentiment"] == -1, 0).otherwise(df3["sentiment"]))
    # df3 = df3.filter((df3["sentiment"] == 1) | (df3["sentiment"] == 0) | (df3["sentiment"] == -1))
    df4 = df4.filter(df4["text"] != 'Positive')
    df4 = df4.filter(df4["text"] != 'Negative')
    df4 = df4.filter(df4["text"] != 'Neutral')
    df4 = df4.filter(
        (df4["sentiment"] == 'Positive') | (df4["sentiment"] == 'Negative') | (df4["sentiment"] == 'Neutral'))

    if binary:
        df4 = df4.withColumn("sentiment", when(df4["sentiment"] == 'Negative', 0).otherwise(df4["sentiment"]))
    else:
        df4 = df4.withColumn("sentiment", when(df4["sentiment"] == 'Negative', -1).otherwise(df4["sentiment"]))

    df4 = df4.withColumn("sentiment", when(df4["sentiment"] == 'Neutral', 0).otherwise(df4["sentiment"]))
    df4 = df4.withColumn("sentiment", when(df4["sentiment"] == 'Positive', 1).otherwise(df4["sentiment"]))

    if binary:
        df5 = df5.withColumn("sentiment", when(df5["sentiment"] == -1, 0).otherwise(df5["sentiment"]))
    df5 = df5.filter((df5["sentiment"] == 1) | (df5["sentiment"] == 0) | (df5["sentiment"] == -1))

    # drop null values
    # df0 = df0.dropna()
    df1 = df1.dropna()
    # df2 = df2.dropna()
    # df3 = df3.dropna()
    df4 = df4.dropna()
    df5 = df5.dropna()
    # print('ORIGINAL 0 DATASET: {} ENTRIES'.format(df0.count()))
    print('ORIGINAL 1 DATASET: {} ENTRIES'.format(df1.count()))
    # print('ORIGINAL 2 DATASET: {} ENTRIES'.format(df2.count()))
    # print('ORIGINAL 3 DATASET: {} ENTRIES'.format(df3.count()))
    print('ORIGINAL 4 DATASET: {} ENTRIES'.format(df4.count()))
    print('ORIGINAL 5 DATASET: {} ENTRIES'.format(df5.count()))
    # df0.show()
    df1.show()
    # df2.show()
    # df3.show()
    df4.show()
    df5.show()

    # Combine the dataframes into one
    combined_df = df1
    dataframes = [df4, df5]
    for df in dataframes:
        combined_df = combined_df.unionByName(df)

    # Convert the sentiment column to integer type
    combined_df = combined_df.withColumn("sentiment", combined_df["sentiment"].cast(IntegerType())).dropna()

    print('FINAL DATASET: {} ENTRIES'.format(combined_df.count()))
    return combined_df


if __name__ == "__main__":
    savePreprocessedData("csv/preprocessed_dataset33.csv")
    # df = extract_df(False)
    # df.coalesce(1).write.csv('path', header=True)
