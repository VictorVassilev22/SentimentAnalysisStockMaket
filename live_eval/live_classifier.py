from pyspark.ml import PipelineModel
from preprocess.preprocess import get_preprocessed_entry
from pyspark.sql.functions import col
import logging
from config.props import transform_pipeline_path as trans_path, lr_best_model_path, nb_best_model_path
from spark.sparkConfig import spark

transformer = PipelineModel.load(trans_path)

nb = PipelineModel.load(nb_best_model_path)
lr = PipelineModel.load(lr_best_model_path)

logging.basicConfig(filename='live-results.log', encoding='UTF-8', level=logging.INFO)
logger = logging.getLogger(__name__)


def to_df(entry):
    df = spark.createDataFrame(get_preprocessed_entry(entry))
    df = df.withColumn("sentiment", df.sentiment.cast("int"))
    df = df.filter((df.sentiment == 1) | (df.sentiment == 0) | (df.sentiment == -1))
    df = df.filter(col("clean_text").isNotNull()).select("clean_text", "sentiment")
    return df


def predictLR(entry):
    return predict(entry, lr)


def predictNB(entry):
    return predict(entry, nb)


# Clean, transform, eval
def predict(entry, model):
    clean_df = to_df(entry)
    clean_df = transformer.transform(clean_df)
    temp = model.transform(clean_df).dropna().select("prediction").collect()[0][0]
    result = 0
    if temp == 2.0:
        result = -1
    elif temp == 0.0:
        result = 1

    logger.info("{}:{} => {}\n"
                .format(model, entry, result))
    return result


# if __name__ == "__main__":
#     # predictLR("$AMD turns red.  That stock still has a long way to go on the downside. #BearMarket $SOX $SMH")  # -1
#     predictLR(
#         "Brokerages Anticipate Xilinx Inc. $XLNX Will Announce Quarterly Sales of $672.19 Million https://t.co/GR3Ex5N840")  # 0
#     predictLR(
#         "Capital One Financial Equities Analysts Boost Earnings Estimates for Carrizo Oil &amp; Gas Inc $CRZO https://t.co/ZgA0Rktv7Z")  # 1
