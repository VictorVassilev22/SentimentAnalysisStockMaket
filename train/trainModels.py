import logging

from pyspark.ml.tuning import ParamGridBuilder
import os
from pyspark.ml.feature import Tokenizer, IDF, HashingTF, StopWordsRemover, StringIndexer
from pyspark.ml.classification import NaiveBayes, LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.tuning import CrossValidator

from spark.sparkConfig import spark
from config.props import prep_dataset_path as dataset_path, transform_pipeline_path, nb_model_path, lr_model_path
from pyspark.sql.types import StructType, StructField, StringType, IntegerType


def cv_save_model(df, pipeline, grid, num_folds, train_p, test_p, func_name, path, code):
    (train, test) = df.randomSplit([train_p, test_p], seed=222)

    # Create the cross-validator
    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid,
                        evaluator=MulticlassClassificationEvaluator(),
                        numFolds=num_folds)

    # Fit the model
    cv_model = cv.fit(train)
    avg_metrics = cv_model.avgMetrics
    for i, metric in enumerate(avg_metrics):
        print("Model", i, "has average", metric)
    cv_model.write().overwrite().save(path + str(code))

    # Make predictions
    predictions = cv_model.transform(test)
    test.show()

    # Evaluate the model
    evaluator = MulticlassClassificationEvaluator()
    f1 = evaluator.evaluate(predictions)
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    fprl = evaluator.evaluate(predictions, {evaluator.metricName: "falsePositiveRateByLabel"})
    print("F1 score:", f1)
    print("acc score:", accuracy)
    print("fprl score:", fprl)
    logger.info(
        "{}: Model: {}, (train/test):({}/{}) ," "F1 score = {}, Accuracy = {}, FalsePositiveRateByLabel = {}".format(
            func_name, code,
            train_p, test_p,
            str(f1), str(accuracy), str(fprl)))


def get_transformed():
    # path to model pipeline transformer
    pipeline_path = transform_pipeline_path
    schema = StructType([
        StructField("clean_text", StringType(), True),
        StructField("sentiment", IntegerType(), True)
    ])

    df = spark.read.csv(dataset_path, header=True, schema=schema).dropna()
    df.show()
    print(df.count())

    if not (os.path.exists(pipeline_path)):
        # build and save the pipeline
        tokenizer = Tokenizer(inputCol="clean_text", outputCol="words")
        swr = StopWordsRemover(inputCol='words', outputCol="cleanWords")
        hashing_tf = HashingTF(inputCol='cleanWords', outputCol='rawFeatures')
        idf = IDF(inputCol='rawFeatures', outputCol='features', minDocFreq=5)
        idx = StringIndexer(inputCol="sentiment", outputCol="label")

        pipeline = Pipeline(stages=[tokenizer, swr, hashing_tf, idf, idx])
        # Read the CSV file
        pipeline_model = pipeline.fit(df)
        # save pipeline
        pipeline_model.write().overwrite().save(pipeline_path)

    else:
        pipeline_model = PipelineModel.load(pipeline_path)

    # transform data ready for model
    df_t = pipeline_model.transform(df)

    return df_t


# Naive Bayes is fast to train and predict and is ideal for real-time classification
def trainNB(df, code, train_p, test_p, num_folds):
    nb = NaiveBayes(featuresCol='features', predictionCol="prediction",
                    modelType="multinomial")

    pipeline = Pipeline(stages=[nb])

    # I have selected these parameters after cross-validation (CV)
    # Do CV on your own data to extract best model
    grid = (ParamGridBuilder()
            .addGrid(nb.smoothing, [1])
            .addGrid(nb.thresholds, [[0.3, 0.2, 0.5]])
            .build())

    cv_save_model(df, pipeline, grid, num_folds, train_p, test_p, trainNB.__name__,
                  nb_model_path, code)


# Logistic Regression seems to perform better in accuracy than NB
def trainLR(df, code, train_p, test_p, num_folds):
    lr = LogisticRegression(family="multinomial", aggregationDepth=8)
    pipeline = Pipeline(stages=[lr])

    grid = (ParamGridBuilder()
            .addGrid(lr.regParam, [0.01])
            .addGrid(lr.maxIter, [50])  # 50
            .addGrid(lr.elasticNetParam, [0.1])
            .addGrid(lr.tol, [1e-6])
            .addGrid(lr.fitIntercept, [True, False])
            # .addGrid(lr.standardization, [True, False])
            .addGrid(lr.thresholds, [[1 / 3, 1 / 3, 1 / 3]])
            .addGrid(lr.family, ['multinomial'])
            .build())

    cv_save_model(df, pipeline, grid, num_folds, train_p, test_p, trainLR.__name__,
                  lr_model_path, code)


# TODO: IDEAS:
# TODO: Test more algorithms, evaluate different metrics, add more params in Cross validation
# TODO: Make specified dataset for every company, and other that consists of tweets of the CEO (or someone important for the company)
# TODO: Mine data from stocktwits
if __name__ == "__main__":
    logging.basicConfig(filename='models_results.log', encoding='UTF-8', level=logging.INFO)
    logger = logging.getLogger(__name__)

    df_trans = get_transformed()
    df_trans.show()
    print(df_trans.count())
    trainLR(df_trans, 33, 0.9, 0.1, 10)
    trainNB(df_trans, 33, 0.9, 0.1, 10)
