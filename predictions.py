"""
Predictions.

Flow:
Two main methods: execute_predictions and model_pipeline

- model_pipeline
Generic pipeline for prediction - feature transformation, prediction and evaluation.

- execute predictions
Uses utility methods to generate features and then uses model_pipeline to perform predictions.
"""

from pyspark.ml import Pipeline
from pyspark.ml.regression import RandomForestRegressor, LinearRegression, GeneralizedLinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.window import Window
from pyspark.sql.functions import col, lag, when
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
import pyspark.sql.functions as sf
import analysis


def model_pipeline(trainingData, testData, reg_model, target_variable, features, print_mse=True):
    """Pipeline method to fit and predict model."""
    assembler = (VectorAssembler()
                 .setInputCols(features)
                 .setOutputCol("raw_features"))

    scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                            withStd=True, withMean=True)

    # Chain assembler, scaler and model in a Pipeline
    pipeline = Pipeline(stages=[assembler, scaler, reg_model])

    model = pipeline.fit(trainingData)
    predictions = model.transform(testData)

    predictions.select("prediction", target_variable, "features").show(20)

    # Select (prediction, true label) and compute test error
    evaluator = RegressionEvaluator(
        labelCol=target_variable, predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    if print_mse:
        print("Root Mean Squared Error (RMSE) on test data = %g" % rmse)

    Model = model.stages[2]
    print(Model)
    return predictions.select("prediction", target_variable, "features")


def execute_predictions(sc):
    """
    Execute all predictions for:.

    - Request / sec.
    - Session time.
    - Unique request / ip.
    """
    print('Reading log data ...')
    df_casted = analysis.process_logs(sc)

    print('Generating load prediction features ... ')
    load_df, next_minute_df = create_features_load_prediction(df_casted)
    (trainingData, testData) = load_df.randomSplit([0.7, 0.3])

    print('Load Prediction: ')
    # Load Prediction.
    load_predictions_all = execute_load_prediction(trainingData, testData)
    print('Load Prediction for next minute')
    load_predictions_next_min = execute_load_prediction(load_df, next_minute_df, False)

    # Unique Request and session time prediction.
    print('Unique Request and session time prediction: ')
    df = create_ip_features(df_casted)
    unique_req_predictions = execute_unique_requests_prediction(df)
    sess_len_predictions = execute_session_length_prediction(df)


def create_features_load_prediction(df_casted):
    """
    Append properties for previous minute.

    - Uses lag of 1 over a window partition.
    - Potentially could be extended to larger intervals. E.g New sessions in last 15 min.

    Features:
    - Percentage of sucessful requests.
    - New sessions started.
    - Requests / sec for that minute.
    - Recieved and sent bytes.
    - Average processing time for that minute.
    """
    time_df = create_time_req_features(df_casted)
    time_grp_df = create_minute_df(time_df)

    w = Window().partitionBy('day', 'hour').orderBy(col("minute"))

    time_prev_df = (time_grp_df
                    .select("day", "hour", "minute", "request_per_sec",
                            sf.lag("sucessful_requests").over(w).alias("prev_sucessful_requests"),
                            sf.lag("new_sessions").over(w).alias("prev_new_sessions"),
                            sf.lag("total_recieved").over(w).alias("prev_total_recieved"),
                            sf.lag("total_sent").over(w).alias("prev_total_sent"),
                            sf.lag("request_per_sec").over(w).alias("prev_request_per_sec"),
                            sf.lag("avg_process_time").over(w).alias("prev_avg_process_time")))

    load_df = (time_prev_df.select('request_per_sec', 'prev_sucessful_requests', 'prev_new_sessions', 'prev_total_recieved',
                                   'prev_total_sent', 'prev_request_per_sec', 'prev_avg_process_time')
               .where(col('prev_request_per_sec').isNotNull()))

    # Generate features for next minute, based on the last minute in the data.
    next_minute_df = (time_grp_df.where(
        (col('day') == 204) & (col('hour') == 2) & (col('minute') == 40))
        .withColumn('prev_sucessful_requests', col('sucessful_requests'))
        .withColumn('prev_new_sessions', col('new_sessions'))
        .withColumn('prev_total_recieved', col('total_recieved'))
        .withColumn('prev_total_sent', col('total_sent'))
        .withColumn('prev_request_per_sec', col('request_per_sec'))
        .withColumn('prev_avg_process_time', col('avg_process_time'))
        ).select('request_per_sec', 'prev_sucessful_requests', 'prev_new_sessions', 'prev_total_recieved',
                 'prev_total_sent', 'prev_request_per_sec', 'prev_avg_process_time')

    return load_df, next_minute_df


def create_time_req_features(df_casted):
    """
    Create various features columns related to requests and time.

    - Day, hour, minute: Based on timestamp.
    - Status type: If the request was sucessfull (200) or otherwise.
    - new_session: If a new session was started.
    """
    time_df = (df_casted
               .withColumn('prev_timestamp', lag(df_casted['timestamp']).over(Window.partitionBy("ip").orderBy("timestamp")))
               .withColumn('time_diff', analysis.get_time_delta(col("prev_timestamp"), col("timestamp")))
               .withColumn('new_session', when((col("time_diff") > 900), 1).otherwise(0))
               .withColumn('count_session', sf.sum(col('new_session')).over(Window.partitionBy("ip").orderBy("timestamp")))
               .withColumn('day', sf.dayofyear(col('timestamp')))
               .withColumn('hour', sf.hour(col('timestamp')))
               .withColumn('minute', sf.minute(col('timestamp')))
               .withColumn('status_type', when((col("elb_status_code") == 200), 1).otherwise(0)))

    return time_df


def create_minute_df(time_df):
    """
    Groupby to find properties for a particular minute interval.

    - Percentage of sucessful requests.
    - New sessions started.
    - Requests / sec for that minute.
    - Recieved and sent bytes.
    - Average processing time for that minute.
    """
    time_grp_df = time_df.groupby('day', 'hour', 'minute').agg(
        (sf.sum('status_type') / sf.count('request')).alias('sucessful_requests'),
        sf.sum('new_session').alias('new_sessions'),
        (sf.count('request') / 60).alias('request_per_sec'),
        sf.sum('recieved_bytes').alias('total_recieved'),
        sf.sum('sent_bytes').alias('total_sent'),
        sf.mean('backend_process_time').alias('avg_process_time'))

    return time_grp_df


def create_ip_features(df_casted):
    """
    Aggregate various features for an IP.
    
    Features:
    - Distinct user agents.    
    - Sent bytes.
    - Recieved bytes.
    - Request process time.
    - Response process time.
    - Backend process time.
    - Average session time.
    - Number of unique requests.
    """
    result = analysis.session_id(df_casted)
    session_window_df = analysis.analyze_session_time(result, write=False)

    ip_df = (session_window_df.groupby('ip').agg(
                     sf.sum('current_session_time').alias('ip_session_time'),
                     sf.countDistinct('session_id').alias('num_sessions'),
                     sf.sum('sent_bytes').alias('total_sent_bytes'),
                     sf.sum('recieved_bytes').alias('total_recieved_bytes'),
                     sf.sum('request_process_time').alias('total_request_time'),
                     sf.sum('response_process_time').alias('total_response_time'),
                     sf.sum('backend_process_time').alias('total_backend_time'),
                     sf.countDistinct('user_agent').alias('unique_user_agents'),
                     sf.countDistinct('backend').alias('unique_backends'),
                     sf.countDistinct('request').alias('unique_requests'))
                .withColumn('avg_session_time', col('ip_session_time') / col('num_sessions'))
                .orderBy(col('avg_session_time'), ascending=False))

    df = ip_df.select(
                   'unique_requests', 'total_sent_bytes',
                   'total_recieved_bytes', 'total_request_time', 'total_response_time', 'total_backend_time',
                   'unique_user_agents', 'unique_backends', 'avg_session_time').where(col('avg_session_time').isNotNull())

    return df


def execute_load_prediction(trainingData, testData, print_mse=True):
    """Execute pipeline for predicting request / sec."""
    # Split the data into training and test sets
    target = "request_per_sec"
    reg_model = LinearRegression(labelCol=target, featuresCol="features", elasticNetParam=0.8, maxIter=10, regParam=0.3)

    return model_pipeline(trainingData, testData, reg_model, target, trainingData.columns[1:], print_mse)


def execute_unique_requests_prediction(df):
    # Split the data into training and test sets
    df = df.drop('avg_session_time')
    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    target = "unique_requests"
    reg_model = LinearRegression(labelCol=target, featuresCol="features", elasticNetParam=0.8, maxIter=10, regParam=0.3)

    return model_pipeline(trainingData, testData, reg_model, target, df.columns[1:])


def execute_session_length_prediction(df):
    # Split the data into training and test sets
    df = df.select(
                   'avg_session_time', 'total_sent_bytes',
                   'total_recieved_bytes', 'total_request_time', 'total_response_time', 'total_backend_time',
                   'unique_user_agents', 'unique_backends').where(col('avg_session_time').isNotNull())

    (trainingData, testData) = df.randomSplit([0.7, 0.3])
    target = "avg_session_time"
    reg_model = LinearRegression(labelCol=target, featuresCol="features", elasticNetParam=0.8, maxIter=10, regParam=0.3)

    return model_pipeline(trainingData, testData, reg_model, target, df.columns[1:])


if __name__ == "__main__":
    conf = SparkConf().setAppName('WeblogPredictions').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sc.setLogLevel('OFF')
    sqlContext = SQLContext(sc)

    execute_predictions(sc)
