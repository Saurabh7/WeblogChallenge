"""
Analysis of data.

Flow:
- execute method performs the whole analysis:
    - Parsing: process_logs.
    - Create session ids: session_id.
    - Analyze session time.
    - Count distince requests.
    - Find longest times.
"""

import re
import datetime
import pyspark.sql.functions as sf
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext
from pyspark.sql import Row
from pyspark.sql.functions import split, regexp_extract
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import col, when
from pyspark.sql.window import Window
from pyspark.sql.functions import col, datediff, lag, mean, min, max
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType, IntegerType, StructType, StructField, FloatType
from datetime import datetime

LOG_PATTERN = '^(\\S+) (\\S+) (\\S+):(\\S+) (\\S+):(\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) (\\S+) \"([^\"]*) ([^\"]*) ([^\"]*)\" \"([^\"]*)\" (\\S+) (\\S+)$'

logFile = "./data/2015_07_22_mktplace_shop_web_log_sample.log"


def parse_log_line(logline):
    """
    Parse a line in the format

    Args:
        logline (str): a line of text in the elb Log format
    """
    match = re.search(LOG_PATTERN, logline)
    if match is None:
        return (logline, 0)

    return (Row(
        time = match.group(1),
        elb = match.group(2),
        ip = match.group(3),
        ip_port = match.group(4),
        backend = (match.group(5)),
        backend_port = (match.group(6)),
        request_process_time = match.group(7),
        backend_process_time = match.group(8),
        response_process_time = match.group(9),
        elb_status_code = int(match.group(10)),
        backend_status_code = int(match.group(11)),
        recieved_bytes = int(match.group(12)),
        sent_bytes = int(match.group(13)),
        http_method = (match.group(14)),
        request = (match.group(15)),
        http_version = (match.group(16)),
        user_agent = match.group(17),
        ssl_cipher = match.group(18),
        ssl_protocol = match.group(19)
    ), 1)


def process_logs(sc):
    parsed_logs = (sc.textFile(logFile)
                     .map(parse_log_line)
                     .cache())

    # Filter out failed rows
    access_logs = (parsed_logs
                   .filter(lambda s: s[1] == 1)
                   .map(lambda s: s[0])
                   .cache())

    # Convert to pyspark dataframe object and cast to a datetime column.
    df_obj = access_logs.toDF()

    df_casted = df_obj.select("*",
                              col("time").cast("timestamp").alias("timestamp"))

    return df_casted


def time_delta(start, end):
    """Get time difference between in seconds."""
    try:
        delta = (end - start).total_seconds()
    except:
        delta = 0
    return delta
get_time_delta = udf(time_delta, FloatType())


def create_session_id(ip, increment):
    """
    Create a unique session id for a session.

    Using the IP and number of session.
    """
    session_id = str(ip) + '_' + str(increment)
    return session_id
create_sid = udf(create_session_id, StringType())


def session_id(df_casted):
    """
    Find start of new sessions and create session ids.

    - Partition for an ip.
    - Compute time difference between consecutive requests to figure out when a new session starts. (> 15 min)
    - Count number of session and create a session id using it.
    """
    ip_window = Window.partitionBy("ip").orderBy("timestamp")

    result = (df_casted
              .withColumn('prev_timestamp', lag(df_casted['timestamp']).over(ip_window))
              .withColumn('time_diff', get_time_delta(col("prev_timestamp"), col("timestamp")))
              .withColumn('new_session', when((col("time_diff") > 900), 1).otherwise(0))
              .withColumn('count_session', sf.sum(col('new_session')).over(ip_window)))

    result = result.withColumn('session_id', create_sid(result.ip, result.count_session))

    return result


def analyze_session_time(result, write=True):
    """
    Compute total session times and overall average session time.

    - Compute time difference between requests INSIDE a session. (current_session_time)
    - Groupby the session ID and sum over current_session_time to get the total session time.
    """
    session_window = Window.partitionBy("session_id").orderBy("timestamp")

    session_window_df = (result
                         .withColumn('prev_timestamp_session', lag(result['timestamp']).over(session_window))
                         .withColumn('current_session_time', get_time_delta(col("prev_timestamp_session"), col("timestamp"))))

    session_time_df = session_window_df.groupby('session_id').agg(sf.sum('current_session_time').alias('total_session_time'))

    # Average session time.
    avg_time_df = session_time_df.select([mean('total_session_time').alias('avg_session_time')])
    avg_time_df.show()

    #if write:
    #    avg_time_df.coalesce(1).write.format('com.databricks.spark.csv').options(header='true').save('avg_session_time.csv')

    return session_window_df


def count_distinct_request(session_window_df):
    """Count distinct requests per session."""
    unique_req_df = session_window_df.groupby('session_id').agg(sf.countDistinct('request').alias('unique_requests'))
    unique_req_df.show()


def longest_session_time(session_window_df):
    """
    Find IPs with longest session times.

    - Groupby IP and compute average session time.
    """
    ip_df = (session_window_df.groupby('ip')
             .agg(
        sf.sum('current_session_time').alias('ip_session_time'),
        sf.countDistinct('session_id').alias('num_sessions'))
        .withColumn('avg_session_time', col('ip_session_time') / col('num_sessions'))
        .orderBy(col('avg_session_time'), ascending=False))

    ip_df.show()


def execute(sc):
    """Execute analysis flow."""
    df_casted = process_logs(sc)
    result = session_id(df_casted)
    session_window_df = analyze_session_time(result)
    count_distinct_request(session_window_df)
    longest_session_time(session_window_df)

if __name__ == "__main__":
    conf = SparkConf().setAppName('WeblogAnalysis').setMaster('local[*]')
    sc = SparkContext(conf=conf)
    sqlContext = SQLContext(sc)

    execute(sc)
