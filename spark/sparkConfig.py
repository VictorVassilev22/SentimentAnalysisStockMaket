import logging

from pyspark.sql import SparkSession
from pyspark import SparkContext

sc = SparkContext.getOrCreate()
print("Spark Streaming version:", sc.version)

spark = (SparkSession.builder
         .appName("SA-Application")
         .master("local[*]")
         .config("spark.executor.memory", "12g")
         .config("spark.driver.memory", "12g")
         .config("spark.driver.maxResultSize", "12g")
         .config("spark.memory.storageFraction", "0.5")
         .config("spark.sql.shuffle.partitions", "5")
         .config("spark.reducer.maxSizeInFlight", "96m")
         .config("spark.shuffle.file.buffer", "512k")
         .config("spark.sql.debug.maxToStringFields", "100")
         .config("spark.memory.offHeap.enabled", True)
         .config("spark.memory.offHeap.size", "16g")
         .getOrCreate())

# spark.sparkContext.setLogLevel("OFF")