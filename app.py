from pathlib import Path
from typing import Optional, List, Dict

from pyspark import SparkConf
from pyspark.ml.feature import MinMaxScaler, StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import *
from pyspark.sql.types import IntegerType, StringType, BooleanType, StructField, StructType

# Constants
DATA_DIRECTORY = Path("data")
COLUMN_FILTER = [
    "ArrTime",
    "ActualElapsedTime",
    "AirTime",
    "TaxiIn",
    "Diverted",
    "CarrierDelay",
    "WeatherDelay",
    "NASDelay",
    "SecurityDelay",
    "LateAircraftDelay",
    # Columns dropped by us (NAs):
    "TailNum",
    "TaxiOut",
    "CancellationCode"
]
COLUMN_SCHEME = StructType([
    StructField("Year", IntegerType(), True),
    StructField("Month", IntegerType(), True),
    StructField("DayofMonth", IntegerType(), True),
    StructField("DayOfWeek", IntegerType(), True),
    StructField("DepTime", IntegerType(), True),
    StructField("CRSDepTime", IntegerType(), True),
    StructField("CRSArrTime", IntegerType(), True),
    StructField("UniqueCarrier", StringType(), True),
    StructField("FlightNum", IntegerType(), True),
    StructField("CRSElapsedTime", IntegerType(), True),
    StructField("ArrDelay", IntegerType(), True),
    StructField("DepDelay", IntegerType(), True),
    StructField("Origin", StringType(), True),
    StructField("Dest", StringType(), True),
    StructField("Distance", IntegerType(), True),
    StructField("Cancelled", BooleanType(), True),
])


class SparkApp:
    """
        Spark application for the Big Data curse's final assignment.
    """

    def __init__(self, input_data_dir: Optional[Path] = None, column_filter: List[str] = None,
                 scheme: StructType = None):
        """
            Initializes the Spark application and the Spark context.
        """
        # Variables construction
        self.__conf = SparkConf()
        self.__spark = SparkSession.builder \
            .appName("Spark Practical Exercise") \
            .enableHiveSupport() \
            .getOrCreate()

        # Attributes initialization
        self.__conf.setAppName("Spark Practical Work")
        self.__in_dir = input_data_dir.absolute()
        self.__column_filter: List[str] = column_filter
        self.__column_scheme: StructType = scheme

    def __del__(self):
        """
            Destroys the Spark context.
        """
        self.__spark.stop()

    def read_data(self, data_dir: Path) -> DataFrame:
        # Scanning dir
        input_files = list(str(path.absolute()) for path in data_dir.glob("*.csv.bz2"))

        # Reading files
        df = self.__preprocess_data(self.__spark.read.csv(input_files, header=True))
        return df

    def __preprocess_data(self, df: DataFrame) -> DataFrame:
        # Filter values and drop NAs
        df_filtered = df.drop(*self.__column_filter).na.drop()
        for column in df_filtered.columns:
            df_filtered = df_filtered.filter(col(column).isNotNull())

        # Cast columns to the specified data types
        for field in self.__column_scheme.fields:
            column_name = field.name
            data_type = field.dataType
            df_filtered = df_filtered.withColumn(column_name, df_filtered[column_name].cast(data_type))

        return df_filtered

    def process_data(self, df: DataFrame) -> DataFrame:
        # Create a VectorAssembler
        # Flights that have not been cancelled (cancelled == 0) and whose elapsed time is positive
        # will be the ones used, as the goal is to predict the arcategorical_columns = ["DayofWeek", "UniqueCarrier", "Origin", "Dest"]
        other_columns = ["Date", "FlightNum", "CRSElapsedTime", "DepDelay", "Distance", "Cancelled", "DepTime", "CRSDepTime", "CRSArrTime"]

        index_columns = [col + "Index" for col in categorical_columns]

        # StringIndexer
        indexer = StringIndexer(inputCols=categorical_columns, outputCols=index_columns)

        vec_columns = [col + "Vec" for col in categorical_columns]

        # OneHotEncoder
        encoder = OneHotEncoder(inputCols=index_columns, outputCols=vec_columns)

        # VectorAssembler
        num_vec_columns = other_columns + vec_columns
        assembler = VectorAssembler(inputCols=num_vec_columns, outputCol="features")

        # Normalizer
        normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)

        # All together in pipeline
        pipeline = Pipeline(stages=[indexer, encoder, assembler, normalizer])
        df = pipeline.fit(df).transform(df)
        df.printSchema()

        # TODO: Remove this
        df.show(10)
        df.printSchema()


if __name__ == "__main__":
    SparkApp(DATA_DIRECTORY, COLUMN_FILTER, COLUMN_SCHEME).run()
