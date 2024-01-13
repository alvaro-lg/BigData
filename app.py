from pathlib import Path
from typing import Optional, List, Dict

from pyspark import SparkConf
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
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
        # TODO: Feeding the data to the ML model
        return df

    def run(self) -> None:
        """
            Main method for the Spark application. Runs the required functionality.
        """
        df = self.read_data(self.__in_dir)
        df = self.process_data(df)

        # TODO: Testing some ML models

        # TODO: Remove this
        df.show(10)
        df.printSchema()


if __name__ == "__main__":
    SparkApp(DATA_DIRECTORY, COLUMN_FILTER, COLUMN_SCHEME).run()
