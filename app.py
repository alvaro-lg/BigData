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
        # will be the ones used, as the goal is to predict the arrival time
        df = df.filter((df["Cancelled"] == False) & (df['CRSElapsedTime'] > 0)).distinct() #6444006 rows without distinct() and 6443924 with distinct(). ???

        # DayofMonth, Month and Year could be an unique field called Date with 'dd/MM/yyyy' format
        df = df.withColumn("Date", concat(col("DayofMonth"), lit("/"), col("Month"), lit("/"), col("Year"))).drop("DayofMonth").drop("Month").drop("Year")

        # DayofWeek could be clearer if instead of using number it uses the names of the week
        df = df.withColumn("DayofWeek", 
                           when(col("DayofWeek") == 1, "Monday")
                           .when(col("DayofWeek") == 2, "Tuesday")
                           .when(col("DayofWeek") == 3, "Wednesday")
                           .when(col("DayofWeek") == 4, "Thursday")
                           .when(col("DayofWeek") == 5, "Friday")
                           .when(col("DayofWeek") == 6, "Saturday")
                           .when(col("DayofWeek") == 7, "Sunday"))
        
        # DepTime, CRSDepTime, CRSArrTime could be clearer if insted of using a number or hour it uses Time of Day
        # It is useful to prevent overfitting, enhance model interpretability and helpful for some models
        df = df.withColumn("DepTime",
                           when((col("DepTime") >= 1800) & (col("DepTime") <= 2359), "Evening")
                           .when((col("DepTime") >= 1200) & (col("DepTime") < 1800), "Afternoon")
                           .when((col("DepTime") >= 600) & (col("DepTime") < 1200), "Morning")
                           .when((col("DepTime") >= 0) & (col("DepTime") < 600), "Overnight")
                           .otherwise("Unknown"))
        
        df = df.withColumn("CRSDepTime",
                    when((col("CRSDepTime") >= 1800) & (col("CRSDepTime") <= 2359), "Evening")
                    .when((col("CRSDepTime") >= 1200) & (col("CRSDepTime") < 1800), "Afternoon")
                    .when((col("CRSDepTime") >= 600) & (col("CRSDepTime") < 1200), "Morning")
                    .when((col("CRSDepTime") >= 0) & (col("CRSDepTime") < 600), "Overnight")
                    .otherwise("Unknown"))
        
        df = df.withColumn("CRSArrTime",
                           when((col("CRSArrTime") >= 1800) & (col("CRSArrTime") <= 2359), "Evening")
                           .when((col("CRSArrTime") >= 1200) & (col("CRSArrTime") < 1800), "Afternoon")
                           .when((col("CRSArrTime") >= 600) & (col("CRSArrTime") < 1200), "Morning")
                           .when((col("CRSArrTime") >= 0) & (col("CRSArrTime") < 600), "Overnight")
                           .otherwise("Unknown"))
        
        # TODO: Missing values
        return df

    def run(self) -> None:
        """
            Main method for the Spark application. Runs the required functionality.
        """
        df = self.read_data(self.__in_dir)
        df = self.process_data(df)

        # TODO: Testing some ML models
        categorical_columns = ["DayofWeek", "DepTime", "CRSDepTime", "CRSDepTime", "UniqueCarrier", "Origin", "Dest"]
        indexed_columns = [col + "_index" for col in categorical_columns]
        encoded_columns = ["encoded_" + col for col in categorical_columns]
        other_columns = ["Date", "FlightNum", "CRSElapsedTime", "DepDelay", "Distance", "Cancelled"]
        assembler_columns = encoded_columns + other_columns
        
        # Create a StringIndexer to convert the categorical column to indices
        indexers = [StringIndexer(inputCol=col, outputCol=indexed_columns) for col, indexed_col in categorical_columns]
        # # Create a OneHotEncoder to encode the indices into one-hot vectors
        # encoder = OneHotEncoder(inputCol=indexed_columns, outputCol=encoded_columns)
        
        # # Create a VectorAssembler
        # assembler = VectorAssembler(inputCols=assembler_columns, outputCol="features")
        # dataset = assembler.transform(df)

        # # Create a Normalizer
        # normalizer = Normalizer(inputCol="features", outputCol="normFeatures", p=1.0)

        # # Create K-Means model
        # kmeans = KMeans(featuresCol="features", k=2)

        # # Create a Pipeline
        # pipeline = Pipeline(stages=[assembler, normalizer, kmeans])

        # # Fit the Pipeline on the training data
        # model = pipeline.fit(training)

        # # Transform the test data and show the results
        # transformed_data = model.transform(test)
        # transformed_data.show(truncate=False)

        

        # TODO: Remove this
        df.show(10)
        df.printSchema()


if __name__ == "__main__":
    SparkApp(DATA_DIRECTORY, COLUMN_FILTER, COLUMN_SCHEME).run()
