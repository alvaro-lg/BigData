from pathlib import Path
from typing import Optional, List, Dict

from pyspark import SparkConf
from pyspark.ml.feature import MinMaxScaler, StringIndexer, OneHotEncoder, VectorAssembler, Normalizer
from pyspark.ml.clustering import KMeans
from pyspark.ml.pipeline import Pipeline
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, StringType, BooleanType, StructField, StructType
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# Constants
DATA_DIRECTORY = Path("data")
TARGET_VARIABLE = "ArrDelay"
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
    "CancellationCode",
    "FlightNum"  # Uninformative
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

    def __init__(self, input_data_dir: Path, target_variable: str, column_filter: List[str] = None,
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
        self.__target_variable = target_variable

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
        df_filtered = df.drop(*self.__column_filter)
        df_filtered = df_filtered.na.drop(subset=df_filtered.columns)
        for column in df_filtered.columns:
            df_filtered = df_filtered.filter(col(column).isNotNull())

        # Cast columns to the specified data types
        for field in self.__column_scheme.fields:
            column_name = field.name
            data_type = field.dataType
            df_filtered = df_filtered.withColumn(column_name, df_filtered[column_name].cast(data_type))

        # Drop NAs from casting
        df_filtered = df_filtered.na.drop(subset=df_filtered.columns)

        # Debugging
        print("Head of cleaned data:")
        df_filtered.show(10)

        return df_filtered

    def process_data(self, df: DataFrame) -> List[DataFrame]:
        # Encoding categorical variables
        categorical_int_columns = ["Month", "DayofMonth", "DayOfWeek"]
        categorical_str_columns = ["UniqueCarrier", "Origin", "Dest"]
        categorical_bool_columns = ["Cancelled"]

        # Processing other variables, we will normalize them
        other_columns = [col_ for col_ in df.columns if col_ not in categorical_int_columns + categorical_str_columns +
                         categorical_bool_columns and col_ != self.__target_variable]

        # StringIndexer for string categorical columns
        categorical_stridx_columns = [col_ + "Index" for col_ in categorical_str_columns]
        string_indexers = ([StringIndexer(inputCol=col_in, outputCol=col_out)
                           for col_in, col_out in zip(categorical_str_columns, categorical_stridx_columns)])

        # OneHotEncoder for both integer and string categorical columns
        encoder = OneHotEncoder(inputCols=categorical_int_columns + categorical_stridx_columns,
                                outputCols=[col_ + "Cat" for col_ in categorical_int_columns] +
                                           [col_ + "Cat" for col_ in categorical_str_columns])

        # Processing other variables, normalize them using MinMaxScaler
        assembler = VectorAssembler(inputCols=other_columns, outputCol="features")
        scaler = MinMaxScaler(inputCol="features", outputCol="scaledFeatures")

        # Cast boolean column to integer
        df = df.withColumn("CancelledCat", col("Cancelled").cast("int"))

        # Join the processed variables into a single column
        assembler_final = VectorAssembler(inputCols=["scaledFeatures"] +
                                                    [col_ + "Cat" for col_ in categorical_int_columns] +
                                                    [col_ + "Cat" for col_ in categorical_str_columns] +
                                                    ["CancelledCat"],
                                          outputCol="finalFeatures")

        # Forming the actual Pipeline and executing it
        pipeline = Pipeline(stages=string_indexers + [encoder, assembler, scaler, assembler_final])
        df = pipeline.fit(df).transform(df)

        print("Head of processed data:")
        df.show(10)

        # Drop all the columns except the finalFeatures and the target variable
        df = df.drop(*[col_ for col_ in df.columns if col_ not in {"finalFeatures", self.__target_variable}])

        # Debugging
        print("Head of processed data (for feeding the model):")
        df.show(10)

        # Returning the split DataFrame
        return df.randomSplit([0.9, 0.1], seed=2024)

    def run(self) -> None:
        """
            Main method for the Spark application. Runs the required functionality.
        """
        df = self.read_data(self.__in_dir)
        df_train, df_test = self.process_data(df)

        # Debugging
        print(f'Train/Test dataset count: {df_train.count()}/{df_test.count()}')

        rf = RandomForestRegressor(featuresCol='finalFeatures', labelCol=self.__target_variable)
        rf_model = rf.fit(df_train)

        train_predictions = rf_model.transform(df_train)
        test_predictions = rf_model.transform(df_test)

        evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="ArrTime", metricName="r2")

        print("Train R2:", evaluator.evaluate(train_predictions))
        print("Test R2:", evaluator.evaluate(test_predictions))


if __name__ == "__main__":
    SparkApp(DATA_DIRECTORY, TARGET_VARIABLE, COLUMN_FILTER, COLUMN_SCHEME).run()
