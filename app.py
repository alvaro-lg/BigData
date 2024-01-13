from pathlib import Path
from typing import Optional

from pyspark import SparkConf, SparkContext

DATA_PATH = Path("data/1987.csv.bz2")


class SparkApp:
    """
        Spark application for the Big Data curse's final assignment.
    """

    def __init__(self, inputFilePath: Optional[Path] = DATA_PATH):
        """
            Initializes the Spark application and the Spark context.
        """
        # Variables creation
        self.__conf = conf = SparkConf()
        self.__sc = SparkContext.getOrCreate(conf)

        # Attributes initialization
        self.__conf.setAppName("Spark Practical Work")
        self.__sc.setLogLevel("ERROR")
        self.__rdd = self.__sc.textFile(str(inputFilePath.absolute()))

    def __del__(self):
        """
            Destroys the Spark context.
        """
        self.__sc.stop()

    def run(self):
        """
            Main method for the Spark application. Runs the required functionality.
        """
        # TODO
        pass


if __name__ == "__main__":
    SparkApp().run()
