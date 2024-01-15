# BigData

The application comprises a single Python script, a requirements text file, and a 'data' directory. Given that the 2009 ASA Statistical Computing and Graphics Data Expo dataset is segmented into multiple files, you can add one or more dataset files to the data directory. The Python script will automatically detect and include all files located within the data directory.

Before running the app.py script, all requirement within the requirements.txt file should be installed with the line: “pip install requirements.txt”

Run the app with: "python app.py".
No other arguments are needed.

Deploy the Spark app with: "spark-submit app.py"
