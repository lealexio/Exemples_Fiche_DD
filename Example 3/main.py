import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, StructField, StructType, DoubleType, StringType

if __name__ == '__main__':
    # Load and verify that we can see and use the file
    spark = SparkSession.builder.appName('Example2').getOrCreate()
    df_pyspark = spark.read.option("delimiter", ";").option('header','true').option('inferSchema','true').csv('cars.csv')
    df_pyspark.printSchema()
    #########################
    df_pyspark.show()
    #########################
    # Uninteresting fields
    car = StructField("Car", StringType(), False)
    mpg = StructField("MPG", IntegerType(), False)
    displacement = StructField("Displacement", IntegerType(), False)
    model = StructField("Model", IntegerType(), False)
    origin = StructField("Origin", StringType(), False)
    # Interesting fields
    cylinders = StructField("Cylinders", IntegerType(), nullable=False)
    horsepower = StructField("Horsepower", IntegerType(), nullable=False)
    weight = StructField("Weight", IntegerType(), nullable=False)
    acceleration = StructField("Acceleration", DoubleType(), nullable=False)
    # Respect the csv file format
    schema = StructType([car, mpg, cylinders, displacement, horsepower, weight, acceleration, model, origin])
    # Read the file following the scheme
    df_pyspark = spark.read.option("delimiter", ";").option('header', 'true').option('inferSchema', 'true').schema(
        schema).csv('cars.csv')
    df_pyspark.na.drop()
    df_pyspark.printSchema()
    #########################
    df_pyspark.show()
    #########################
    # Try to predict Acceleration following the Cylinders, Horsepower and Weight variations
    vecAssembler = VectorAssembler(outputCol="Independent Feature")
    vecAssembler.setInputCols(["Cylinders", "Horsepower", "Weight"])
    output = vecAssembler.transform(df_pyspark)
    # Assemble the data
    treatedOuput = output.select("Independent Feature", "Acceleration")
    treatedOuput.show()
    #########################
    # Launch accelerations predictions
    train_data, test_data = treatedOuput.randomSplit([0.75, 0.25])
    linearReg = LinearRegression(featuresCol="Independent Feature", labelCol="Acceleration")
    linearReg = linearReg.fit(train_data)
    #########################
    # Display what MLib found side by side with the Actual Acceleration
    result = linearReg.evaluate(test_data)
    result.predictions.show()
