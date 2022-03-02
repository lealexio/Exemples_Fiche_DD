from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#
# .appName : Sets a name for the application, which will be shown in the Spark web UI.
# .getOrCreate() : Create SparkSession based on the options set in this builder.
#
spark = SparkSession.builder.appName("Example of classification").getOrCreate()

# Read a csv file into Spark DataFrame
# inferSchema = If True, type of column is automatically detected
# header = Read the first line of the CSV file as column names.
data_frame = spark.read.csv(r"bank_account.csv", inferSchema=True, header=True)

print("\nDataFrame :")

data_frame.show()

print("\nDataFrame Schema :")

# Prints out the schema in the tree format
# It show columns with associated type
data_frame.printSchema()

print("\nPrint 5 first lines transposed :")
# Get they 5 first lines of dataset and transpose it
print(pd.DataFrame(data_frame.take(5), columns=data_frame.columns).T)

print("\nCount Deposit values :")

# Count nb of deposits values
data_frame.groupBy("deposit").count().show()

# Get numeric columns
numericCols = [feature for feature, dtype in data_frame.dtypes if dtype == 'int']
print("\nNumeric Columns :", numericCols)

print("\nDescription of numeric variables of current Dataset :")

print(data_frame.select(*[numericCols]).describe().toPandas().T)

print("\nCompute pairwise correlation of columns :")

# Statistics on variables
numeric_data = data_frame.select(*numericCols).toPandas()

# Compute pairwise correlation of columns, excluding NA/null values.
print(abs(numeric_data.corr()))

print("\nPlot pairwise relationships in a dataset")
# Show correlations between independent variables
seaborn.pairplot(numeric_data)
plt.show()
plt.close()

print("\nPlot rectangular data as a color-encoded matrix, it show  correlations between independent variables")
# Heatmap show correlations between independent variables
seaborn.heatmap(abs(numeric_data.corr()))
plt.show()
plt.close()

print("")
# No highly correlated numerical variables.
# Therefore, we will use them all in the model.
# However, the day and month columns are not very useful, we will remove both columns.

data_frame = data_frame.select(*data_frame.columns)
cols = data_frame.columns

data_frame.printSchema()

print("\nColumns", data_frame.columns)


print("\nData Preprocessing ")
# Code from DataBrick
# Indexes each categorical column using StringIndexer, then converts the indexed categories into coded variables in one shot.
# The resulting output has the binary vectors added to the end of each row. We again use the StringIndexer to encode our labels into label indices.
categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol=categoricalCol, outputCol=categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol='deposit', outputCol='label')
stages += [label_stringIdx]
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols

# The VectorAssembler combines all the feature columns into a single vector column.
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

print("\nSpark Pipeline ")
# Building a machine learning pipeline with Spark
# A machine learning pipeline is a complete workflow combining multiple machine learning algorithms.
# There may be many steps required to process and learn from the data, requiring a sequence of algorithms.
pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(data_frame)
df = pipelineModel.transform(data_frame)
print("Pipeline Dataframe columns : ", df.columns)
selectedCols = ['label', 'features'] + data_frame.columns
df = df.select(selectedCols)
df.printSchema()

print("\nShow label and features ")
df.select("label", "features").show()

print("\nSplit dataset randomly")
# Randomly splits this DataFrame with the provided weights.
# seed=The seed for sampling
train, test = df.randomSplit([0.7, 0.3], seed=2018)
print("Dataset Count: " + str(df.count()))
print("Training Dataset Count: " + str(train.count()))
print("Test Dataset Count: " + str(test.count()))

print("\nInitialize Binary Classification Evaluator")
evaluator = BinaryClassificationEvaluator()

print("\nRandom-Forest Classifier ")
rf = RandomForestClassifier(featuresCol='features', labelCol='label' )
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)

print("Random-Forest Classifier Dataset :")
print(predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas())

rf_prd = evaluator.evaluate(predictions_rf, {evaluator.metricName: "areaUnderROC"})
print("Random-Forest Classifier score :", rf_prd)

print("\nDecisionTree Classifier ")
rf = DecisionTreeClassifier(featuresCol='features', labelCol='label' )
rfModel = rf.fit(train)
predictions_rf = rfModel.transform(test)

print("DecisionTree Classifier Dataset :")
print(predictions_rf.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas())

rf_prd = evaluator.evaluate(predictions_rf, {evaluator.metricName: "areaUnderROC"})
print("DecisionTree Classifier score :", rf_prd)


print("\nGradient-Boosted Tree Classifier ")
# We use GBTClassifier as as our Machine Learning Model
gbt = GBTClassifier(maxIter=100, featuresCol='features', labelCol='label')
gbt_model = gbt.fit(train)
gbt_predictions = gbt_model.transform(test)
print("Gradient-Boosted Tree Classifier Dataset :")
print(gbt_predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').toPandas())

print("Gradient-Boosted Tree Classifier Score: ", evaluator.evaluate(gbt_predictions, {evaluator.metricName: "areaUnderROC"}))

paramGrid = (ParamGridBuilder().addGrid(gbt.maxDepth, [2, 4, 6]).addGrid(gbt.maxBins, [20, 60]).addGrid(gbt.maxIter, [10, 20]).build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
cv_model = cv.fit(train)
final_predictions = cv_model.transform(test)
cv_model_score = evaluator.evaluate(final_predictions, params={evaluator.metricName: "areaUnderROC"})
print('Cross Validation for Gradient-Boosted Tree Model Score: ', cv_model_score)
