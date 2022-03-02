from pyspark.sql import SparkSession
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
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

print("\nDataFrame Schema --------------------------------")

# Prints out the schema in the tree format
data_frame.printSchema()

print("\n????????? -------------------------------")
# Count deposit of each account :
print(pd.DataFrame(data_frame.take(5), columns=data_frame.columns).T)


print("\nDeposit --------------------------------")

data_frame.groupBy("deposit").count().show()

numeric_features = [feature for feature, dtype in data_frame.dtypes if dtype == 'int']
print("\nNumeric Columns :", numeric_features)

print("\nSummary statistics for numeric variables --------------------------------")

print(data_frame.select(*[numeric_features]).describe().toPandas().T)

print("\nCompute pairwise correlation of columns --------------------------------")

# Statistics on variables
numeric_data = data_frame.select(*numeric_features).toPandas()

# Compute pairwise correlation of columns, excluding NA/null values.
print(abs(numeric_data.corr()))

print("\nPlot pairwise relationships in a dataset. --------------------------------")
# Show correlations between independent variables
sns.pairplot(numeric_data)
plt.show()
plt.close()

print("Plot rectangular data as a color-encoded matrix --------------------------------")
# Heatmap show correlations between independent variables
sns.heatmap(abs(numeric_data.corr()))
plt.show()
plt.close()

print("--------------------------------")
# No highly correlated numerical variables.
# Therefore, we will use them all in the model.
# However, the day and month columns are not very useful, we will remove both columns.

data_frame = data_frame.select(*data_frame.columns)
cols = data_frame.columns

data_frame.printSchema()

print("Columns", data_frame.columns)


print("Data Preprocessing --------------------------------")

categoricalColumns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
stages = []
for categoricalCol in categoricalColumns:
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
    encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
    stages += [stringIndexer, encoder]
label_stringIdx = StringIndexer(inputCol = 'deposit', outputCol = 'label')
stages += [label_stringIdx]
numericCols = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous']
assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
stages += [assembler]

print("Pipeline --------------------------------")

pipeline = Pipeline(stages=stages)
pipelineModel = pipeline.fit(data_frame)
df = pipelineModel.transform(data_frame)
selectedCols = ['label', 'features'] + cols
df = df.select(selectedCols)
df.printSchema()

print("Show label features --------------------------------")
df.select("label", "features").show()

print("???? --------------------------------")
# Randomly splits this DataFrame with the provided weights.
# seed=The seed for sampling
train, test = df.randomSplit([0.7,0.3], seed=2018)
print(train.count())
print(test.count())

print("Gradient-Boosted Tree Classifier --------------------------------")

gbt = GBTClassifier(maxIter=10)
gbt_model = gbt.fit(train)
predictions = gbt_model.transform(test)
predictions.select("label", "prediction").show()

print("Model success evaluation --------------------------------")
# Evaluator for binary classification, which expects input columns rawPrediction, label and an optional weight column.
evaluator = BinaryClassificationEvaluator()
score = evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})
print("Model Score: ", score)

paramGrid = (ParamGridBuilder().addGrid(gbt.maxDepth, [2, 4, 6]).addGrid(gbt.maxBins, [20, 60]).addGrid(gbt.maxIter, [10, 20]).build())
cv = CrossValidator(estimator=gbt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

cv_model = cv.fit(train)
final_predictions = cv_model.transform(test)
final_score = evaluator.evaluate(final_predictions, params={evaluator.metricName: "areaUnderROC"})
print('Model Score: ', final_score)
