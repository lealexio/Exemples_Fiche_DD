from pyspark.context import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

if __name__ == "__main__":
    sc = SparkContext('local')
    data = sc.textFile("test.data")

    # Here we transform test.data into RDD
    ratings = data.map(lambda l: l.split(',')) \
        .map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    # Construct model using Alternating Least Squares (ALS) and the data
    rank = 10
    numIterations = 10
    model = ALS.train(ratings, rank, numIterations)

    # Get data without rate value
    testdata = ratings.map(lambda p: (p[0], p[1]))

    # Predict thanks to the model
    predictions = model.predictAll(testdata).map(lambda r: (r[0], r[1]))

    # Evaluate model on training data
    ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean()
    print("Mean Squared Error = " + str(MSE))

    # Save model
    model.save(sc, "myCollaborativeFilter")
    # Load model so we can predict on it
    sameModel = MatrixFactorizationModel.load(sc, "myCollaborativeFilter")

    # Predict part
    print(sameModel.predict(1, 4))
    print(sameModel.predict(6, 1))

    # the user 7 have rate the object 3 4 time
    print(sameModel.predict(7, 3))
    # compare if the algorithm is making an average on the user ratings item
    print((7.1 + 7.8 + 5.7 + 4.2) / 4)

    # So we are more interest in predict of items rate if user
    # did not rate it yet
    print(sameModel.predict(10, 3))
    print(sameModel.predict(10, 4))
