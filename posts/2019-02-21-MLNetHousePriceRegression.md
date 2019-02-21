ML.NET is an opensource cross-platform machine learning framework intended for .NET developers. It provides a great set of tools to let you implement machine learning applications using .NET – you can find out more about ML.NET here

To understand how the functionality fits into the typical workflow of accessing data, selecting features, normalisation, training the model and evaluating the fit using test data sets. I took a look at implementing a simple regression application to predict the sale price of a house given a simple set of features over about 800 home sales.

The focus was on getting a small sample up and running, that can then be used to experiment with the choice of feature and training algorithms. You can find the code for this article on GitHub here

Loading Data
My starting point was a .csv file( see HouseDataExtended3Anon.csv in the sample code) with different features for 800 sales from a friendly realtor – so the first thing I would want to do is to load that into my app.

A strong point for ML.NET is that there are plenty of ways that a .NET developer already knows to pull data in and especially easy access to data held in corporate databases.

In this case, we will use a TextLoader to define the features we want to work with in terms of their names, type and position in the .csv file.

Its easy to change the features, so makes experimenting with different features to understand their effect of training and be aware of over and under fitting.

_textLoader = mlContext.Data.CreateTextLoader(new NextLoader.Arguments()
{
Separators = new[] { ‘,’ },
HasHeader = true,
Column = new[]
{
new TextLoader.Column(“Area”, DataKind.Text, 3),
new TextLoader.Column(“Rooms”, DataKind.R4, 4),
new TextLoader.Column(“BedRooms”, DataKind.R4, 13),
new TextLoader.Column(“BedRoomsBsmt”, DataKind.R4, 12),
new TextLoader.Column(“FullBath”, DataKind.R4, 5),
new TextLoader.Column(“HalfBath”, DataKind.R4, 6),
new TextLoader.Column(“Floors”, DataKind.R4, 7),
new TextLoader.Column(“LotSize”, DataKind.R4, 22),
new TextLoader.Column(“GarageType”, DataKind.Text, 16),
new TextLoader.Column(“SoldPrice”, DataKind.R4, 9)
}
});

The TextLoader can then be used to read the input file into a DataView that is used downstream for training and testing.

Pipeline to process data
To support the next stages we’ll create a Pipeline that supports the different transforms and operations that are then carried out.  In particular, you can manage:

Category data – for example, GarageTypes (Attached, detached, none), most regression trainers require numeric features.
Normalizing Features – For speed of training and accuracy, it important to normalized the numeric features so that they are all in the same range.
The trainer to be used and its parameters – ML.NET offers different trainers and that can be configured in different ways.
So the pipeline lets you organize the operations used to train and support prediction, for example, the normalization determined in the pipeline applied to the prediction.

Check out the code to see this in action.

var pipeline = mlContext.Transforms.CopyColumns(inputColumnName: “SoldPrice”, outputColumnName: “Label”)
.Append(mlContext.Transforms.Categorical.OneHotEncoding(“GarageType”))
.Append(mlContext.Transforms.Categorical.OneHotEncoding(“Area”))
.Append(mlContext.Transforms.Concatenate(“Features”, “Area”, “Rooms”, “BedRooms”, “BedRoomsBsmt”, “FullBath”, “HalfBath”, “Floors”, “GarageType”, “LotSize”))
.Append(mlContext.Transforms.Normalize(new NormalizingEstimator.MinMaxColumn(inputColumnName: “Features”, outputColumnName: “FeaturesNormalized”, fixZero: true)))
.Append(mlContext.Regression.Trainers.FastTree(featureColumn: “FeaturesNormalized”));

Once we have our pipeline set up we can get to work training and evaluating the fit, before we can do that we need to split off some data from the set provided for  testing.

For example to split the data 90% for training and 10% for a test, you use the TrainTestSplit method:

var (trainData, testData) = mlContext.Regression.TrainTestSplit(dataView, testFraction: 0.1);

Training
We now have the pipeline so armed with our training data we can create a trained model.

// Train the model.
var model = pipeline.Fit(trainData);

Evaluation
We need to understand just how well training has worked, so we can evaluate the test data and check the quality metrics:

// Compute quality metrics on the test set.
var metrics = mlContext.Regression.Evaluate(model.Transform(testData));

The following data points are available – for more info see here

L1 – Gets the absolute loss of the model.

L2  – Gets the squared loss of the model.

Rms – Gets the root mean square loss (or RMS) which is the square root of the L2 loss.

RSquared – Gets the R squared value of the model, which is also known as the coefficient of determination.

Saving the model and using it to predict a house sale price
When we are happy with the training, for example, if the RMS metric is close to 1 – we can then save the model for further use:

using (var fileStream = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
mlContext.Model.Save(model, fileStream);

To predict a price we’ll first load the model as follows:

ITransformer loadedModel;
using (var stream = new FileStream(_modelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
{
loadedModel = mlContext.Model.Load(stream);
}

Then create a helper function to run the predictions:

var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);

This uses two classes:

HouseData (to enter the features of the house we want to make the prediction for.
HousePrediction – used to return the SalePrice
You can see these data classes in the code provided.

To run the prediction:

var prediction = predictionFunction.Predict(housePriceSample);

 