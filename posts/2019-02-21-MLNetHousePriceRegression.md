## Applying ML .NET to a regression problem
ML .Net is an opensource cross-platform machine learning framework intended for .NET developers. Python(with routines are written in C++) is generally used to develop many ML libraries, e.g. TensorFlow, and this can add extra steps and hurdles when you need to tightly integrate ML components on the .Net platform. ML .Net provides a great set of tools to let you apply machine learning applications using .NET – you can find out more about ML .NET [here](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)



ML .NET provides a developer-friendly API for machine learning, in terms of:
* Transforms(Feature selection, Text, Schema, Categorical, data normalisation, handling missing data)
* Learners(Linear, Boosted trees, SVM, K-Means)
* Tools(Data framework, Evaluators, calibrators, Data Loaders)

Put together these support the typical ML workflow: 
* Data preparation(loading and feature extraction) 
* Training (Training and evaluating models) 
* Running( using the trained model)

A significant advantage of using the ML .NET framework is that it allows the user to quickly experiment with different learning algorithms, changing the set of features, sizes of training and test datasets to get the best results for their problem. This avoids a common issue where teams spend a lot of time collecting unnecessary data and produce models that do not perform well.

# Key Concept
When discussing ML .NET, it is important to recognise to use of:
* Transformers - these convert and manipulate data and produce data as an output.
* Estimators - these take data and provide a transformer or a model, e.g. when training
* Prediction - this uses a single row of features and predicts a single row of results.

![key concept]({{ site.url }}/assets/MLNETConcepts.png)

We will see how these come into play in the simple regression sample.
ML .NET lets you develop a range of ML systems
* Forecasting/Regression
* Issue Classification
* Predictive maintenance
* Image classification
* Sentiment Analysis
* Recommender Systems
* Clustering systems

# House price sample
To understand how the functionality fits into the typical workflow of data preparation, training the model and evaluating the fit using test data sets and using the model. I took a look at implementing a simple regression application to predict the sale price of a house given a simple set of features over about 800 home sales. In the sample, we are going to take a look at a supervised learning problem of Multivariate linear regression. In this case, we want to use one or more features to predict the sale price of a house. The focus was on getting a small sample up and running, that can then be used to experiment with the choice of feature and training algorithms. You can find the code for this article on GitHub here

We will train the model using a set of sales data to predict the sale price of a house given a set of features over about 800 home sales. While the sample data has a wide range of features, a key aspect of developing a useful system would be to understand the choice of features used affects the model.

# Data class
Our first job is to define a data class that we can use when loading our .csv file of house data. The important part to note is the [LoadColumn()] attributes; these allow us to associate the fields to different columns in the input. It gives us a simple way to adapt to changes in the data sets we can process. When a user wants to predict the price of a house they use the data class to give the features. Note,we do not need to use all the fields in the class when training the model.

```C#
    public class HouseData
        {
            [LoadColumn(3)]
            public string Area;
    
            [LoadColumn(4)]
            public float Rooms;
    
            [LoadColumn(13)]
            public float BedRooms;
    
            [LoadColumn(12)]
            public float BedRoomsBsmt;
    
            [LoadColumn(5)]
            public float FullBath;
    
            [LoadColumn(6)]
            public float HalfBath;
    
            [LoadColumn(7)]
            public float Floors;
    
            [LoadColumn(9)]
            public float SoldPrice;
    
            [LoadColumn(22)]
            public float LotSize;
    
            [LoadColumn(16)]
            public string GarageType;
        }
 ```
# Training and Saving the model
CreateHousePriceModelUsingPipeline(...) method does most of the interesting work in creating the model used to predict house prices.

The code snippet shows how you can:
* Read the data from a .csv file
* Choose the training algorithm
* Choose features to use when training
* Handle string features
* Create a pipeline to process the data and train the model

In these types of problem you will typically need to normalise the inbound data, consider the feature Rooms and BedRooms, the range of values of Rooms is usually larger then BedRooms, we normalise them to have the same influence on fit, and to speed up (depending on the trainer) the time to fitting. The trainer we using automatically normalises the features, though the framework provides tools to support normalizing if you need to do this yourself.

Similarly, depending on the number of features used (and if the model is overfitting) we apply Regularization to the trainer - this essentially keeps all the features but adds a weight to each of the features parameters to reduce the effect. IN our case the trainer will handle regularisation, and adjustments can be made when creating the trainer.

```C#

    Console.WriteLine("Training product forecasting");

    // Read the sample data into a view that we can use for training
    var trainingDataView = mlContext.Data.ReadFromTextFile<HouseData>(dataPath, hasHeader: true, separatorChar: ',');

    // create the trainer we will use  - ML .NET supports different training methods
     var trainer = mlContext.Regression.Trainers.FastTree(labelColumn: DefaultColumnNames.Label, featureColumn: DefaultColumnNames.Features);

    // Create the training pipeline, this determines how the input data will be transformed
    // We can also select the features we want to use here, the names used correspond to the porperty names in 
    // HouseData
    string[] numericFeatureNames = { "Area","Rooms", "BedRooms", "BedRoomsBsmt", "FullBath", "HalfBath", "Floors","LotSize"};

    // We distinguish between features that are strings e.g. {"attached","detached","none") garage types and 
    // Numeric faeature, since learning systems only work with numeric values we need to convert the strings.
    // You can see that in the training pipeline we have applied OneHotEncoding to do this.
    string[] categoryFeatureNames = { "GarageType" };
    
    var trainingPipeline = mlContext.Transforms.Concatenate(NumFeatures, numericFeatureNames)
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(CatFeatures, inputColumnName: categoryFeatureNames[0]))
        .Append(mlContext.Transforms.Concatenate(DefaultColumnNames.Features, NumFeatures, CatFeatures))
        .Append(mlContext.Transforms.CopyColumns(DefaultColumnNames.Label, inputColumnName: nameof(HouseData.SoldPrice)))
        .Append(trainer);

    // Split the data 90:10 into train and test sets, train and evaluate.
    var (trainData, testData) = mlContext.Regression.TrainTestSplit(trainingDataView, testFraction: 0.2);
```
# Evaluation
After training we need to evaluate our model using test data, this will indicate the size of the error between the predicted result and the actual results. This will be part of an iterative process on a relatively small set of data to determine the best mix of features. There are different approaches supported by ML .NET We use cross-validation to estimate the variance of the model quality from one run to another, it and also eliminates the need to extract a separate test set for evaluation. We display the quality metrics to evaluate and get the model's accuracy metrics

https://en.wikipedia.org/wiki/Cross-validation_(statistics)

Note that after the training and evaluation we save the model for prediction.

 ```C#
//  We use cross-valdiation toestimate the variance of the model quality from one run to another,
// it and also eliminates the need to extract a separate test set for evaluation.
// We display the quality metrics in order to evaluate and get the model's accuracy metrics
Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
var crossValidationResults = mlContext.Regression.CrossValidate(data: trainingDataView, estimator: trainingPipeline, numFolds: 6,                   labelColumn: DefaultColumnNames.Label);

Helpers.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

// Train the model
var model = trainingPipeline.Fit(trainingDataView);

// Save the model for later comsumption from end-user apps
using (var file = File.OpenWrite(outputModelPath))
    model.SaveTo(mlContext, file);
```


#Load and predict house sale prices
Once you have tweaked the features and evaluate different training, you can then use the model to predict sales prices. I think this where the ML .NET framework shines because we can then use the cool tools in .Net to support different ways to use the model.
```C#
 public static void PredictSinglePrice(HouseData houseData, MLContext mlContext, string dataPath, string outputModelPath = "housePriceModel.zip")
        {
            //  Load the prediction model we saved earlier
            ITransformer loadedModel;
            using (var stream = new FileStream(outputModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                loadedModel = mlContext.Model.Load(stream);
            }

            // Creete a handy function based on our HouseData class and a class to contain the result
            var predictionFunction = loadedModel.CreatePredictionEngine<HouseData, HousePrediction>(mlContext);

            // Predict the Sale price - TA DA
            var prediction = predictionFunction.Predict(houseData);

            var pv = prediction.SoldPrice;

            Console.WriteLine($"**********************************************************************");
            Console.WriteLine($"Predicted SellPrice: {pv:0.####}");
            Console.WriteLine($"**********************************************************************");
        }
    }
```
For this type of ML application, a typical use would be to create a simple REST service running in a docker container deployed to Windows Azure. A web app written in javascript consumes the service to let people quickly see what a house should sell for.

Using .Net Core we can run the backend on different hardware platforms, Visual Studio 2019, 2017 makes the creation, deployment, and management of a robust service quick and easy.