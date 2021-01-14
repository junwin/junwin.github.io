# Setting up a regression model for house prices using Keras and Tensorflow
Using Tensorflow, Keras and Python in Jupyter Notebooks is a popular way to develop machine learning applications.

To understand how the functionality fits in the machine learning workflow ( data preparation, training the model and evaluating the fit using test data sets) I decided to apply the tools to estimate the sale prices of houses.

The approach is similar to the sample I produced using ML.Net from Microsoft: https://towardsdatascience.com/predicting-a-house-price-using-ml-net-6555ff3caeb

We will run the code in Google's Colaboratory since this provides an excellent environment that allows access to GPU's, requires no configuration and enables sharing.

We need to sanity test the raw data we loaded to ensure the columns are in the date range we expect.  It is a good idea to check the data for any outliers; for example, in my first iteration, one home had 82 garage spots.

Be aware that there is a regional bias in the data since all the test data came from a few areas of Chicago's northern suburbs.


## Dependancies
First lets load any dependancies required.


```python
#@title Dependancies
import pandas as pd
! pip install tensorflow==2.4.0
# TensorFlow is an open source machine learning library
import tensorflow as tf

# Keras is TensorFlow's high-level API for deep learning
from tensorflow import keras
from tensorflow.keras import regularizers
# Numpy is a math library
import numpy as np
# Pandas is a data manipulation library 
import pandas as pd
# Matplotlib is a graphing library
import matplotlib.pyplot as plt
# Math is Python's math library
import math
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler,StandardScaler
```

## Let's load a dataset of house sales. 
We need to sanity test the raw data we loaded to ensure the columns are in the date range we expect. It is a good idea to check the data for any outliers; for example, in my first iteration, one home had 82 garage spots. Be aware that there is a regional bias in the data since all the test data came from a few areas of Chicago's northern suburbs.



```python
url="https://junwin.github.io/HouseData3.csv"
rawData=pd.read_csv(url).sample(frac=1)
rawData.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>MLS#</th>
      <th>YearClosed</th>
      <th>SoldPr</th>
      <th>Locale</th>
      <th>Zip</th>
      <th>Area</th>
      <th>Rooms</th>
      <th>FullBaths</th>
      <th>HalfBaths</th>
      <th>Beds</th>
      <th>BsmtBeds</th>
      <th>GarageSpaces</th>
      <th>ParkingSpaces</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>9.605000e+03</td>
      <td>9605.000000</td>
      <td>9.605000e+03</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
      <td>9605.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.037041e+07</td>
      <td>2019.130765</td>
      <td>4.643093e+05</td>
      <td>78.721603</td>
      <td>60079.478605</td>
      <td>2222.006767</td>
      <td>7.968974</td>
      <td>2.225299</td>
      <td>0.549922</td>
      <td>3.222957</td>
      <td>0.144300</td>
      <td>1.660718</td>
      <td>0.409787</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.808318e+05</td>
      <td>0.948912</td>
      <td>3.246592e+05</td>
      <td>59.308417</td>
      <td>67.705670</td>
      <td>1243.632265</td>
      <td>2.500578</td>
      <td>0.987977</td>
      <td>0.578789</td>
      <td>1.033955</td>
      <td>0.388828</td>
      <td>1.342160</td>
      <td>3.543110</td>
    </tr>
    <tr>
      <th>min</th>
      <td>8.866215e+06</td>
      <td>2017.000000</td>
      <td>2.500000e+04</td>
      <td>2.000000</td>
      <td>60002.000000</td>
      <td>372.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.013837e+07</td>
      <td>2019.000000</td>
      <td>2.600000e+05</td>
      <td>53.000000</td>
      <td>60053.000000</td>
      <td>1344.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.043821e+07</td>
      <td>2019.000000</td>
      <td>3.650000e+05</td>
      <td>62.000000</td>
      <td>60062.000000</td>
      <td>1868.000000</td>
      <td>8.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.067198e+07</td>
      <td>2020.000000</td>
      <td>5.750000e+05</td>
      <td>76.000000</td>
      <td>60077.000000</td>
      <td>2750.000000</td>
      <td>9.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.095755e+07</td>
      <td>2021.000000</td>
      <td>4.300000e+06</td>
      <td>201.000000</td>
      <td>63104.000000</td>
      <td>17365.000000</td>
      <td>18.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>79.000000</td>
      <td>308.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Features
Let's grab the features we want to model; the good thing is that its relatively easy to experiment with different features using pandas.


```python
selectedFeatures = rawData[['YearClosed', 'Type', 'Area', 'Zip', 'Rooms','FullBaths','Beds','GarageSpaces']]
prices = rawData['SoldPr']
SAMPLES = len(selectedFeatures.index)
featureCount = len(selectedFeatures.columns)

```

### Converting enumerations
Before we can use the data we need to convert any features like type from string-based labels(e.g. Condo, Townhouse, Duplex ) to a numeric, we can use the sklearn.preprocessing tools for preprocessing and normalization.

You can use the ordinal encoder in "auto" mode to identify categories in the training data - here I have specified the columns of interest.

You need to ensure consitent types(e.g. strings, int, float) used in the columns to be converted.


```python
ordinalColumns = ['YearClosed', 'Type', 'Zip']
ordinalData = selectedFeatures[ordinalColumns]
enc = preprocessing.OrdinalEncoder()
enc.fit(ordinalData)
enc.categories

selectedFeatures[ordinalColumns] = enc.transform(ordinalData)

```

### Normalization
We should normalize the features and corresponding price targets to facilitate learning.


```python
featureScaler = MinMaxScaler()
selectedFeaturesScale =  pd.DataFrame(featureScaler.fit_transform(selectedFeatures), columns=selectedFeatures.columns)

priceScaler = MinMaxScaler()
pricesScale =  pd.DataFrame(priceScaler.fit_transform(prices.values.reshape(-1, 1)),columns=['SoldPr'])

```

## Saving the Sclaers and Encoders
You will need to save the scalers and encoders if you want to use the model on its own without training.


```python
# save and load your scalers and encoders
import joblib 
joblib.dump(featureScaler, 'my_cool_scaler.pkl')
joblib.dump(priceScaler, 'my_cool_pxscaler.pkl')
joblib.dump(enc, 'my_cool_encoder.pkl')  
#see joblib.load
```




    ['my_cool_encoder.pkl']



### Understanding the input data
It is imperative to have a good overview of the model's data, so you need to plot some of the features to check any issues. Suppose you look at the first plot of the area against price:
* there are many issues as the area increases. 
* most of the data is for smaller homes.

I am not going to address these issues here, but the input merits some scrutiny.



```python
# Plot our data to examine the relationship between some features and price
plt.plot(selectedFeaturesScale['Area'], pricesScale, 'b.')
plt.show()
plt.plot(selectedFeaturesScale['Rooms'], pricesScale, 'b.')
plt.show()
plt.plot(selectedFeaturesScale['Zip'], pricesScale, 'b.')
plt.show()
```


    
![png](output_15_0.png)
    



    
![png](output_15_1.png)
    



    
![png](output_15_2.png)
    


## Separate data
We need to split our data into three parts, the first 60% is for training, 20% are set asside for validation and the last 20% for a final test.


```python
TRAIN_SPLIT =  int(0.6 * SAMPLES)
TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

training_examples = selectedFeaturesScale.head(TRAIN_SPLIT)
training_targets = pricesScale.head(TRAIN_SPLIT)

validation_examples = selectedFeaturesScale[TEST_SPLIT:TEST_SPLIT+int(0.2 * SAMPLES)]
validation_targets = pricesScale[TEST_SPLIT:TEST_SPLIT+int(0.2 * SAMPLES)]

test_examples = selectedFeaturesScale.tail(int(0.2 * SAMPLES))
test_targets = pricesScale.tail(int(0.2 * SAMPLES))

# Double-check that we've done the right thing.
print("Training examples summary:")
training_examples.describe()
print("Validation examples summary:")
validation_examples.describe()

print("Training targets summary:")
training_targets.describe()
print("Validation targets summary:")
validation_targets.describe()

```

## Training a model using Keras
Now we have our data we can use Keras (a high-level API to Tensorflow) to create a model.
 Keras makes it easy to experiment with different model architectures and visualize the results.

We will begin with three layers and use the "relu" activation function. Notice that the first layer uses the featureCount to define its input shape, and the final layer outputs to a single neuron since its the predicted price.

If you think the model is overfitting you can easily add regularization to manage the problem - these are commented out below.



```python
model_1 = tf.keras.Sequential()

model_1.add(keras.layers.Dense(16, activation='relu', input_shape=(featureCount,)))
#model_1.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(featureCount,)))
model_1.add(keras.layers.Dense(16, activation='relu'))
#model_1.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
model_1.add(keras.layers.Dense(1))

# Compile the model using the standard 'adam' optimizer and the mean squared error or 'mse' loss function for regression.
model_1.compile(optimizer='adam', loss='mse', metrics=['mae'])
```

### Having created the model, we will now train it.


```python
# Train the model on our training data while validating on our validation set
history_1 = model_1.fit(training_examples, training_targets, epochs=500, batch_size=50,
                        validation_data=(validation_examples, validation_targets))
```

### Understanding the results
Having trained our model, we now need to check the training metrics to see how the model converges. First, we will check out training over all the epochs, then show a graph that excludes some of the initial epochs to focus on whats happening nearer the end.

We can see a very rapid convergance, where the loss flatens out after about 50 epochs. On the second plot, it looks like a reasonable difference between the training and validation loss.



```python
# Draw a graph of the loss, which is the distance between
# the predicted and actual values during training and validation.
train_loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'g.', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Exclude the first few epochs so the graph is easier to read
SKIP = 50

plt.plot(epochs[SKIP:], train_loss[SKIP:], 'g.', label='Training loss')
plt.plot(epochs[SKIP:], val_loss[SKIP:], 'b.', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```


    
![png](output_23_0.png)
    



    
![png](output_23_1.png)
    


### Final test
We can now use the 20% of the data we kept aside for a final test. First, we use the model to evaluate the test data and print out the mean squared error and the mean average error for predictions. The overall loss on the test data seem to correlate well with the results seen in tthe validation set.

It's a great idea to visualize some of the data by plotting actuals against predicted results.


```python
# Calculate and print the loss on our test dataset
test_loss, test_mae = model_1.evaluate(test_examples, test_targets)
print("mean squared error:", test_loss, " mean average error:", test_mae)

#y_test_pred = model_1.predict(test_examples)

# Graph the predictions against the actual values
#plt.clf()
#plt.title('Comparison of predictions and actual values')
#plt.plot( test_targets,  y_test_pred, 'b.', label='Actual values')
#plt.show()

```

    61/61 [==============================] - 0s 939us/step - loss: 0.0012 - mae: 0.0213
    mean squared error: 0.0012115772115066648  mean average error: 0.02132599614560604
    

## Plot a comparision of actual and predicted for a single zip
Plotting the actual values against the predcited shows reasonable results for this particular training run.


```python
y_test_pred = model_1.predict(test_examples)
```


```python
sampleInfo = pd.DataFrame();
samplePx = pd.DataFrame();

i=0
while i < test_examples.values.shape[0]:
  myRow =  test_examples.iloc[[i]]
  myPxRow = test_targets.iloc[[i]]

  if round(myRow['Zip'].values[0],2) == 0:
    sampleInfo = sampleInfo.append(myRow)
    samplePx = samplePx.append(myPxRow)
    
  i = i + 1

y_test_pred = model_1.predict(sampleInfo)

plt.clf()
plt.title('Comparison of predictions and actual values - single zip')
plt.plot( sampleInfo['Area'], y_test_pred, 'b.', label='predicted values')
plt.plot( sampleInfo['Area'], samplePx, 'g.', label='Actual values')
#plt.plot( samplePx,  y_test_pred, 'b.', label='Actual values')
plt.show()
```


    
![png](output_28_0.png)
    


## Run your own examples
Lets try with a couple of test inferences  - the single family home (SFH) should be lower in zip 60002 than 60076


```python
houseData = {'YearClosed': [2020.00, 2019.00, 2019.00, 2019.00, 2019],
	'Type': ['SFH', 'SFH', 'SFH', 'Condo', 'Townhouse'],
	'Area': [2940, 1500, 1500, 1500, 1500],
  'Zip': [60002, 60002, 60076, 60076, 60076],
	'Rooms': [9, 7, 7, 7, 7],
  'FullBaths': [2.5, 2.5, 2.5, 2.5, 2.5],
  'Beds': [4, 3, 3, 3, 3],
  'GarageSpaces': [2, 2, 2, 0, 0]  }

houseInfo = pd.DataFrame(houseData)
ordinalData = houseInfo[ordinalColumns]
houseInfo[ordinalColumns] = enc.transform(ordinalData)


houseInfo =  pd.DataFrame(featureScaler.transform(houseInfo), columns=houseInfo.columns)
newPrices = priceScaler.inverse_transform(model_1.predict(houseInfo))
print(newPrices)




```

    [[314568.53]
     [219923.77]
     [347135.53]
     [199014.89]
     [257478.86]]
    

## Saving a model for later use
If you want to use the model with out traing, its a good idea to save it.



```python
keras.models.save_model(model_1, 'housePriceModel', True)
! zip -r hpmodel.zip housePriceModel *.pkl

```

    INFO:tensorflow:Assets written to: housePriceModel/assets
      adding: housePriceModel/ (stored 0%)
      adding: housePriceModel/assets/ (stored 0%)
      adding: housePriceModel/saved_model.pb (deflated 88%)
      adding: housePriceModel/variables/ (stored 0%)
      adding: housePriceModel/variables/variables.index (deflated 63%)
      adding: housePriceModel/variables/variables.data-00000-of-00001 (deflated 45%)
      adding: my_cool_encoder.pkl (deflated 38%)
      adding: my_cool_pxscaler.pkl (deflated 30%)
      adding: my_cool_scaler.pkl (deflated 42%)
    

# Loading a model
Lets load the scalers, encoders and model and predict prices using the model we loaded, they should be the same as above.


```python
#! unzip  /content/hpmodel.zip

featureScaler2 = joblib.load('/content/my_cool_scaler.pkl')
priceScaler2 = joblib.load('/content/my_cool_pxscaler.pkl')
enc2 = joblib.load('/content/my_cool_encoder.pkl')  
ordinalColumns = ['YearClosed', 'Type', 'Zip']
model_2 = keras.models.load_model("housePriceModel")

```


```python
houseData2 = {'YearClosed': [2020.00, 2019.00, 2019.00, 2019.00, 2019],
	'Type': ['SFH', 'SFH', 'SFH', 'Condo', 'Townhouse'],
	'Area': [2940, 1500, 1500, 1500, 1500],
  'Zip': [60002, 60002, 60076, 60076, 60076],
	'Rooms': [9, 7, 7, 7, 7],
  'FullBaths': [2.5, 2.5, 2.5, 2.5, 2.5],
  'Beds': [4, 3, 3, 3, 3],
  'GarageSpaces': [2, 2, 2, 0, 0]  }

houseInfo2 = pd.DataFrame(houseData2)
ordinalData2 = houseInfo2[ordinalColumns]
houseInfo2[ordinalColumns] = enc2.transform(ordinalData)

houseInfo2 =  pd.DataFrame(featureScaler2.transform(houseInfo2), columns=houseInfo2.columns)
newPrices2 = priceScaler2.inverse_transform(model_2.predict(houseInfo2))
print(newPrices)
houseInfo2.to_json(r'houseInfo.json')
```

    [[314568.53]
     [219923.77]
     [347135.53]
     [199014.89]
     [257478.86]]
    

# Using your model in different environments
A good thing about using Tensorflow and Keras is that it is fairly easy to convert your trained model to work in a differen environment.
Here we will generate a model that can be using in JavaScript.


```python
! pip install tensorflowjs
import tensorflowjs as tfjs

tfjs.converters.save_keras_model(model_1, 'tensorFlowJs')
! zip -r hpmodelJs.zip tensorFlowJs

```

# Optional fun

If we are happy with the model we can now convert and save it - I choose to save the model using Tensorflow lite to be able to run it on a small device.



```python
# Convert the model to the TensorFlow Lite format without quantization
converter = tf.lite.TFLiteConverter.from_keras_model(model_1)
model_no_quant_tflite = converter.convert()

# Save the model to disk
open("housePriceModelNoQuantization", "wb").write(model_no_quant_tflite)

```

If you want to run the model using C on some smaller device (e.g. a Raspberry Pi) you can dump the model weightd as a C++ file.



```python
# Install xxd if it is not available
!apt-get update && apt-get -qq install xxd
# Convert to a C source file, i.e, a TensorFlow Lite for Microcontrollers model
!xxd -i housePriceModelNoQuantization > model_1.cc
!cat model_1.cc
# Update variable names
#REPLACE_TEXT = MODEL_TFLITE.replace('/', '_').replace('.', '_')
#!sed -i 's/'{REPLACE_TEXT}'/g_model/g' {MODEL_TFLITE_MICRO}
```
