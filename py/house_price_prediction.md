# Setting up a regression model for house prices using Keras and Tensorflow

To understand how the functionality fits into the typical workflow of data preparation, training the model and evaluating the fit using test data sets and using the model. I took a look at implementing a simple regression application to predict the sale price of a house given a simple set of features over about 4000 home sales. 

In the sample, we are going to take a look at a supervised learning problem of regression. A supervised learning task we train the model predicts a label's value from a set of features. 

The focus was on getting a small sample model up and running. 
We then experiment with the choice of feature and training algorithms to minimize prediction errors.


We will run the code in Google's Colaboratory since this provides an excellent environment that allows access to GPU's, requires no configuration and enables sharing.


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
We then need to sanity test the raw data we loaded to ensure the columns are in the date range we expect.  It is a good idea to check the data for any outliers; for example, in my first iteration, one home had 82 garage spots.
The test data is from partsof the norther suberbs of Chicago, over a few years - so there is a regional bias in the data.



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
      <td>6.851000e+03</td>
      <td>6851.000000</td>
      <td>6.851000e+03</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
      <td>6851.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.030734e+07</td>
      <td>2018.963801</td>
      <td>4.203940e+05</td>
      <td>90.858415</td>
      <td>60091.594220</td>
      <td>2120.467815</td>
      <td>7.750401</td>
      <td>2.124799</td>
      <td>0.530434</td>
      <td>3.152168</td>
      <td>0.130784</td>
      <td>1.589899</td>
      <td>0.364180</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.097783e+05</td>
      <td>1.030840</td>
      <td>2.824380e+05</td>
      <td>64.187286</td>
      <td>73.971679</td>
      <td>1154.188644</td>
      <td>2.402407</td>
      <td>0.923679</td>
      <td>0.577486</td>
      <td>1.034099</td>
      <td>0.377626</td>
      <td>0.963217</td>
      <td>1.048626</td>
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
      <td>9.997535e+06</td>
      <td>2018.000000</td>
      <td>2.430000e+05</td>
      <td>62.000000</td>
      <td>60062.000000</td>
      <td>1304.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.038995e+07</td>
      <td>2019.000000</td>
      <td>3.440000e+05</td>
      <td>76.000000</td>
      <td>60076.000000</td>
      <td>1800.000000</td>
      <td>7.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.064400e+07</td>
      <td>2020.000000</td>
      <td>5.200000e+05</td>
      <td>76.000000</td>
      <td>60077.000000</td>
      <td>2620.000000</td>
      <td>9.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>2.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.095755e+07</td>
      <td>2020.000000</td>
      <td>4.300000e+06</td>
      <td>201.000000</td>
      <td>63104.000000</td>
      <td>13000.000000</td>
      <td>18.000000</td>
      <td>8.000000</td>
      <td>5.000000</td>
      <td>11.000000</td>
      <td>3.000000</td>
      <td>11.000000</td>
      <td>20.000000</td>
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
We should normalize the features and corresponding price targets to facilitate laerning.


```python
featureScaler = MinMaxScaler()
selectedFeaturesScale =  pd.DataFrame(featureScaler.fit_transform(selectedFeatures), columns=selectedFeatures.columns)

priceScaler = MinMaxScaler()
pricesScale =  pd.DataFrame(priceScaler.fit_transform(prices.values.reshape(-1, 1)),columns=['SoldPr'])

```

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


    
![png](output_13_0.png)
    



    
![png](output_13_1.png)
    



    
![png](output_13_2.png)
    



```python
# We'll use 60% of our data for training and 20% for testing. The remaining 20%
# will be used for validation. Calculate the indices of each section.
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

    Training examples summary:
    Validation examples summary:
    Training targets summary:
    Validation targets summary:
    




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
      <th>SoldPr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1370.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>0.091325</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.063134</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.050292</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.074971</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.115731</td>
    </tr>
    <tr>
      <th>max</th>
      <td>0.625731</td>
    </tr>
  </tbody>
</table>
</div>



## Training a model using Keras
Now we have our data we can use Keras (a high-level API to Tensorflow) to create a model.
 Keras makes it easy to experiment with different model architectures and visualize the results.

We will begin with three layers and use the "relu" activation function. Notice that the first layer uses the featureCount to define its input shape, and the final layer outputs to a single neuron since its the predicted price.

Notice that we are using some regularization in the model this is to reduce the chance of overfitting.


```python
model_1 = tf.keras.Sequential()

#model_1.add(keras.layers.Dense(16, activation='relu', input_shape=(featureCount,)))
model_1.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001), input_shape=(featureCount,)))
#model_1.add(keras.layers.Dense(16, activation='relu'))
model_1.add(keras.layers.Dense(16, activation='relu', kernel_regularizer=regularizers.l2(0.001)))
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


    
![png](output_20_0.png)
    



    
![png](output_20_1.png)
    


### Final test
We can now use the 20% of the data we kept aside for a final test. First, we use the model to evaluate the test data and print out the mean squared error and the mean average error for predictions.

It's a great idea to visualize some of the data by plotting actuals against predicted results.


```python
# Calculate and print the loss on our test dataset
test_loss, test_mae = model_1.evaluate(test_examples, test_targets)
print("mean squared error:", test_loss, " mean average error:", test_mae)

y_test_pred = model_1.predict(test_examples)

# Graph the predictions against the actual values
plt.clf()
plt.title('Comparison of predictions and actual values')
plt.plot( test_targets,  y_test_pred, 'b.', label='Actual values')
plt.show()

```

    43/43 [==============================] - 0s 714us/step - loss: 0.0012 - mae: 0.0204
    mean squared error: 0.0011992050567641854  mean average error: 0.02039984054863453
    


    
![png](output_22_1.png)
    


## Run your own examples
Lets try with a couple of test inferences  - the single family home (SFH) should be lower in zip 60002 than 60076


```python


#Simple Sample
houseData = {'YearClosed': [2020.00, 2019.00, 2019.00, 2019],
	'Type': ['SFH', 'SFH', 'Condo', 'Townhouse'],
	'Area': [2940, 1500, 1500, 1500],
  'Zip': [60002, 60076, 60076, 60076],
	'Rooms': [9, 7, 7, 7],
  'FullBaths': [2.5, 2.5, 2.5, 2.5],
  'Beds': [4, 3, 3, 3],
  'GarageSpaces': [2, 2, 0, 0]  }

houseInfo = pd.DataFrame(houseData)
ordinalData = houseInfo[ordinalColumns]
houseInfo[ordinalColumns] = enc.transform(ordinalData)


houseInfo =  pd.DataFrame(featureScaler.transform(houseInfo), columns=houseInfo.columns)
newPrices = priceScaler.inverse_transform(model_1.predict(houseInfo))
print(newPrices)




```

    [[338236.22]
     [375158.7 ]
     [270084.6 ]
     [327426.75]]
    

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
