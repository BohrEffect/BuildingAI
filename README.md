# Predicting remaining useful life

## Summary 
A machine learning methodology is used in order to better predict the so called remaining useful life of a component. Using regression analysis, a relation can be computed between a number of variables and remaining life, in order to able to determine suitable points in time when maintenance is appropriate.
Building AI course project

## Background
An important feature of any type of machinery is the uptime, i.e. what percentage of total possible working time that the machine is operational. High uptime is necessary for any type of business to be efficient and profitable and is affected by many factors. Two factors that may decrease uptime is scheduled maintenance and unscheduled maintenance, i.e. repairing the machine. An interesting methodology to use in order to decrease both scheduled and unscheduled maintenance is predictive maintenance, which implies to gather data of a particular individual machine and use it to determine individual, perhaps unique points in time of when to carry out maintenance. Normally, one uses only some feature like working hours to decide maintenance, which could be both unnecessarily early in some cases and too late in some. 

## How is it used?
Like many other machine learning algorithms, it is detrimental to have both relevant, representative and enough amount of data. For instance, to predict remain useful life of the suspension of a car, relevant features would be total distance driven, road roughness(asphalt/gravel), weight of carr, dimensions of components, to name a few. 
Describe the process of using the solution.

```
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    # Read data from csv file, with delimiter
    dataset = pd.read_csv('ai4i2020processed.csv',';')

    # Separate variables and results.
    X = dataset.iloc[:, [1, 2, 3, 4, 5]].values
    y = dataset.iloc[:, 6].values

    # Train model using linear regression.
    regressor = LinearRegression()
    regressor.fit(X, y)

    print(regressor.coef_)

main()

```


## Data sources and AI methods
Linear regression analysis is used to find coefficients to an expression that may relate values of some features to predicted life. 
The used data is a set of synthetically produced data that reflects true instances which could be found when gathering from a true machine. It consists of 10 000 instances of 14 features. 

•	Unique ID – a number in the interval 1 – 10 000
•	Product ID – Low(L), Medium(M) or High(H), representing product quality.
•	Air Temperature [K]
•	Process Temperature [K]
•	Rotational speed [RPM]
•	Torque [Nm]
•	Tool wear [minutes]
•	Machine failure – 1(true) or 0(false)

The last features represent the mode of failure, which is simply describing how a component has failed. These are not considered, instead the information regarding if any type of failure has occurred is deemed sufficient. Also, the unique ID is not input to the actual regression analysis as it is only an identifier and not a feature. 		


## Challenges
This article only describes a method to approximate remaining useful life, which is not enough in order to increase the uptime. The actual optimised planning of maintenance is missing. One should use this method as a tool to estimate remaining life for as many components of a machine as possible. The result is then used to plan maintenance.

## What next?
A large obstacle to tackle before one could apply machine learning described here is the gathering of suitable data. The output data of sensors of machinery in operation should be logged for an expanded period of time and it is also important to avoid bias of the data. For instance, the machines used to gather data from should be representative for all machines one intends to use the predictive maintenance model.
Only linear regression has been used in this article, which may be limiting the accuracy of the model. Using non-linear regression has the potential to be at least as good a linear one, so fitting one to the data should be performed before the model is deployed. 

## Acknowledgments
Used data is available at:
https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset
