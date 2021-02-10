# Decision-Trees
#What are Decision Trees?

Decision Trees are flowchart-like tree structures of all the possible solutions to a decision, based on certain conditions. It is called a decision tree as it starts from a root and then branches off to a number of decisions just like a tree.

The tree starts from the root node where the most important attribute is placed. The branches represent a part of entire decision and each leaf node holds the outcome of the decision.

Attribute Selection Measure
The best attribute or feature is selected using the Attribute Selection Measure(ASM). The attribute selected is the root node feature.

Attribute selection measure is a technique used for the selecting best attribute for discrimination among tuples. It gives rank to each attribute and the best attribute is selected as splitting criterion.

The most popular methods of selection are:

Entropy
Information Gain
Gain Ratio
Gini Index
1. Entropy

To understand information gain, we must first be familiar with the concept of entropy. Entropy is the randomness in the information being processed.


It measures the purity of the split. It is hard to draw conclusions from the information when the entropy increases. It ranges between 0 to 1. 1 means that it is a completely impure subset.

Entropyyyy 1

Here, P(+) /P(-) = % of +ve class / % of -ve class

Example:

If there are total 100 instances in our class in which 30 are positive and 70 are negative then,

P(+) = 3/10 and P(-) = 7/10
H(s)= -3/10 * log2 (3/10) - 7/10 * log2 ( 7/10)  ≈ 0.88

2. Information Gain

Information gain is a decrease in entropy. Decision trees make use of information gain and entropy to determine which feature to split into nodes to get closer to predicting the target and also to determine when to stop splitting.

Gaineee

Here, S is a set of instances , A is an attribute and Sv is the subset of S .

Example:

1 2 3 4 5 6 7 8 9 10 1

Possession of TV at home against monthly income


For overall data, Yes value is present 5 times and No value is present 5 times. So,


H(s) = -[ ( 5/10)  * log2 (5/10) +  (5/10) * log2 (5/10) ] = 1

Let’s analyze True values now. Yes is present 4 times and No is present 2 times.


H(s) = -[ ( 4/6) * log2 ( 4/6) + (2/6) * log2 (2/6) ] = 0.917

For False values,

H(s)= - [ ( 3/4) * log2 (3/4) + (1/4) * log2 (1/4) ] = 0.811

Net Entropy = (6/10) * 0.917 + (4/10) * 0.811 = 0.874

Total Reduction = 1- 0.874 = 0.126  

This value ( 0.126) is called information gain.

3. Gain Ratio

The gain ratio is the modification of information gain. It takes into account the number and size of branches when choosing an attribute. It takes intrinsic information into account.

GR(S,A) = Gain( S,A)/ IntI(S,A)

4. Gini Index

Gini index is also type of criterion that helps us to calculate information gain. It measures the impurity of the node and is calculated for binary values only.

Gini Impu

Example:

C1 = 0 , C2 = 6

P(C1) = 0/6 = 0
P(C2) = 6/6 = 1
Giniiiiiiiii
Gini impurity is more computationally efficient than entropy.

Decision Tree Algorithms in Python
Let’s look at some of the decision trees in Python.

 1. Iterative Dichotomiser 3 (ID3)
This algorithm is used for selecting the splitting by calculating information gain. Information gain for each level of the tree is calculated recursively.

2. C4.5
This algorithm is the modification of the ID3 algorithm. It uses information gain or gain ratio for selecting the best attribute. It can handle both continuous and missing attribute values.

3. CART (Classification and Regression Tree)
This algorithm can produce classification as well as regression tree. In classification tree, target variable is fixed. In regression tree, the value of target variable is to be predicted.

Decision tree classification using Scikit-learn
We will use the scikit-learn library to build the model and use the iris dataset which is already present in the scikit-learn library or we can download it from here.

The dataset contains three classes- Iris Setosa, Iris Versicolour, Iris Virginica with the following attributes-

sepal length
sepal width
petal length
petal width

We have to predict the class of the iris plant based on its attributes.

1. First, import the required libraries
import pandas as pd 
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

2. Now, load the iris dataset
iris=load_iris()
To see all the features in the datset, use the print function

print(iris.feature_names) 
Output:

['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

To see all the target names in the dataset-


print(iris.target_names) 

Output:

['setosa' 'versicolor' 'virginica']

3. Remove the Labels

Now, we will remove the elements in the 0th, 50th, and 100th position. 0th element belongs to the Setosa species, 50th belongs Versicolor species and the 100th belongs to the Virginica species.

This will remove the labels for us to train our decision tree classifier better and check if it is able to classify the data well.

#Spilitting the dataset

removed =[0,50,100]

new_target = np.delete(iris.target,removed)

new_data = np.delete(iris.data,removed, axis=0) 

4. Train the Decision Tree Classifier

The final step is to use a decision tree classifier from scikit-learn for classification.

#train classifier

clf = tree.DecisionTreeClassifier() # defining decision tree classifier

clf=clf.fit(new_data,new_target) # train data on new data and new target

prediction = clf.predict(iris.data[removed]) #  assign removed data as input

Now, we check if our predicted labels match the original labels

print("Original Labels",iris.target[removed])

print("Labels Predicted",prediction)

Output:


Original Labels [0 1 2]

Labels Predicted [0 1 2]

Wow! The accuracy of our model is 100%. To plot the decision tree-
