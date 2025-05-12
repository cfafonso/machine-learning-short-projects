# machine-learning-short-projects
This repository contains a collection of distinct short projects in self-contained Jupyter notebooks, each showcasing practical applications of various machine learning algorithms.

## chd-risk-prediction-bayesian-network
The purpose of this project is to implement a Bayesian network that guarantees the best results in predicting the risk of cardiovascular heart disease (CHD). The data for this project has 4240 patients with the following variables:

Demographic risk factors: 
-  `Sex`: sex of patient 
-  `age`: age in years at first examination 
-  `education`: some high school (1), high school/GED (2), some college/vocational school (3), college (4) 

Behavioral risk factors: 
-  `currentSmoker`, `cigsPerDay`: smoking behavior

Medical history risk factors: 
-  `BPMeds`: on blood pressure medication at the time of first examination 
-  `prevalentStroke`: previously had a stroke 
-  `prevalentHyp`: currently hypertense 
-  `diabetes`: currently has diabetes

Risk factors from the first examination:
-  `totChol`: total cholesterol (mg/dL)
-  `sysBP`: systolic blood pressure
-  `diaBP`: diastolic blood pressure
-  `BMI`: body mass index, weight(kg)/height(m)<sup>2</sup>
-  `heartRate`: heart rate (beats/minute)
-  `glucose`: blood glucose level (mg/dL)

The last variable in the dataset is the target variable (`TenYearCHD`) which tells us if the patient has the disease or not. The original data is from a large cohort from the [BioLINCC](https://biolincc.nhlbi.nih.gov/home/) website. 

- Algorithms: Bayesian Networks


## critical-temperature-prediction-1
The purpose of this project is to produce the best possible regression and binary classification models using decision trees and linear models on a chemical dataset . The latter contains information on 21174 superconductors and 81 chemical features along with the critical temperature, which constitutes the target variable and was adapted from the [supercondutivity data](https://archive.ics.uci.edu/dataset/464/superconductivty+data).

- Algorithms: Decision Trees, Linear Regression (with and without regularization), Logistic Regression


## critical-temperature-prediction-2
The purpose of this project is to produce and compare regression and classification models with a full chemical dataset and its projection into principal components. The dataset for this project is from the [supercondutivity data](https://archive.ics.uci.edu/dataset/464/superconductivty+data).

- Algorithms: Decision Trees, Linear Regression, Naïve Bayes, Principal Component Analysis (PCA)


## hand-posture-classifiers
The purpose of this project is to produce the best classifier using exclusively linear, tree-based, naïve Bayes and K-nearest neighbours algorithms on a real-world dataset. The latter contains 78095 instances on 14 distinct users corresponding to a single frame of a hand posture recorded by a camera system. Each record is associated with a particular class of hand posture, which can range from 1 to 5 and constitutes our target variable. The dataset features consist of x, y and z coordinates of the *i*-th unlabeled marker position on glove fingers in a motion capture environment, with i ranging from 0 to 11. In addition, the ‘User’ variable is present in the dataset, but it has no meaning other than acting as an identifier. This dataset was adapted from the [Motion Capture Hand Postures data] (https://archive.ics.uci.edu/dataset/405/motion+capture+hand+postures).

- Algorithms: Logistic Regression, Decision Tree Classifier, Gaussian Naïve Bayes, K-Nearest Neighbours


## market-basket-association-rule-mining
The purpose of this project is to conduct a comprehensive analysis of association rule mining, focusing on two main objectives: (1) benchmark frequent itemset mining approaches to determine an optimal support threshold and identify the most efficient one based on the computational capabilities available, and (2) perform a detailed analysis of association rules generated from the data, including the identification of significant rules, examination of maximal and closed itemsets, and statistical evaluation of rule relevance using metrics such as support, confidence, lift, and conviction. The dataset was adapted from the [instacart market basket data] (https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis).

- Algorithms: Frequent Item Set Mining and Association Rule Learning Algorithms, namely Apriori, Frequent Pattern Growth (FP-growth) and Equivalence Class Clustering and bottom-up Lattice Traversal (ECLAT)


## mountain-car-reinforcement-learning
The purpose of this project is to implement an agent based on the Q-learning algorithm to solve the Mountain Car problem. Mountain Car is a classic reinforcement learning problem, often used as a benchmark for testing the performance of reinforcement learning algorithms. In this problem, an underpowered car must climb a steep hill to reach a goal located at the top of the hill. The car is subject to the laws of physics, which means that it cannot simply drive straight up the hill. Instead, it must build up speed by accelerating back and forth across the hill.  

![Mountain Car](mountain_car_q_learning.gif)

The state of the Mountain Car environment is represented by two continuous variables, the position and velocity of the car. The goal of the agent is to learn how to control the car's acceleration to climb the hill and reach the goal as quickly as possible while using the least amount of energy. The agent receives a negative reward for every time step it takes to reach the goal, so the goal is to minimize the number of time steps required to reach the goal.

- Algorithms: Q-Learning (Reinforcement Learning)


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Structure
```
machine-learning-projects/
|-- chd-risk-prediction-bayesian-network.ipynb
    |-- chd_dataset.csv
    |-- chd_risk_prediction_bayesian_network.ipynb        
|-- critical_temperature_prediction_1/
    |-- critical_temperature_prediction.ipynb
    |-- scaled_dataset.tsv
|-- critical_temperature_prediction_2/
    |-- critical_temperature_prediction_2.ipynb
    |-- train.csv
    |-- unique_m.csv
|-- hand-posture-classifiers/
    |-- hand_posture_classifiers.ipynb
    |-- postures.csv
|-- market-basket-association-rule-mining/
    |-- market_basket_association_rule_mining.ipynb
    |-- order_products.pickle
    |-- products.txt
|-- mountain-car-reinforcement-learning/
    |-- mountain_car_q_learning.gif
    |-- mountain_car_q_learning.ipynb
    |-- pseudocode_q_learning.png
|-- LICENSE
|-- README.md```
