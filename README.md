# Schizophrenia Patients Mortality Prediction

#### [Kaggle Link](https://www.kaggle.com/c/cityu-cs4487-2018-course-project-1/leaderboard)

#### Group 1: Oops, it's not the ground truth

## Goal

The course project is the following:

### Predicting outcomes of schizophrenia patient

> The goal is to predict the 12-year outcomes of patients with schizophernia using 3-years of data collected after the first diagnosed episode.  The dataset contains two prediction tasks: 1) predict Suicide death within 12 year time-span; 2) predict treatment resistance to antipsychotic medication (if a patient becomes resistant to medication, than another medication Clozapine is prescribed). A good prediction model would help create early treatment models to identify potential future problems.

> There are two types of data: 1) cross-sectional data consists of patient measurements taken at a specific time point (12 years after the first diagnosis); 2) longitidual data consists of 3 years of measurements after the first episode.  For this problem, the cross-sectional data contains the class we want to predict.  The longitudinal data contains baseline information and monthly reports of symptoms and functioning for the first 3 years.


You only need to select one of these tasks (mortality or treatment resistance) for the course project. 

<span style="color:red">_**NOTE: This dataset is provided by the Psychiatry department HKU and contains some sensitive and propietary information. Do NOT redistribute this dataset to other people.**_</span>


## Groups
Projects should be done in Groups of 2.  To sign up for a group, go to Canvas and under "People", join one of the existing "Project Groups".  _For group projects, the project report must state the percentage contribution from each project member._

## Methodology
You are free to choose the methodology to solve the task.  In machine learning, it is important to use domain knowledge to help solve the problem.  Hence, instead of blindly applying the algorithms to the data you need to think about how to represent the data in a way that makes sense for the algorithm to solve the task. 


## Evaluation on Kaggle

The final evaluation will be performed on Kaggle. See the below code about the evaluation procedure.

---
## Proposed Solutions
- Data Pre-processing
    - Remove ignore variables
    - Process Time-robust data
    - Filter variables with too many NaN
    - Imputer: replace NaN cells
        - Simple imputer: replace by "mean", "median", "mode"
        - Imputer model: Latent Factor Model (LFM); IterSVD
    - Data normalization: "l1", "l2", "minmax"
- Classifier training
    - Logistic Regression
    - SVM
    - AdaBoost
    - Neural Network
    - Random Forest
- Solving data imbalance
    - Under sampling
    - Over sampling
        - Inspired by data augmentation
        - SOMTE
        - MSOMTE
        - ANASYN
    - Adapted Algorithms
        - Gradient Boosting Classifier
        - Random Forest (discussion)
    - Anomaly detection
        - DBSCAN
        - Isolated Forest
- Feature selection
    - Manual selection
    - Correlation-based feature selection
        - Spearman's rank correlation
        - Kendall's Tau correlation
    - Model-based feature selection
        - Select by `clf.feature_importance_`
        - Select by `SelectFromModel`
        - Universal feature selection
- Other tricks
    - Model Ensemble
    - Classification by time-sensitive data
        - Process Time-sensitive data
        - Neural Network
            - 1D Convolotional
            - RNN
            - LSTM
            - CNN + RNN/LSTM
        - Distance-based classification
            - Introducing DTW
            - DTW + kNN
            - DTW + MDS + Other classification
            - DTW + anomaly detection
        
---
#### [CityU CS4487 Machine Learning Syllabus](https://www.cityu.edu.hk/catalogue/ug/201819/course/CS4487.htm)

