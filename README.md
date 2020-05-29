# Introduction
Cancer is one of the most dangerous disease. As there is cure for some types of cancer it is very costly and all humans cannot afford it. So diagnosing the cancer in early stages itself is the best thing we can do. So this project is about classify the different types of cancer using different machine learning algorithms.

![](https://miro.medium.com/max/700/0*1b88wdy0KFYH2ef5)

For detailed analysis please check this [article](https://medium.com/@tulasiram11729/personalized-cancer-diagnosis-3d6f09a6b8c9)

## Machine Learning Algorithms Used
- `Naive Bayes`
- `Logistic Regression`
- `Support Vector Machine`
- `Random Forest`
- `Stacking Classifier(NB,SVM,LR)`
- `Voting Classifier`

## Dataset Link
Data is used from Kaggle website. Use this [link](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data) to download data.

## About the files
- `TFIDF_approaches.ipynb` - This file consists code and results of all machine learning algorithms with TFIDF vectorization(unigrams and bigrams) of data
- `LR with 4 grams.ipynb` - This file consists detailed analyis of Logistic Regression with one hot encoding and response coding with 4 grams
- `LR AND LSTM approach.ipynb` - This file consists results for two layer LSTM model.
- `TFIDF_2000.ipynb` - This file consits analysis on only top 2000 features. For best results check this notebook.

## Results

##### BOW VECTORIZER
| ALGORITHM USED                | TRAIN LOG LOSS | CV LOG LOSS | TEST LOG LOSS | MISCLASSIFIED POINTS(%)|
| :---------                    |      :-----:   |   :----:    |    :----:     |         :-------:      |
|NB(ONE HOT)                    | 0.90           |    1.27     |   1.21        |          39.84         |
|KNN(RESPONSE CODING)           | 0.705          |    1.130    |   1.002       |          39.47         |  
|LR+BALANCING(ONE HOT)          | 0.614          |    1.143    |   1.048       |          34.77         |  
|LR+IMBALANCING(ONE HOT)        | 0.628          |    1.185    |   1.054       |          36.28         |  
|SVM+BALANCING(ONE HOT)         | 0.739          |    1.132    |   1.063       |          36.47         |  
|RF(ONE HOT)                    | 0.703          |    1.192    |   1.097       |          36.84         |  
|RF(RESPONSE CODING)            | 0.052          |    1.325    |   1.211       |          46.8          |  
|STACKING CLASSIFIER(NB,LR,SVM) | 0.663          |    1.177    |   1.081       |          36.24         |  

 Observation: RF(Response Coding) model is overfitted. Considering all business constraints mentioned in this [blog](https://medium.com/@tulasiram11729/personalized-cancer-diagnosis-3d6f09a6b8c9),  LR+Balancing better than all other models.

##### TF-IDF VECTORIZER
| ALGORITHM USED                      | TRAIN LOG LOSS | CV LOG LOSS | TEST LOG LOSS | MISCLASSIFIED POINTS(%)|
| :---------                          |      :-----:   |   :----:    |    :----:     |         :-------:      |
|NB(ONE HOT)                          | 0.906          |    1.229    |   1.225       |          41            |
|LR+BALANCING(ONE HOT)                | 0.57           |    1.114    |   1.113       |          36.84         |  
|LR+IMBALANCING(ONE HOT)              | 0.56           |    1.13     |   1.12        |          36.09         |  
|LINEAR SVM(ONE HOT)                  | 0.68           |    1.15     |   1.19        |          37.78         |  
|RF(ONE HOT)                          | 0.64           |    1.12     |   1.16        |          37.7          |  
|RF(RESPONSE CODING)                  | 0.05           |    1.38     |   1.44        |          48.8          |  
|STACKING CLASSIFIER(NB,LR,SVM)       | 0.63           |    1.16     |   1.19        |          38.90         |
|MAXIMUM VOTING CLASSIFIER(NB,LR,SVM) | 0.63           |    1.16     |   1.19        |          40.10         |

Observation : When compared with BOW vectorizer results of TF-IDF Vectorizer are better and LR is outperforming all other algorithms and RF is overfitting again. So lets see detailed analysis on LR

##### LR (UNIGRAMS AND BIGRAMS)
| ALGORITHM USED                      | TRAIN LOG LOSS | CV LOG LOSS | TEST LOG LOSS | MISCLASSIFIED POINTS(%)|
| :---------                          |      :-----:   |   :----:    |    :----:     |         :-------:      |
|LR(BIGRAMS)                          | 0.83           |    1.17     |   1.19        |          40.7          |

##### LR(4 GRAMS WITH TOP 2000 FEATURES)
| ALGORITHM USED                      | TRAIN LOG LOSS | CV LOG LOSS | TEST LOG LOSS | MISCLASSIFIED POINTS(%)|
| :---------                          |      :-----:   |   :----:    |    :----:     |         :-------:      |
|LR(4 GRAMS)                          | 0.439          |    0.957    |   0.982       |          35.4          |

LR (4 Grams With top 2000 features) outperformed all other algorithms and log loss value less than 1.0 

Feel free to add pull request if you good results on this dataset.

**Don't forget to star the repo if you liked the analysis.**

## Acknowledgment

This case study is part of this [course](https://www.appliedaicourse.com/)
