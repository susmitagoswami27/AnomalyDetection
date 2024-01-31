# Domain Knowledge Enhanced Anomaly Detection for Industrial Applications

This Repository explores different machine learning algorithms for data cleaning, preparation, modeling, anomaly detection, and hybrid data modeling methods for anomaly detection in multivariate time series data.
This repository also includes files to create dummy multivariate weather data and apply all the methods as client data in proprietary.
Along with that the repository also shows different visualization mechanisms as well as statistical analysis of the results. 

# Package Requirements
 1. Python
 2. Pyod
 3. Tensorflow
 4. Pandas
 5. Scikit-learn
 6. Numpy
 7. Matplotlib
 8. Seaborn
 9. Plotly
 10. PyQt

# Notebooks and Descriptions
1. **Synthetic_datacreation_and_anomalyinjection**: It holds a function for synthetically generated data creation and a function for anomalous data injection. Additionally, here explicit labeling has been done on anomolous and nonanomolous data.
2. **Raw_anomalydetection + knn**: Then on the synthetically generated raw data, robustscaler has been used for feature scaling of three multivariate time series, here train test split has been done with 3 years of Test data and 12 years of train data. And kNN anomaly detection algorithm has been applied. Additionally, the visualization of detected anomalies and missed anomalies has been visualized in 1d individually and in 3d.
3. **RAW_AD(Proximity-based, Linear-Model, etc)_Unsupervised_supervised**: 40 different anomaly detection methods have been applied to raw data to check the performance of different algorithms in detecting the anomalies.
4. **ML_GB+kNN_featurecombination test**: Gradient Boosting regressor has been used here and an uninformed lag 7 has been used to capture the underlined data behavior. Feature combinations like predicted+actual, predicted+residual+actual, and predicted+residual have been tested. And, it has been seen that the feature combination actual+predicted worked better. The performance gradient boosting has shown great performance.
5. **ML_RF+kNN_featurecombination test**: Random Forest regressor has been used here and an uninformed lag 7 has been used to capture the underlined data behavior. Feature combinations like predicted+actual, predicted+residual+actual, and predicted+residual have been tested. And, it has been seen that the feature combination actual+predicted worked better. The performance random forest has shown great performance.
6. **ML_SVR+kNN_featurecombination test**: Support Vector regressor has been used here and an uninformed lag 7 has been used to capture the underlined data behavior. Feature combinations like predicted+actual, predicted+residual+actual, and predicted+residual have been tested. And, it has been seen that the feature combination actual+predicted worked better.
7. **ML_XG+kNN_featurecombination test**: Support Vector regressor has been used here and an uninformed lag 7 has been used to capture the underlined data behavior. Feature combinations like predicted+actual, predicted+residual+actual, and predicted+residual have been tested. And, it has been seen that the feature combination actual+predicted worked better.
8. **ML_XGB+AD(Proximity-based, linear-model, etc)**: 40 different anomaly detection methods have been applied after modeling the data with machine learning(XG Boost algorithm) to check the performance of different algorithms in detecting the anomalies.
9. **ML_RF+AD(Proximity-based, Linear-Model, etc)**: 40 different anomaly detection methods have been applied after modeling the data with machine learning(Random Forest) to check the performance of different algorithms in detecting the anomalies.
10. **ML_algorithms_consolidated+AD__3yearTestdata** The performance of above mentioned 4 different machine learning methods is compared to choose the two best-performing methods.
11. **VAR-Statistical_Test_with_1year test data** Run the statistical test to check whether the VAR model can be applied to the data or not.
12. **VAR_damping effect_modified_with_extendedlag_3yrtestdata** It shows the dampening effect of VAR in long-term data behavior prediction.
13. **HM+AD((Physical+ANN)=hybrid)+kNN(3 ANN combined)_3yearTestdata** It shows the Hybrid modelling with VAR and ANN. ANN predicts the coefficient for the lagged feature identified by VAR.
14. **HM+AD((Physical+ANN)=hybrid)_(Proximity-based, Linear-Model, etc)** 40 different anomaly detection methods have been applied after modeling the data with hybrid modeling to check the performance of different algorithms in detecting the anomalies.
15. **HM+AD((Physical+1D-CNN)=hybrid)_(Proximity-based, Linear-Model,etc)** It shows the Hybrid modelling with VAR and 1D-CNN. CNN predicts the coefficient for the lagged feature identified by VAR.
16. **Custom_HCM_HybridModelling_Coeffsasweights** Here with second variant of the ANN coefficient has been fixed as weights. So, they are not changing here. To do this, A custom layer has been written.
17. **VAR_TCN_Hybridmodelling_with_3yrtestdata** It shows the Hybrid modeling with VAR and TCN.
18. **Consolidated_(RAWAD,MLAD,HMAD)_comparision** Consolidated comparison of three anomaly detection pipeline.
