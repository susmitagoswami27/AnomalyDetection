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
5. **ML_RF+AD(Proximity-based, Linear-Model, etc)**: Random Forest regressor has been used here and an uninformed lag 7 has been used to capture the underlined data behavior. Feature combinations like predicted+actual, predicted+residual+actual, and predicted+residual have been tested. And, it has been seen that the feature combination actual+predicted worked better. The performance gradient boosting has shown great performance.
