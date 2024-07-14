# Project Title
Robust Gas Classification under Sensor Drift Conditions: A Machine Learning Approach
# ## Project Overview
This project focuses on the development of machine learning algorithms to accurately classify gases despite the presence of sensor drift, a common issue that affects the reliability and precision of gas sensors over time. By leveraging established machine learning techniques such as K-Nearest Neighbors (KNN), Support Vector Machines (SVM), and Artificial Neural Networks (ANN), this project aims to mitigate the effects of sensor drift. The approach involves data collection, feature extraction, and the application of robust machine learning models to enhance the reliability and prediction of gasses.
## Dataset
The gas sensor array drift dataset (GSAD) is used as the research object in this study. Created and donated by Alexander Vergara in 2012, the dataset contains 13,910 chemical gas sensor data points collected by 16 sensors for six different gases (ethanol, ethylene, ammonia, acetaldehyde, acetone, and toluene). The data was collected over a 36-month period from January 2008 to February 2011 at the University of California, San Diego.
## Methodology

The methodology involves several key steps:
1. **Data Collection:** Gathering raw sensor data from the GSAD dataset.
2. **Feature Extraction:** Processing the raw data to extract relevant features.
3. **Model Application:** Applying various machine learning models (KNN, SVM, Random Forest, and ANN) to classify gases accurately.
4. **Evaluation:** Analyzing and comparing the effectiveness of the models in mitigating the impact of sensor drift.

## Results

The models' performance is evaluated based on their accuracy in classifying gases under sensor drift conditions. Initial results demonstrate the potential of these machine learning techniques to enhance gas classification accuracy despite sensor drift.
## Usage

To use this project, follow these steps:
1. Clone the repository:
   ```bash
   [git clone https://github.com/HakanKARASU/Gas-Sensor-Array-Drift-Dataset.git](https://github.com/HakanKARASU/Gas-Sensor-Array-Drift-Dataset.git)
2. cd Gas-Sensor-Array-Drift-Dataset
3. pip install -r requirements.txt
4. python run.py
5. python neural_network_1.py

