# Image-Classification-ML-Project
An image classification machine learning project developed to separately identify Cristiano Ronaldo and Lionel Messi.


Project Overview
- This project is developed to identify two different sports celebrities (Cristiano Ronaldo and Lionel Messi) from images. 
- The system uses Machine Learning techniques along with a Flask-based REST API and a web user interface that allows users to upload images and view predictions.

Data Collection
- Images were collected from Kaggle.
- The dataset contains images of two classes: Messi and Ronaldo.

Data Cleaning and Feature Engineering
- Image preprocessing and face detection were performed using OpenCV.
- Faces were cropped using Haar Cascade classifiers.
- Images were resized to 32×32 pixels.
- Haar Wavelet Transformation was applied to extract frequency-based features.
- Raw image features and wavelet features were combined into a single feature vector.
- Wavelet transformation was selected due to its effectiveness with limited datasets.

Model Building
- A Support Vector Machine (SVM) classifier was used.
- Hyperparameter tuning was performed using GridSearchCV.

Application Architecture
- The trained model was saved as a pickle file.
- A Flask API was developed to load the trained model and serve predictions.
- A web-based UI was developed to allow users to upload images and view results.

Technologies Used
- Python
- NumPy, OpenCV (Data cleaning and preprocessing)
- Matplotlib, Seaborn (Data visualization)
- Scikit-learn (Model training and evaluation)
- Flask (REST API)
- HTML, CSS, JavaScript (Web UI)
- Jupyter Notebook, Visual Studio Code (Development tools)

User Interface Images



![image alt](https://github.com/Piyumika2020/Image-Classification-Machine-Learning-Project/blob/797c2f208d2b22c35aa53f1a9d3456bd598769ba/Capture1.PNG)




