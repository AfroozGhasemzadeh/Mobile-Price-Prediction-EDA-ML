# Mobile-Price-Prediction-EDA-ML
This project predicts the price range of mobile phones based on their technical specifications. The dataset comes from Kaggle’s Mobile Price Classification (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) challenge and includes a mix of numerical and categorical features such as battery capacity, RAM, screen resolution, camera specifications, and connectivity options. The target variable, , contains four classes (0–3), representing increasing price levels.
To keep the workflow modular and avoid oversized notebooks, the project is organized into four separate notebooks, each focusing on a specific stage of the pipeline: preprocessing, EDA for train and test sets, and model development. All figures are saved externally to maintain notebook performance and readability.
# Motivation
Mobile phones vary widely in price depending on their hardware features. Understanding how these features influence price categories can help manufacturers, retailers, and consumers make informed decisions.
This project aims to:
• 	Explore the relationship between mobile features and price range
• 	Build a machine learning model capable of predicting price categories
• 	Compare multiple algorithms to identify the most effective one

# Repository Structure

MobilePriceClassification/
│

├── 01_Data Processing.ipynb

├── 02_Train Dataset Visualization.ipynb

├── 03_Test Dataset Visualization.ipynb

├── 04_Mobile Price Modelling.ipynb
│

├── Figures-Mobile Price/
│       ├── 1/

│       └── Train/

│       ├── Test/

│       └── Modelling/
│

└── README.md
This structure keeps the project clean, modular, and easy to navigate.
# Technologies & Libraries Used
The project uses a combination of data analysis, visualization, and machine learning libraries:
• 	pandas, numpy — data manipulation
• 	matplotlib, seaborn, plotly.express — visualizations
• 	scikit-learn — preprocessing, modeling, evaluation, tuning
  • 	MinMaxScaler
  • 	train_test_split
  • 	DecisionTreeClassifier
  • 	RandomForestClassifier
  • 	SVC
  • 	GridSearchCV, RandomizedSearchCV
  • 	RepeatedStratifiedKFold, KFold
  • 	Pipeline
  • 	Metrics: accuracy, F1, ROC-AUC, confusion matrix, classification report
• 	plotly.io — interactive plotting
# Notebooks
  # Notebook 1: Data Processing
This notebook prepares the raw Kaggle dataset for analysis and modeling.
It includes:
- Inspecting feature types
- Separating numerical and categorical features
- Cleaning and transforming data
- Removing redundant or irrelevant columns
- Encoding categorical variables
- Scaling numerical features
- Saving processed datasets (df4_train, df4_test)
These saved files allow the next notebooks to run efficiently without repeating preprocessing.
  # Notebook 2: Train Dataset Visualization
This notebook explores the processed training dataset to understand feature behavior and their relationship with the target variable.
It includes:
- Boxplots and KDE plots for numerical features
- Count plots and pie charts for categorical features
- Visual analysis of how features vary across price ranges
- All figures saved in Figures-Mobile Price/Train/
  # Notebook 3: Test Dataset Visualization
A parallel EDA is performed on the test dataset to ensure consistency with the training data.
All plots are saved in Figures-Mobile Price/Test/.
  # Notebook 4: Mobile Price Modeling
This notebook builds and evaluates machine learning models to classify mobile phones into price categories.
Three algorithms are compared:
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
Evaluation metrics include:
- Accuracy
- F1-score
- ROC curves (multiclass)
- AUC
- Confusion Matrix
- Classification Report
- Cross-validation
- Hyperparameter tuning
The best-performing model is selected to predict the price range for the test dataset.
# Dataset Features
The dataset includes:
battery_power, blue, clock_speed, dual_sim, fc, four_g,
int_memory, m_dep, mobile_wt, n_cores, pc, px_height,
px_width, ram, sc_h, sc_w, talk_time, three_g,
touch_screen, wifi, price_range


Some columns were removed during preprocessing due to redundancy or low predictive value.
These decisions are documented in the data-processing notebook.
# How to Run the Project
- Clone the repository
- Install required libraries
- Run the notebooks in order:
- 01_Data Processing.ipynb
- 02_Train Dataset Visualization.ipynb
- 03_Test Dataset Visualization.ipynb
- 04_Mobile Price Modelling.ipynb
- View saved figures in the Figures-Mobile Price/ folder
- Review model performance and predictions
# Results Summary
- Random Forest and SVM typically outperform Decision Trees
- ROC-AUC and F1-score provide strong insight into model quality
- The final selected model achieves strong predictive performance on the test dataset

# Future Improvements
- Add feature engineering (interaction terms, polynomial features)
- Try additional models (XGBoost, LightGBM, CatBoost)
- Deploy the model as an API or web app
- Add SHAP or LIME for model interpretability







