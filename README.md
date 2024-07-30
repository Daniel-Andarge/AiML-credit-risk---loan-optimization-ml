# Integrated Predictive Credit Risk Modeling and Optimized Lending Decisions with Advanced Segmentation
This project focuses on leveraging advanced data analytics and machine learning techniques to develop a comprehensive credit risk assessment, credit scoring, and loan optimization framework. The system is designed to support a Buy-Now-Pay-Later (BNPL) service offered by a financial service provider in partnership with an eCommerce company. The primary objectives include customer segmentation, credit risk modeling, and loan optimization.

## Project Objectives

### 1. Customer Segmentation
- **Objective**: Conduct in-depth customer segmentation using RFMS (Recency, Frequency, Monetary Value, and Standard Deviation of Amount Spent) scores.
- **Purpose**: Classify customers into high-risk and low-risk segments to tailor the BNPL or loan service offerings.

### 2. Credit Risk Modeling
- **Objective**: Develop machine learning models to predict credit risk and default probabilities.
- **Outcome**: Provide a risk probability for each customer, aiding in the assessment of creditworthiness and default risk.
- **Credit Score Model**: Create a credit score model based on risk probabilities, aligned with FICO standards.
  ![creditscore](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/assets/FICO_Score.png)

### 3. Loan Optimization Model
- **Objective**: Develop a model to determine optimal loan amounts, repayment periods, and other terms.
- **Outcome**: Offer personalized financing options to customers, enhancing the BNPL service's value proposition.

## Methodology

The project integrates supervised and unsupervised machine learning techniques, including logistic regression, decision trees, random forests, and clustering algorithms. These models are trained and validated using historical BNPL data and external credit bureau information. The ultimate goal is to embed these models into the BNPL platform, improving credit decision-making, customer satisfaction, and business performance.

## Table of Contents

1. [Data Collection and Preprocessing](#data-collection-and-preprocessing)
2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
3. [Feature Engineering](#feature-engineering)
4. [Weight of Evidence (WoE) Binning](#weight-of-evidence-woe-binning)
5. [Feature Selection](#feature-selection)
6. [Model Development](#model-development)
7. [Model Evaluation and Selection](#model-evaluation-and-selection)
8. [Model Deployment and Integration](#model-deployment-and-integration)
9. [Monitoring and Continuous Improvement](#monitoring-and-continuous-improvement)
10. [Installation](#installation)
11. [Usage](#usage)
12. [Contributing](#contributing)
13. [License](#license)
14. [Acknowledgments](#acknowledgments)

## Data Collection and Preprocessing

Gather and preprocess historical BNPL application and repayment data. This includes data cleaning, handling missing values, and normalization.

- **Notebook**: [Data Cleaning](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/data_cleaning.ipynb)

## Exploratory Data Analysis (EDA)

Analyze customer characteristics, behaviors, and credit profiles to identify patterns influencing credit risk and repayment.

- **Notebook**: [EDA](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/eda.ipynb)

## Feature Engineering

Create new features, including RFMS scores, based on insights from EDA to enhance the models' predictive power.

#### Customer Segmentation
Constructing a default estimator (proxy) By visualizing all transactions in the RFMS space to establish a boundary Where customers are classified as high and low RFMS scores.

1.Visualizing Transactions in RFMS space & Establishing boundaries.

![rfms](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/assets/rfms_space.png)

2. Segmentation 


- **Notebook**: [Feature Engineering](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/feature_engineering.ipynb)

## Weight of Evidence (WoE) Binning

Apply WoE binning to transform features into a suitable format for machine learning models.

- **Notebook**: [WoE Binning](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/feature_engineering.ipynb)

## Feature Selection

Select the most relevant features using techniques like correlation analysis and recursive feature elimination.

1. Correlation Analysis 

![cor](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/assets/correlation.png)

2. Selected Features 

![features](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/assets/selected_features.png)

- **Notebook**: [Feature Selection](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/feature_engineering.ipynb)

## Model Development

Develop various machine learning models to predict credit risk, score, and loan optimization metrics.

- **Notebook**: [Model Development](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/model_building.ipynb)

## Model Evaluation and Selection

Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

- **Notebook**: [Model Evaluation](https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml/blob/main/notebooks/model_building.ipynb)

## Model Deployment and Integration

Deploy the selected models into the BNPL platform to enhance decision-making processes.

## Monitoring and Continuous Improvement

Continuously monitor and refine the models to ensure they maintain high accuracy and effectiveness over time.

## Installation

### Prerequisites
- Python 3.x
- Virtual environment (e.g., `virtualenv`, `conda`)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Daniel-Andarge/AiML-credit-risk---loan-optimization-ml.git
   ```

2. Navigate to the project directory:
   ```bash
   cd AiML-credit-risk---loan-optimization-ml
   ```

3. Create and activate a virtual environment:
   ```bash
   # Using virtualenv
   virtualenv venv
   source venv/bin/activate

   # Using conda
   conda create -n your-env python=3.x
   conda activate your-env
   ```

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Open the Jupyter notebooks in your preferred environment and follow the instructions. Customize the code based on your dataset and requirements.

## Contributing

Contributions are welcome. Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your branch to your forked repository.
5. Create a pull request to the main repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

Special thanks to the contributors and the open-source community for their support and resources.
