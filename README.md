# Integrated Credit Risk and Score Modeling with Customer Segmentation 

This project aims to perform in-depth data analysis, feature engineering, and develop advanced machine learning models for comprehensive credit risk assessment, credit scoring, and loan optimization for a buy-now-pay-later (BNPL) loan service provided by a financial service provider (Bank) partnering with an eCommerce company. The key objectives of this project are:

##### 1. Customer Segmentation:
   - Perform in-depth customer segmentation analysis based on RFMS (Recency, Frequency, Monetary Value, and Standard Deviation of Amount Spent) scores derived from the customers' characteristics, behaviors, and credit profile features. This will enable the classification of customers into high-risk and low-risk segments for the BNPL or loan service.

#### 2. Credit Risk Modeling:
   - Develop a machine learning model that can accurately predict the credit risk and default probability, and assign a risk probability for each customer applying for the BNPL or loan service. This model will assess the creditworthiness and potential default risk of the applicants.
   - Develop a credit score model that can accurately predict/assign a credit score based on the risk probability estimates, using the FICO credit score standard as a reference.

##### 3. Loan Optimization Model:
   - Develop a machine learning model that can determine the optimal loan amount, repayment period, and other terms for each applicant based on their credit profile and other relevant factors. This will help the BNPL service provide tailored financing options to customers.

By integrating advanced analytics capabilities, this project aims to provide the buy-now-pay-later (BNPL) service with a comprehensive, data-driven framework for credit risk assessment, credit scoring, loan optimization, and customer-centric decision making. This will ultimately support the service's growth, improve the customer experience, and ensure responsible lending practices. The project leverages various supervised and unsupervised machine learning techniques, such as logistic regression, decision trees, random forests, and clustering algorithms, to build robust and performant credit risk and loan optimization models. The models will be trained and validated using historical BNPL application and repayment data, as well as external credit bureau information, and will be integrated into the BNPL platform to enhance the customer experience, improve credit decisions, and optimize loan portfolios. This project aims to deliver a comprehensive machine learning-based solution that can help the BNPL service provider make more informed and data-driven decisions, ultimately leading to increased customer satisfaction, reduced default rates, and improved overall business performance.

## Getting Started

### Prerequisites

- Python 3.x
- Virtual environment (e.g., `virtualenv`, `conda`)
- Required Python packages (listed in `requirements.txt`)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Daniel-Andarge/AiML-bati-bank-credit-scoring-model.git
   ```
2. Change to the project directory:
   ```
   cd your-project
   ```
3. Create a virtual environment and activate it:

   ```
   # Using virtualenv
   virtualenv venv
   source venv/bin/activate

   # Using conda
   conda create -n your-env python=3.x
   conda activate your-env
   ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

### Usage

1. Start the Jupyter Notebook:
   ```
   jupyter notebook
   ```
2. Navigate to the `notebooks/` directory and open the relevant notebooks:
   - `data_understanding.ipynb`
   - `data_cleaning.ipynb`
   - `feature_engineering.ipynb`
   - `eda.ipynb`
   - `model_building.ipynb`
     Each notebook corresponds to a step in the data analysis process, as outlined in the introduction.

## Scripts and Notebooks

The project is organized into the following scripts and Jupyter Notebooks:

1. **Data Understanding**:

   - `data_understanding.ipynb`

2. **Data Cleaning and Preprocessing**:

   - `data_cleaning.ipynb`

3. **Feature Engineering**:

   - `feature_engineering.ipynb`

4. **Exploratory Data Analysis (EDA)**:

   - `eda.ipynb`

4.1. **EDA log File**

- You can Find the EDA Log file in notebooks/eda_analysis.log

5. **Model Building**:
   - `model_building.ipynb`

Each notebook corresponds to a step in the data analysis process, as outlined in the introduction.

## Dependencies

The required Python packages for this project are listed in the `requirements.txt` file. You can install them using the following command:

```
pip install -r requirements.txt
```

## Contributing

If you would like to contribute to this project, please follow the standard GitHub workflow:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Make your changes and commit them
4. Push your branch to your forked repository
5. Create a pull request to the main repository

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- Thank you to the contributors and the open-source community for their support and resources.
