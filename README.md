# Bati Bank Credit Scoring Model
The Bati Bank Credit Scoring Model leverages transactional data sourced from the eCommerce platform to establish a robust framework for categorizing users into high-risk and low-risk segments. This initiative involves identifying pertinent predictors of default, selecting relevant features, and constructing predictive models to assess risk probabilities. Ultimately, the model aims to compute credit scores, determine optimal loan amounts, and recommend suitable durations, thereby facilitating informed lending decisions and risk management strategies at Bati Bank.

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
