# Car Price Predictor

![A picture from the project](./project-screenshot.png)

A machine learning system for predicting car prices using a Random Forest regression model. The project implements a complete data science pipeline from raw data acquisition through model training, evaluation, and deployment as a production-ready web application.

## Project Architecture

The project follows a microservices architecture with clear separation between data science workflows, model serving, and user interface components.

### Project Structure

```
car-price-predictor/
├── .gitignore                     # Git ignore rules
├── .python-version                # Python version specification
│
├── back-end/                      # Backend FastAPI service
│   ├── app/                       # Application code
│   │   ├── __init__.py           
│   │   ├── main.py                # FastAPI application entry point
│   │   ├── model.py               # Model loading and inference logic
│   │   └── schemas.py             # Pydantic request/response schemas
│   ├── Dockerfile                 # Docker configuration for backend
│   └── requirements.txt           # Backend Python dependencies
│
├── front-end/                     # Frontend Streamlit service
│   ├── app.py                     # Streamlit application
│   ├── api.py                     # API client for backend communication
│   ├── config.py                  # Configuration settings
│   ├── styles.py                  # UI styling
│   ├── Dockerfile                 # Docker configuration for frontend
│   └── requirements.txt           # Frontend Python dependencies
│
├── notebooks/                     # Jupyter notebooks for analysis and modeling
│   ├── 00_get_data.ipynb          # Data acquisition and loading
│   ├── 01_explore_and_split_data.ipynb # EDA and stratified train/test split
│   ├── 02_understand_and_visualize_data.ipynb # Data visualization
│   ├── 03_clean_data.ipynb        # Data cleaning
│   ├── 04_statistical_testing.ipynb # Statistical hypothesis testing
│   ├── 05_feature_engineering.ipynb # Feature creation and transformation
│   └── 06_ml_pipeline.ipynb       # Model training and evaluation
│
├── models/                        # Trained model artifacts
│   └── rf_model_pipeline_v1.pkl   # Serialized Random Forest pipeline
│
│
├── pyproject.toml                 # Project configuration and dependencies
├── uv.lock                        # Dependency lock file
├── compose.yaml                   # Docker Compose multi-container configuration
├── README.md                      # Project documentation
├── utils.py                       # Utility functions and helpers
└── __pycache__/                   # Python bytecode cache
```

### System Components

**Backend Service (FastAPI)**
- RESTful API endpoint for price predictions
- Model loading and inference pipeline
- Input validation using Pydantic schemas
- Located in `back-end/app/`

**Frontend Service (Streamlit)**
- Interactive web interface for user input
- HTTP client for API communication
- Located in `front-end/`

**Data Science Workflow**
- Sequential Jupyter notebooks for exploratory analysis, preprocessing, and modeling
- Located in `notebooks/`
- Produces trained model artifacts stored in `models/`

**Deployment**
- Docker containerization for both services
- Docker Compose orchestration for multi-container deployment
- Service communication via internal networking

### Data Flow

```
Raw Dataset (cars.csv)
    ↓
[Notebook 01] Exploratory Analysis & Stratified Split
    ↓
train.csv / test.csv
    ↓
[Notebook 03] Data Cleaning
    ↓
cleaned.csv
    ↓
[Notebook 05] Feature Engineering
    ↓
preprocessed.csv
    ↓
[Notebook 06] ML Pipeline Training
    ↓
rf_model_pipeline_v1.pkl
    ↓
[Backend] Model Loading & Inference
    ↓
[Frontend] User Interface
```

## Model Architecture

### Overview

The model is a Random Forest Regressor wrapped in a scikit-learn pipeline that includes comprehensive preprocessing transformations. The pipeline uses a `TransformedTargetRegressor` to apply log transformation to the target variable, addressing the right-skewed distribution of car prices.

### Preprocessing Pipeline

The preprocessing pipeline consists of multiple specialized transformers organized using scikit-learn's `ColumnTransformer`:

#### 1. Numeric Feature Engineering (`BaseNumericFeatures`)

Extracts and normalizes numeric features from string-formatted inputs:

- **Engine Capacity**: Extracts numeric value from strings like "1197 CC"
- **Max Power**: Extracts numeric value from strings like "82 bhp"
- **Mileage Normalization**: Converts mileage to a standardized unit (kmpl) by:
  - Extracting numeric values and units from strings
  - Applying fuel-specific conversion factors:
    - Petrol: km/kg ÷ 0.74
    - Diesel: km/kg ÷ 0.832
    - LPG: km/kg ÷ 0.54
    - CNG: km/kg ÷ 0.128
  - This normalization accounts for different energy densities of fuel types
- **Age Calculation**: Converts manufacturing year to vehicle age (2026 - year)
- **Feature Dropping**: Removes redundant `year` and `fuel` columns after transformation

#### 2. Interaction Features (`InteractionFeatures`)

Creates multiplicative and ratio-based features to capture non-linear relationships:

- **Engine-Mileage Ratio**: `engine / (mileage + ε)` - Captures the trade-off between engine power and fuel efficiency
- **Kilometers-Age Interaction**: `km_driven × age` - Models cumulative wear and depreciation

These interaction terms help the model capture complex relationships that individual features cannot represent.

#### 3. Categorical Feature Processing

**Seats Transformation**
- Groups seats into categories: `less_than_five`, `five`, `more_than_five`
- Applies median imputation for missing values
- One-hot encoding for final representation

**Brand Effect (`NameTransformation`)**
- Extracts brand name (first word) from full car name
- Groups rare brands (frequency < threshold) into "other" category
- Applies Target Encoding with smoothing parameter (smoothing=10) to prevent overfitting
- StandardScaler normalization after encoding

Target encoding is used here because brand names have high cardinality and exhibit strong predictive power. The smoothing parameter balances between the category mean and the global mean, reducing overfitting risk.

**Fuel Type**
- Groups rare fuel types (CNG, LPG) into "other"
- One-hot encoding

**Owner Type**
- Consolidates owner categories:
  - "Third Owner" → "Third & Above Owner"
  - "Fourth & Above Owner" → "Third & Above Owner"
  - "Test Drive Car" → "First Owner"
- Ordinal encoding with ordered categories: ["Third & Above Owner", "Second Owner", "First Owner"]
- Unknown values encoded as -1

**Seller Type & Transmission**
- Direct one-hot encoding without prior transformation

#### 4. Numeric Feature Post-Processing

After interaction feature creation, numeric features undergo:

1. **Median Imputation**: Handles missing values with median (robust to outliers)
2. **Log Transformation**: `log1p(x)` transformation to address right-skewed distributions
3. **Standardization**: `StandardScaler` for zero mean and unit variance

### Model Selection and Training

#### Baseline Models Evaluated

Three regression models were evaluated using 10-fold cross-validation:

1. **Ridge Regression**: Linear model with L2 regularization
2. **Decision Tree Regressor**: Non-linear single-tree model
3. **Random Forest Regressor**: Ensemble of decision trees

Random Forest demonstrated superior performance and was selected as the final model.

#### Hyperparameter Optimization

Hyperparameter tuning was performed using `RandomizedSearchCV` with 200 iterations:

**Search Space:**
- `n_estimators`: Uniform distribution [50, 800]
- `max_depth`: [2, 4, 8, 16, 24, 32]
- `min_samples_leaf`: Uniform distribution [2, 150]
- `max_features`: ["sqrt", "log2", 0.3, 0.6, 0.8]
- `name__regroup__threshold`: Uniform distribution [1, 31]

**Cross-Validation Strategy:**
- 10-fold K-Fold cross-validation
- Shuffled splits with random_state=42 for reproducibility
- Scoring metric: Negative Root Mean Squared Error

#### Target Transformation

The model uses `TransformedTargetRegressor` with:
- Forward transformation: `log1p(y)` - Natural logarithm of (1 + price)
- Inverse transformation: `expm1(y_pred)` - Exponential of prediction minus 1

This transformation addresses the right-skewed price distribution, improving model performance on the log scale while maintaining interpretability through inverse transformation.

### Model Performance

**Training Set Metrics:**
- RMSE: 137,623.25
- MAE: 55,700.84
- R²: 0.97

**Cross-Validation Metrics (10-fold):**
- RMSE: 162,935.01 (mean)
- R²: 0.96 (mean)

**Test Set Metrics:**
- RMSE: 148,360.72
- MAE: 71,207.18
- R²: 0.96

### Feature Importance

Analysis of the trained Random Forest model reveals the following feature importance ranking:

1. **Mileage** - Fuel efficiency is a primary price determinant
2. **km_driven_age_interaction** - Cumulative wear factor
3. **engine_mileage_ratio** - Power-to-efficiency trade-off
4. **brand_effect** - Brand reputation and market perception
5. **km_driven** - Direct usage indicator
6. **age** - Depreciation factor

### Model Serialization

The complete pipeline (preprocessing + model) is serialized using `joblib` to `models/rf_model_pipeline_v1.pkl`. This ensures that all preprocessing steps are automatically applied during inference, maintaining consistency between training and prediction.

## Technical Implementation Details

### Backend Implementation

The backend service (`back-end/app/model.py`) implements a `CarPriceModel` class that:

1. **Pickle Compatibility**: Registers custom transformer classes in the module namespace to ensure proper deserialization of the pickled pipeline
2. **Model Loading**: Loads the serialized pipeline from disk
3. **Feature Validation**: Extracts expected feature names from the loaded model
4. **Prediction Interface**: Provides a `predict()` method that accepts pandas DataFrames

The FastAPI application (`back-end/app/main.py`) configures scikit-learn to output pandas DataFrames during transformation, ensuring compatibility with the training pipeline that was designed to work with DataFrames.

### Frontend Implementation

The Streamlit frontend (`front-end/app.py`) provides:

- Form-based input collection for all required car features
- HTTP POST requests to the backend API
- Error handling and user feedback
- Price conversion (dividing by 73, likely for currency conversion)

### Data Science Workflow

The project follows a structured notebook-based workflow:

1. **00_get_data.ipynb**: Data acquisition and initial loading
2. **01_explore_and_split_data.ipynb**: Exploratory data analysis and stratified train/test split
3. **02_understand_and_visualize_data.ipynb**: Data visualization and distribution analysis
4. **03_clean_data.ipynb**: Missing value handling and data quality improvements
5. **04_statistical_testing.ipynb**: Statistical hypothesis testing and validation
6. **05_feature_engineering.ipynb**: Feature creation and transformation design
7. **06_ml_pipeline.ipynb**: Model training, evaluation, and serialization

### Dependencies

**Core ML Stack:**
- scikit-learn >= 1.8.0: Machine learning pipeline and models
- pandas >= 2.3.3: Data manipulation
- numpy >= 2.3.5: Numerical computations
- category-encoders >= 2.9.0: Target encoding implementation

**Web Stack:**
- FastAPI >= 0.128.0: Backend API framework
- Streamlit >= 1.52.2: Frontend web framework
- requests >= 2.32.5: HTTP client

**Analysis Stack:**
- matplotlib >= 3.10.8: Visualization
- seaborn >= 0.13.2: Statistical visualization
- scipy >= 1.16.3: Statistical functions
- statsmodels >= 0.14.6: Statistical modeling
- pingouin >= 0.5.5: Statistical testing

## Deployment

The project uses Docker Compose for containerized deployment:

- **Backend Container**: Exposes port 8000, serves FastAPI application
- **Frontend Container**: Exposes port 8501, serves Streamlit application
- **Service Communication**: Frontend connects to backend via internal Docker network (`http://backend:8000`)

## Usage

### Local Development

1. Install dependencies using `uv` (Python >= 3.13 required)
2. Run notebooks sequentially to reproduce the model training process
3. Start services using Docker Compose: `docker-compose up`

### API Usage

```python
import requests

payload = {
    "name": "Hyundai i20",
    "year": 2018,
    "km_driven": 50000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage": "18.5 kmpl",
    "engine": "1197 CC",
    "max_power": "82 bhp",
    "torque": "",
    "seats": 5.0
}

response = requests.post("http://localhost:8000/predict", json=payload)
predicted_price = response.json()["predicted_price"]
```
