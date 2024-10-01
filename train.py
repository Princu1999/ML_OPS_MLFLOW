import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('data/hour.csv')


#df['day_night'] = df['hr'].apply(lambda x: 'day' if 6 <= x <= 18 else 'night')

df.drop(['instant', 'casual', 'registered'], axis=1, inplace=True)
df['dteday'] = pd.to_datetime(df.dteday)
df['season'] = df.season.astype('category')
df['holiday'] = df.holiday.astype('category')
df['weekday'] = df.weekday.astype('category')
df['weathersit'] = df.weathersit.astype('category')
df['workingday'] = df.workingday.astype('category')
df['mnth'] = df.mnth.astype('category')
df['yr'] = df.yr.astype('category')
df['hr'] = df.hr.astype('category')
df.drop(columns=['dteday'], inplace=True)
# Separating features and target variable
X = df.drop(columns=['cnt']) # Features
y = df['cnt'] # Target

X = df.drop(columns=['cnt']) # Features
y = df['cnt'] # Target



numerical_features = ['temp']
numerical_pipeline = Pipeline([
('imputer', SimpleImputer(strategy='mean')), # Impute missing values with mean
('scaler', MinMaxScaler()) # Normalize using MinMaxScaler
])

X[numerical_features] = numerical_pipeline.fit_transform(X[numerical_features])
# Categorical features
categorical_features = ['season']
categorical_pipeline = Pipeline([
('imputer', SimpleImputer(strategy='most_frequent')),
('onehot', OneHotEncoder(sparse_output=False, drop='first'))
])
# Transforming above
X_encoded = categorical_pipeline.fit_transform(X[categorical_features])
# Converting it to a dataframe

X_encoded = pd.DataFrame(X_encoded,
columns=categorical_pipeline.named_steps['onehot'].get_feature_names_out(categorical_features))
# Encoded categorical features + Numerical features
X = pd.concat([X.drop(columns=categorical_features), X_encoded], axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
random_state=42)

mlflow.set_experiment("my_experiment")

def train_linear_regression():
    with mlflow.start_run() as run:
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f'Mean Squared Error LR: {mse}')
        print(f'R-squared LR: {r2}')
        # Log parameters, metrics, and model
        mlflow.log_param("model", "LinearRegression")
        mlflow.log_metric("mse_lr", mse)
        mlflow.log_metric("r2_lr", r2)
        mlflow.sklearn.log_model(model, "model_lr")
        mlflow.register_model(f"runs:/{run.info.run_id}/linear_regression_model", "LinearRegressionModel")

        print(f"Run URL: {mlflow.active_run().info.artifact_uri}")

def train_random_forest():
    with mlflow.start_run() as run:
        model_rf = RandomForestRegressor()
        model_rf.fit(X_train, y_train)

        # Predict and evaluate
        y_pred_rf = model_rf.predict(X_test)
        mse_rf = mean_squared_error(y_test, y_pred_rf)
        r2_rf = r2_score(y_test, y_pred_rf)
        print(f'Mean Squared Error RF: {mse_rf}')
        print(f'R-squared RF: {r2_rf}')
        # Log parameters, metrics, and model
        mlflow.log_param("model_rf", "RandomForestRegressor")
        mlflow.log_metric("mse_rf", mse_rf)
        mlflow.log_metric("r2_rf", r2_rf)
        mlflow.sklearn.log_model(model_rf, "model_rf")
        mlflow.register_model(f"runs:/{run.info.run_id}/random_forest_model", "RandomForestModel")

        print(f"Run URL: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    train_linear_regression()
    train_random_forest()


