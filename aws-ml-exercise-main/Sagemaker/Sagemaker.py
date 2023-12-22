import pandas as pd
from sklearn.model_selection import train_test_split
import shap
from io import StringIO
import sagemaker
from sagemaker.sklearn import SKLearn

# Load your dataset
X, y = shap.datasets.adult()

# Use only the first 10 rows
X = X.iloc[:10, :]
y = y[:10]

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up SageMaker session and role
sagemaker_session = sagemaker.Session()
role = sagemaker.get_execution_role()

# Use the default SageMaker bucket
bucket_name = sagemaker_session.default_bucket()
train_prefix = 'train'
test_prefix = 'test'

# Upload training dataset to a temporary local file
train_csv_buffer = StringIO()
Xy_train = pd.concat([X_train, pd.Series(y_train, name='target')], axis=1)
Xy_train.to_csv(train_csv_buffer, index=False)
temp_train_file = '/tmp/your_train_dataset.csv'
train_csv_buffer.getvalue().encode('utf-8')
with open(temp_train_file, 'w') as f:
    f.write(train_csv_buffer.getvalue())

# Upload training dataset to S3 using SageMaker Session
train_s3_path = sagemaker_session.upload_data(
    path=temp_train_file,
    bucket=bucket_name,
    key_prefix=f'{train_prefix}/your_train_dataset.csv'
)

# Upload test dataset to S3
test_csv_buffer = StringIO()
Xy_test = pd.concat([X_test, pd.Series(y_test, name='target')], axis=1)
Xy_test.to_csv(test_csv_buffer, index=False)
temp_test_file = '/tmp/your_test_dataset.csv'
test_csv_buffer.getvalue().encode('utf-8')
with open(temp_test_file, 'w') as f:
    f.write(test_csv_buffer.getvalue())

# Upload test dataset to S3 using SageMaker Session
test_s3_path = sagemaker_session.upload_data(
    path=temp_test_file,
    bucket=bucket_name,
    key_prefix=f'{test_prefix}/your_test_dataset.csv'
)

# Define the script to be run by the SageMaker Estimator
script_path = 'script.py'  # Replace with your actual script file

# Set up the SKLearn Estimator
estimator = SKLearn(entry_point=script_path,
                    role=role,
                    instance_count=1,
                    instance_type='ml.m4.4xlarge',  # Adjust the instance type if needed
                    framework_version='0.23-1')

# Define S3 URIs for training data
train_data_uri = train_s3_path
test_data_uri = test_s3_path

# Train the model
estimator.fit({'train': train_data_uri, 'test': test_data_uri})

# Deploy the model to an endpoint with the name 'xgb-fraud-model'
predictor = estimator.deploy(initial_instance_count=1,
                             instance_type='ml.m4.xlarge',
                             endpoint_name='xgb-fraud-model-2023-01-12')
