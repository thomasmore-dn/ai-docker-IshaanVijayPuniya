# aws-ml-exercise part 1 and PART 2 IN SAGEMAKER FOLDER
Ishaan vijay Puniya r0865976 
 AWS exercise that  uses 2 services Rekognition and Polly.

app.py ->  python flask web application  
model.py -> AWS services to process the image  

# HOW TO USE
Create an AWS CLI IAM with __AmazonPollyFullAccess__ and __AmazonRekognitionFullAccess__.  
Pip install
-boto3
-flask  
Go in a console to the main file and enter __$ flask run__  
Go to the link that Flask is running on, example: "http://110.0.0.1:5000/"  
Upload a picture (.jpg or .jpeg) with  a quote, upload on the web application and submit!
