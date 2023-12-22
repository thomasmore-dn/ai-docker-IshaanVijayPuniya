import boto3
from botocore.exceptions import ClientError

class ProcessImage:
    def __init__(self, image, image_name):
        self.image = image
        self.image_name = image_name

    def detect_text(self):
        texts = list()

        # Create a session and client for Amazon Rekognition
        session = boto3.Session()
        client = session.client('rekognition', "eu-west-1")

        try:
            # Use Amazon Rekognition to detect text in the image
            response = client.detect_text(Image={'Bytes': self.image.read()})
            detection = response['TextDetections']

            # Extract and append WORD-type text to the list
            for text in detection:
                if text["Type"] == "WORD":
                    texts.append(text["DetectedText"])

            print(f"Found {len(texts)} texts in {self.image_name}")

        except ClientError:
            # Handle client errors when detecting text
            print(f"Couldn't detect text in {self.image_name}")
            texts = "No text found in this image"
        else:
            return texts

    def text_to_speech(self, image_text):
        # Create a session and client for Amazon Polly
        session = boto3.Session()
        client = session.client('polly', "eu-west-1")

        try:
            # Use Amazon Polly to synthesize speech from the provided text
            response = client.synthesize_speech(Text=image_text,
                                                OutputFormat="mp3", VoiceId="Matthew")

        except ClientError as error:
            # Handle client errors when synthesizing speech
            print(error)

        if "AudioStream" in response:
            # If audio stream is available, write it to a file
            audio = response['AudioStream'].read()
            file_name ='static/converted_text.mp3'

            with open(file_name, 'wb') as file:
                file.write(audio)

            return
