from flask import Flask, render_template, request
import boto3
import model

app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route for processing the uploaded image
@app.route('/process', methods=['GET', 'POST'])
def process_image():
    # Check if the request method is POST
    if request.method == 'POST':
        # Get the uploaded image from the request
        image = request.files['uploadedImage']

        # Check if the filename is empty
        if image.filename == "":
            return  # Add a response or redirect if the filename is empty

        # Get the name of the uploaded image
        image_name = image.filename

        # Create an instance of the ProcessImage class from the model module
        processor = model.ProcessImage(image, image_name)

        # Use the detect_text method to extract words from the image
        extracted_words = processor.detect_text()

        # Join the extracted words into a single string
        extracted_text = ' '.join(extracted_words)

        # Use the text_to_speech method to convert the extracted text to audio
        audio = processor.text_to_speech(extracted_text)

        # Render the processed.html template with the extracted text
        return render_template('processed.html', extracted_text=extracted_text)



# Run the Flask application if this script is executed
if __name__ == '__main__':
    app.run(debug=True)
