from flask import Flask, request, jsonify, render_template
from model import predict_pcos

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Serves the frontend

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    image_file = request.files.get('image')

    if not image_file:
        return jsonify({'error': 'No image file provided'}), 400

    # Save the file temporarily
    image_path = f"./temp/{image_file.filename}"
    image_file.save(image_path)

    # Call the predict_pcos function
    prediction = predict_pcos(image_path)

    # Return the result
    return jsonify({'prediction': prediction})

if __name__ == "__main__":
    app.run(debug=True)
