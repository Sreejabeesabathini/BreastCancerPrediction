from flask import Flask, request, render_template
import numpy as np
import joblib

# Load the trained model
model = joblib.load("model.pkl")

# Flask app
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")  # Ensure 'index.html' exists in a 'templates' folder


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input from the form
        feature_input = request.form.get("features")
        if not feature_input:
            return render_template("index.html", prediction_text="Error: Please provide the features.")

        # Split the input into individual features
        features = feature_input.split(",")
        if len(features) != 31:
            return render_template("index.html", prediction_text="Error: Please provide exactly 31 features.")

        # Convert to floats
        features = [float(f) for f in features]

        # Convert to numpy array and reshape for model
        np_features = np.array(features).reshape(1, -1)

        # Make prediction
        pred = model.predict(np_features)

        # Map prediction to output
        output = "Breast Cancer Detected" if pred[0] == 1 else "No Signs of Breast Cancer"

        return render_template("index.html", prediction_text=output)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


# Run Flask app
if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False in production