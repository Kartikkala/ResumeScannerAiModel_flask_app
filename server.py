from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl") 

@app.route("/analyze", methods=["POST"])
def analyze():
    if 'resume' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['resume']
    pdf_bytes = file.read()

    prediction = model.predict([pdf_bytes]) 

    return jsonify({
        "prediction": prediction[0]
    })

if __name__ == "__main__":
    app.run(debug=True, port=8888)
