import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open("breast_cancer_pred.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)[0]
    return render_template('index.html',
                           prediction_text=("The cancer is not malignant." if (prediction == 0) else "The cancer is malignant."),
                           mean_perimeter=request.form["mean perimeter"],
                           mean_radius=request.form["mean radius"],
                           mean_texture=request.form["mean texture"],
                           mean_area=request.form["mean area"],
                           worst_texture=request.form["worst texture"],
                           worst_perimeter=request.form["worst perimeter"],
                           worst_area=request.form["worst area"],
                           texture_error=request.form["texture error"],
                           perimeter_error=request.form["perimeter error"],
                           area_error=request.form["area error"]
                           )


if __name__ == "__main__":
    app.run(debug=True)
