from flask import Flask,render_template,redirect,session,request,url_for
import os

from src.utils import load_object
from src.pipeline.predict_pipeline import predictPipeline,CustomData

from dotenv import load_dotenv
load_dotenv()

label_encoder_path = os.path.join("artifacts", "label_encoder.pkl")
label_encoder = load_object(label_encoder_path)


app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY")

@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/predict", methods = ["GET","POST"])
def prediction():
    if request.method == "GET":
        return render_template("home.html")
    
    else:
        data = CustomData(
            N=request.form.get("N"),
            P=request.form.get("P"),
            K=request.form.get("K"),
            temperature=request.form.get("temperature"),
            humidity=request.form.get("humidity"),
            ph=request.form.get("ph"),
            rainfall=request.form.get("rainfall")
        )

        df = data.get_data_as_dataframe()
        print("dataframe:", df)

        pipeline = predictPipeline()
        pred = pipeline.predict(df)

        result = label_encoder.inverse_transform(pred.astype(int))

        session["predicted_crop"] = result[0]
        session["crop"] = True
        return redirect(url_for("result_page"))
    

@app.route("/result")
def result_page():
    if session.get("crop"):
        session.pop("crop")
        crop = session.get("predicted_crop")
        if crop is None:
            return redirect(url_for("hello"))
        return render_template("result.html", crop=crop)
    
    else:
        return redirect(url_for("prediction"))

if __name__ == "__main__":
    app.run(debug=True)