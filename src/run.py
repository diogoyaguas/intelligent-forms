import pickle
import json
from itertools import zip_longest
from flask import Flask, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

cors = CORS(app, resources={
            r"/classify": {"origins": "http://10.227.107.138:8025"}})


@app.route('/classify', methods=['POST'])
@cross_origin(origin='10.227.107.138', headers=['Content-Type', 'Authorization'])
def classify():

    if request.method == 'POST':
        req = json.loads(request.data.decode("utf-8"))
        text = req.get('text')
        model_name = req.get('model')
        task = req.get('task')

        print(model_name)
        print(task)

        with open(f"{model_name}/{task}/model/vectorizer.pkl", "rb") as input_file:
            vectorizer = pickle.load(input_file)

        with open(f"{model_name}/{task}/model/label_vectorizer.pkl", "rb") as input_file:
            label_vectorizer = pickle.load(input_file)

        with open(f"{model_name}/{task}/model/svm.pkl", "rb") as input_file:
            clf = pickle.load(input_file)

        if(model_name == "/home/tro/server/out/20210616_103813"):
            print(clf.estimators_)
            for x in clf.estimators_:
                print("HEY")
                print(x)
                y_predict_proba = x.predict_proba(vectorizer.transform([text]))
                print(y_predict_proba)

        classes = {idx: value for idx, value in enumerate(
            label_vectorizer.classes_)}
        y_predict_proba = clf.predict_proba(vectorizer.transform([text]))
        res = [dict(zip_longest(classes, probs))
               for probs in y_predict_proba][0]
        for key in range(len(res)):
            res[classes[key]] = res.pop(key)

        print(res)
        return res


if __name__ == "__main__":
    app.run()
