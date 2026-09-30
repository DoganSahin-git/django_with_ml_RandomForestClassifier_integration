## Django with ML RandomForestClassifier Integration

This is a small Django project that classifies iris flowers with a scikit-learn model, through both a web form and a JSON API. 
I ran it with Django's test client: training reached about 0.90 test accuracy, and both the form and the API returned correct predictions.

Download everything and create python virtual environment inside the directory:
```
python3 -m venv venv
source venv/bin/activate
```
Install the required packages:
```
pip install -r requirements.txt
```
Train the model and save while offline (saves predictor/model.joblib)
```
python train_model.py
```
Run Django server
```
python manage.py runserver   
```
It will serve at http://127.0.0.1:8000
You can either fill the form manually or send an api request with the given values.

Trigger the API from the terminal while server is up:
```
curl -X POST http://127.0.0.1:8000/api/predict/ \
  -H "Content-Type: application/json" \
  -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'
```
