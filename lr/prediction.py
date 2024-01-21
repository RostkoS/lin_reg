import joblib


def predict(data):
    lr = joblib.load("lr/rf_model.sav")
    return lr.predict(data)
