import pickle
import pandas as pd


def predict(sample):

    with open(
        "models/random_forest.pkl",
        "rb"
    ) as f:

        model = pickle.load(f)

    df = pd.DataFrame(
        [sample]
    )

    prediction = model.predict(df)[0]

    probability = model.predict_proba(df)[0][1]

    return prediction, probability