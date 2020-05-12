# import dependencies
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.svm import SVR


def main():
    # sets the seed to be 33
    np.random.seed(33)

    # reads in the csv file for training
    df = pd.read_csv("train.csv")
    dv = pd.read_csv("test.csv")

    # one hot labels the data so it can be used
    df = pd.get_dummies(df)
    dv = pd.get_dummies(dv)
    dv = dv.reindex(columns=df.columns).fillna(0)
    dv = dv.drop("SalePrice", axis=1) # Doesn't need SalePrice as this is just submission data

    # sets the data up to be used
    Y = df.loc[:, "SalePrice"]
    X = df.drop("SalePrice", axis=1)

    # sets up training and validation sets
    m = len(X)
    train_range = round(m / 1.05)
    X_validation = X.loc[train_range:]
    X_validation = X_validation.fillna(0)
    Y_validation = Y.loc[train_range:]

    X = X.loc[:train_range]
    X = X.fillna(0)
    Y = Y.loc[:train_range]

    # sets the data to be submitted
    X_submission = dv[dv.columns]
    X_submission = X_submission.fillna(0)

    # applies preprocessing to the data
    scalar = preprocessing.MinMaxScaler()

    # Creates the model and prints the accuracy
    model = SVR(kernel="linear")
    model.fit(X, Y)
    print("Model accuracy: " + str(model.score(X,Y)))

    # Predicts the validation set
    predictions = model.predict(X_validation)
    print(predictions[:5])
    print(Y_validation[:5])

    sub_col1 = dv.iloc[:, 0]
    sub_col2 = model.predict(X_submission)

    # Creates the table for kaggle submission
    d = {"ID": sub_col1, "SalePrice": sub_col2}
    submission = pd.DataFrame(data=d)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()
