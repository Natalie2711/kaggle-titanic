import pandas as pd
# import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
# plt.style.use("ggplot")


train = pd.read_csv("train.csv",
                    header=0,
                    index_col="PassengerId",
                    usecols=["PassengerId",
                             "Survived",
                             "Pclass",
                             "Sex"])
train["Gender"] = [0 if _ == "female" else 1 for _ in train["Sex"]]
train.drop(["Sex"], axis=1, inplace=True)

X_train = train.loc[:, ["Gender", "Pclass"]]
Y_train = train.Survived

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(X_train, Y_train)

test = pd.read_csv("test.csv",
                   header=0,
                   usecols=["PassengerId",
                            "Pclass",
                            "Sex"])

test["Gender"] = [0 if _ == "female" else 1 for _ in test["Sex"]]
test.drop(["Sex"], axis=1, inplace=True)

x_test = test.loc[:, ["Gender", "Pclass"]]
y_test = knn.predict(x_test)

df = pd.DataFrame({"PassengerId": test.PassengerId, "Survived": y_test}).set_index("PassengerId")
df.to_csv("submission.csv")
