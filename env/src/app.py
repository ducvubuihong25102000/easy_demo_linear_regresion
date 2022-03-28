from operator import mod
from unicodedata import numeric
import pandas as pd
import statsmodels.formula.api as smf
import random
import wget
from pathlib import Path
import src.graphing as graphing
import plotly.express
import joblib

#get data and extension library for this program 
url1 = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/graphing.py"
url2 = "https://raw.githubusercontent.com/MicrosoftDocs/mslearn-introduction-to-machine-learning/main/Data/doggy-boot-harness.csv"
path1 = "D:\code\Python\env\src\graphing.py"
path2 = "D:\code\Python\env\src\doggy-boot-harness.csv"
if not Path(path1).is_file():
    wget.download(url1)
elif not Path(path2).is_file():
    wget.download(url2)

data = {
    'boot_size': [39, 38, 37, 39, 38, 35, 37, 36, 35, 40,
                  40, 36, 38, 39, 42, 42, 36, 36, 35, 41,
                  42, 38, 37, 35, 40, 36, 35, 39, 41, 37,
                  35, 41, 39, 41, 42, 42, 36, 37, 37, 39,
                  42, 35, 36, 41, 41, 41, 39, 39, 35, 39
                  ],
    'harness_size': [58, 58, 52, 58, 57, 52, 55, 53, 49, 54,
                     59, 56, 53, 58, 57, 58, 56, 51, 50, 59,
                     59, 59, 55, 50, 55, 52, 53, 54, 61, 56,
                     55, 60, 57, 56, 61, 58, 53, 57, 57, 55,
                     60, 51, 52, 56, 55, 57, 58, 57, 51, 59
                     ]
}

# nothing_list = []
# random.randrange(0,100)
# for i in range(0,50):
#     n = random.randrange(0,100)
#     nothing_list.append(n)
# data["nothing"] = nothing_list


formular = "boot_size ~ harness_size"
df = pd.DataFrame(data)

model = smf.ols(formular, df)
trained_model = model.fit()
if not hasattr(trained_model, 'params'):
    print("Model selected but it does not have parameters set. We need to train it!")
else:
    print("model is trained")
    print("The following model parameters have been found:\n" +
          f"Line slope: {trained_model.params[1]}\n" +
          f"Line Intercept: {trained_model.params[0]}")

print(df)
# graphing.scatter_2D(df, label_x="harness_size",
#                     label_y="boot_size",
#                     trendline=lambda x: trained_model.params[1] *
#                     x + trained_model.params[0]
#                     )


#predict
harness_size_example = {"harness_size":52.5}
approximate_boot_size = trained_model.predict(harness_size_example)
print(approximate_boot_size[0])

#visualize data in 2D graph by "plotly"
#if dont have parameter to determine which is x axis and y axis it will take [index] as default for x axis
#and in if there is more than 2 rows have the same data, it will show as one dot in the graph without change color
plotly.express.scatter(df,x="boot_size",y="harness_size")
#create a new column "imperial_harness_size"
df["imperial_harness_size"] = df.harness_size/2.54
plotly.express.scatter(df,x="boot_size",y="imperial_harness_size")


#saving trained_model
trained_model_filename = "./avalanche_dog_bootsize.pkl"
if not Path(trained_model_filename).is_file():
    joblib.dump(trained_model,trained_model_filename)
    print("Model is saved")
else:
    print("Model is saved before")

#Load model from pkl file. We can use this load file in self define method to run the estimation
model_loaded = joblib.load(trained_model_filename)
print("We have loaded a model with the following parameters:")
print(model_loaded.params)
print (model_loaded)


#Self define function to use the model when it's trained and stored in pkl file
def load_model_and_predict(harness_size: numeric) -> numeric:
    """This method load the trained model from pkl file with input is harness_size of a dog
    then it will estimate the boot_size by Linear Regression model with ols method(ordinary least squared)"""
    loaded_model = joblib.load(trained_model_filename)

    print("The model's params")
    print(loaded_model.params)

    inputs = {"harness_size":[harness_size]} 

    predicted_boot_size = loaded_model.predict(inputs)[0]

    return predicted_boot_size

predicted_boot_size = load_model_and_predict(45)

print("Predicted dog boot size:", predicted_boot_size)