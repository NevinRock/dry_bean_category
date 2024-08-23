from sympy import false
from ucimlrepo import fetch_ucirepo
import datetime
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np
import pprint


def predict_scoreing(y_true: pd.Series, y_pred: pd.Series) -> None:
    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate F1-Score (assuming multi-class classification)
    f1 = f1_score(y_true, y_pred, average='macro')

    # Generate a classification report for detailed metrics
    class_report = classification_report(y_true, y_pred, zero_division=0, output_dict=False)

    # Print detailed metrics
    pprint.pprint({"Accuracy": accuracy, "F1-Score": f1, "Classification Report": class_report})

    # Combine F1-Score and Accuracy into a single score
    score = 0.5 * f1 + 0.5 * accuracy

    # Print the combined score
    print("\n\n" + "The accurate score is: " + str(score))


# Define a callback function to print the results of the validation set
def print_validation_result(env: pd.Series) -> None:
    result = env.evaluation_result_list[-1]

def coutn_unique_value(s: pd.Series) -> None:
    # 获取唯一值
    unique_values = s.unique()
    print("Unique values:", unique_values)

    # 获取每个值的计数
    value_counts = s.value_counts()
    print("Unique values and their counts:")
    print(value_counts)

def encoder(s: pd.Series) -> pd.Series:
    category = {'SEKER': 1,
                'BARBUNYA': 2,
                'BOMBAY': 3,
                'CALI': 4,
                'HOROZ': 5,
                'SIRA': 6,
                'DERMASON': 7}
    return s.map(category)


def decoder(s: pd.Series) -> pd.Series:

    reverse_category = {
        1: 'SEKER',
        2: 'BARBUNYA',
        3: 'BOMBAY',
        4: 'CALI',
        5: 'HOROZ',
        6: 'SIRA',
        7: 'DERMASON'
    }

    return s.map(reverse_category)



# fetch dataset
dry_bean = fetch_ucirepo(id=602)

# data (as pandas dataframes)
X = dry_bean.data.features
y = dry_bean.data.targets

y_encodered= encoder(y["Class"])

df = pd.concat([X, y_encodered], axis=1)
df_output = pd.concat([X, y], axis=1)

# df.to_csv('dry_bean.csv', index=False)



X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, -1], test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, y_train)
test_data = lgb.Dataset(X_test, y_test)

params = {
        "boosting_type": "gbdt",
        "objective": "regression",
        "metric": "root_mean_squared_error",
        "max_depth": 7,
        "learning_rate": 0.02,
        "verbose": 0,
    }

gbm = lgb.train(
        params,
        train_data,
        num_boost_round=30000,
        valid_sets=[test_data],
        callbacks=[print_validation_result],
    )

prediction = np.round(gbm.predict(df.iloc[:, :-1]))

df_output["quality_prediction"] = prediction
df_output["quality_prediction"] = decoder(df_output["quality_prediction"])
df_output["quality_prediction"] = df_output["quality_prediction"].where(pd.notna(df_output["quality_prediction"]), "None")



df_output.to_csv(("output/output_"+datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + ".csv"), index=False)


# predict_scoreing(y_encodered, pd.Series(prediction))

predict_scoreing(df_output["Class"], df_output["quality_prediction"])