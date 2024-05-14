import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    ConfusionMatrixDisplay,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from mlxtend.frequent_patterns import association_rules, fpgrowth


def load_and_preprocess_data():
    df = pd.read_csv("dataset.csv")

    print(df.shape)
    # print(df.duplicated().sum())  # 0
    print(f"Brakujące wartości: {df.isnull().sum().sum()}")

    mappings = {
        "Sex": {"Male": 0, "Female": 1},
        "Diet": {"Healthy": 0, "Average": 1, "Unhealthy": 2},
        "Hemisphere": {"Northern Hemisphere": 0, "Southern Hemisphere": 1},
        "Continent": {
            "Asia": 0,
            "Europe": 1,
            "South America": 2,
            "Australia": 3,
            "Africa": 4,
            "North America": 5,
        },
    }

    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping)

    df[["Systolic", "Diastolic"]] = (
        df["Blood Pressure"].str.strip().str.split("/", expand=True)
    )

    df["Systolic"] = df["Systolic"].astype(int)
    df["Diastolic"] = df["Diastolic"].astype(int)

    df = df.dropna()
    df = df.drop(columns=["Patient ID", "Blood Pressure", "Country"])

    return df


# --- Downsampling ---
# def balance_data(df):
#     df_majority = df[df["Heart Attack Risk"] == 0]
#     df_minority = df[df["Heart Attack Risk"] == 1]
#     df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)
#     df_balanced = pd.concat([df_majority_downsampled, df_minority])
#     return df_balanced


# --- Upsampling ---
# def balance_data(df):
#     true = df["Heart Attack Risk"].sum()
#     rows, cols = df.shape
#     new_true_amount = (rows - true) // true
#     new_rows = df[df["Heart Attack Risk"] == 1]
#     for i in range(new_true_amount):
#         df = pd.concat([df, new_rows])
#     return df


# --- SMOTE ---
def balance_data(df):
    sm = SMOTE(random_state=285806)

    X = df.drop(["Heart Attack Risk"], axis=1)
    y = df["Heart Attack Risk"]

    X_res, y_res = sm.fit_resample(X, y)

    df_res = pd.concat([X_res, y_res], axis=1)

    return df_res


def split_data(df):
    df = df.sample(frac=1)
    X = df.drop(["Heart Attack Risk"], axis=1)
    y = df["Heart Attack Risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def plot_history(history):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True, linestyle="--", color="grey")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", color="grey")
    plt.legend()

    plt.tight_layout()
    plt.show()


def rfc(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(random_state=285806)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    roc = roc_auc_score(y_test, predict) * 100
    print("Random Forest")
    print(f"Accuracy: {acc}%")
    print(f"AUC ROC: {roc}%")

    cm = confusion_matrix(y_test, predict)
    cfd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cfd.plot()
    plt.show()


def dtc(X_train, y_train, X_test, y_test):
    clf = DecisionTreeClassifier(random_state=285806)
    clf = clf.fit(X_train, y_train)
    predict = clf.predict(X_test)
    acc = accuracy_score(y_test, predict) * 100
    roc = roc_auc_score(y_test, predict) * 100
    print("Decision Tree")
    print(f"Accuracy: {acc}%")
    print(f"AUC ROC: {roc}%")

    cm = confusion_matrix(y_test, predict)
    cfd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
    cfd.plot()
    plt.show()


def gaussian(X_train, y_train, X_test, y_test):
    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    roc = roc_auc_score(y_test, y_pred) * 100
    print("Naive Bayes")
    print(f"Accuracy: {acc}%")
    print(f"AUC ROC: {roc}%")

    cm = confusion_matrix(y_test, y_pred)
    cfd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=gnb.classes_)
    cfd.plot()
    plt.show()


def knn(X_train, y_train, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, ax in zip([3, 5, 11], axes):
        knc = KNeighborsClassifier(n_neighbors=i)
        knc.fit(X_train, y_train)
        y_pred = knc.predict(X_test)

        acc = accuracy_score(y_test, y_pred) * 100
        roc = roc_auc_score(y_test, y_pred) * 100
        print(f"{i}NN")
        print(f"Accuracy: {acc}%")
        print(f"AUC ROC: {roc}%")

        cm = confusion_matrix(y_test, y_pred)
        cfd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knc.classes_)
        cfd.plot(ax=ax)
        ax.title.set_text(f"{i}-NN")

    plt.tight_layout()
    plt.show()


def neural_network(X_train, y_train, X_test, y_test):
    model = Sequential(
        [
            Dense(
                64,
                activation="relu",
                input_shape=(X_train.shape[1],),
                kernel_regularizer=l2(0.01),
            ),
            Dropout(0.5),
            Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
            Dense(32, activation="selu", kernel_regularizer=l2(0.01)),
            Dropout(0.5),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    early_stopping = EarlyStopping(monitor="val_loss", patience=10)
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0,
    )

    _, test_acc = model.evaluate(X_test, y_test)
    roc = roc_auc_score(y_test, model.predict(X_test)) * 100
    print("Neural Network")
    print(f"Accuracy: {test_acc * 100}%")
    print(f"AUC ROC: {roc}%")

    cm = confusion_matrix(
        y_test,
        np.round(model.predict(X_test)),
    )
    cfd = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    cfd.plot()
    plt.show()

    return history


def create_association_rules(df):
    selected_columns = [
        "Age",
        "Cholesterol",
        "Heart Rate",
        "Exercise Hours Per Week",
        "Sedentary Hours Per Day",
        "Income",
        "BMI",
        "Triglycerides",
        "Physical Activity Days Per Week",
        "Sleep Hours Per Day",
        "Systolic",
        "Diastolic",
    ]
    for col in selected_columns:
        df[col] = pd.cut(df[col], bins=3, labels=["Low", "Medium", "High"])

    for col in df.columns:
        if col in selected_columns:
            continue
        unique_values = df[col].nunique()
        labels = [i for i in range(unique_values)]
        df[col], _ = pd.cut(df[col], bins=unique_values, labels=labels, retbins=True)

    encoded_df = pd.get_dummies(df)

    frequent_itemsets = fpgrowth(encoded_df, min_support=0.008, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)

    rules = rules[rules["consequents"].apply(lambda x: "Heart Attack Risk_1" in str(x))]

    return rules


def plot_rules(rules, column1, column2):
    plt.figure(figsize=(10, 6))
    plt.scatter(rules[column1], rules[column2], alpha=0.5)
    plt.xlabel(column1)
    plt.ylabel(column2)
    plt.title(f"{column1} vs {column2}")
    plt.show()


def main():
    # --- Prepare data ---
    df = load_and_preprocess_data()
    # for column in df.columns:
    #     if column == "Heart Attack Risk":
    #         print(df[column].value_counts())

    # print("df.shape przed upsampling:\n", df.shape)
    # print(df.head())
    df = balance_data(df)
    # for column in df.columns:
    #     if column == "Heart Attack Risk":
    #         print(df[column].value_counts())
    # print("df.shape po upsampling:\n", df.shape)
    # df = df.sample(frac=1)
    # df["Heart Attack Risk"].value_counts().plot.bar()
    # plt.show()

    # --- Split and scale data ---
    X_train, X_test, y_train, y_test = split_data(df)
    X_train, X_test = scale_data(X_train, X_test)

    # --- Train and evaluate models ---
    rfc(X_train, y_train, X_test, y_test)
    dtc(X_train, y_train, X_test, y_test)
    gaussian(X_train, y_train, X_test, y_test)
    knn(X_train, y_train, X_test, y_test)
    history = neural_network(X_train, y_train, X_test, y_test)
    plot_history(history)

    # --- Association rules ---
    rules = create_association_rules(df)
    # print(rules)

    male_heart_attack_risk_rules = rules[
        rules["antecedents"].apply(lambda x: "Sex_1" in x)
    ].sort_values(by="confidence", ascending=False)

    print(male_heart_attack_risk_rules)  # []

    female_heart_attack_risk_rules = rules[
        rules["antecedents"].apply(lambda x: "Sex_0" in x)
    ].sort_values(by="confidence", ascending=False)

    print(female_heart_attack_risk_rules)  # [34672 rows x 10 columns]

    plot_rules(female_heart_attack_risk_rules, "lift", "confidence")

    diabetes_heart_attack_risk_rules = rules[
        rules["antecedents"].apply(lambda x: "Diabetes_1" in x)
    ].sort_values(by="confidence", ascending=False)

    print(diabetes_heart_attack_risk_rules)  # [507 rows x 10 columns]

    plot_rules(diabetes_heart_attack_risk_rules, "support", "lift")

    smoking_heart_attack_risk_rules = rules[
        rules["antecedents"].apply(lambda x: "Smoking_1" in x)
    ].sort_values(by="confidence", ascending=False)

    print(smoking_heart_attack_risk_rules)  # [17526 rows x 10 columns]

    plot_rules(smoking_heart_attack_risk_rules, "support", "confidence")

    obesity_heart_attack_risk_rules = rules[
        rules["antecedents"].apply(lambda x: "Obesity_1" in x)
    ].sort_values(by="confidence", ascending=False)

    print(obesity_heart_attack_risk_rules)  # [5 rows x 10 columns]


if __name__ == "__main__":
    main()
