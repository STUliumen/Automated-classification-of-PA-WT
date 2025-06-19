import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

path = 'filtered_features.csv'
data = pd.read_csv(path)

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

le = LabelEncoder()
y = le.fit_transform(y)

k=0

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(8,), random_state=seed,
                max_iter=3000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accu =accuracy_score(y_test, y_pred)
if accu>k:
    k = accu
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f'Test Accuracy: {accu}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print('Confusion Matrix:')
    print(conf_matrix)

