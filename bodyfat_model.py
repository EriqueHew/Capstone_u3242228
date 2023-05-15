# Import Python packages
import pandas as pd
import sklearn
import matplotlib.pyplot as pyplot

# Load libraries
from matplotlib import pyplot as plt
from sklearn import linear_model, preprocessing
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

# Data file import
bodyfat = pd.read_csv('bodyfat.csv')

# Attribute to be predicted
predict = "bf"

# Dataset/Column to be Predicted, X is all attributes and y is the features
#x = np.array(bodyfat.drop([predict], 1)) # Will return a new data frame that doesnt have hd in it
#y = np.array(bodyfat[predict])
le = preprocessing.LabelEncoder()
agegroup = le.fit_transform(list(bodyfat["AgeGroup"]))
density = le.fit_transform(list(bodyfat["Density"]))
body_fat = le.fit_transform(list(bodyfat["BodyFat"]))
age = le.fit_transform(list(bodyfat["Age"]))
weight = le.fit_transform(list(bodyfat["Weight"]))
height = le.fit_transform(list(bodyfat["Height"]))
neck = le.fit_transform(list(bodyfat["Neck"]))
chest = le.fit_transform(list(bodyfat["Chest"]))
abdomen = le.fit_transform(list(bodyfat["Abdomen"]))
hip = le.fit_transform(list(bodyfat["Hip"]))
thigh = le.fit_transform(list(bodyfat["Thigh"]))
knee = le.fit_transform(list(bodyfat["Knee"]))
ankle = le.fit_transform(list(bodyfat["Ankle"]))
biceps = le.fit_transform(list(bodyfat["Biceps"]))
forearm = le.fit_transform(list(bodyfat["Forearm"]))
wrist = le.fit_transform(list(bodyfat["Wrist"]))

#x = list(zip(age, body_fat))
x = list(zip(density, body_fat, age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, biceps, forearm, wrist))
y = list(agegroup)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'Accuracy'

# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.20, random_state=seed)
#splitting 20% of our data into test samples. If we train the model with higher data it already has seen that information and knows

# Check with  different Scikit-learn classification algorithms
models = []
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []

for name, model in models:
    kfold = KFold(n_splits=num_folds,shuffle=True,random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

# Compare Algorithms' Performance
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

# Make predictions on validation/test dataset
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()

best_model = gb
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
model_accuracy = accuracy_score(y_test, y_pred)
print("Best Model Accuracy Score on Test Set:", model_accuracy)

#Model Evaluation Metric 1
print(classification_report(y_test, y_pred))

#Model Evaluation Metric 2
#Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
cm = confusion_matrix(y_test, y_pred)
# Create Matrix for Accuracy, Precision, F1_Score
Accuracy = accuracy_score(y_test, y_pred) * 100
print ("Accuracy measure % is", Accuracy)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

#Model Evaluation Metric 3
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve

#Check actual/ground truth vs predicted diagnosis
for x in range(len(y_pred)):
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x],)

#Computer ROC-AUC metric
#For multiclass classifiers (More than 2 classes)
#you need to binarize the labels with OVR strategy(One vs. Rest of the classes)
from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(y_train)
y_onehot_test = label_binarizer.transform(y_test)
y_onehot_test.shape  # (n_samples, n_classes)

#ROC curve for class 3 ('YoungAge')
from sklearn.metrics import RocCurveDisplay
class_id = 3
class_of_interest = "Age"

RocCurveDisplay.from_predictions(y_onehot_test[:, class_id],y_pred,
                                 name=f"{class_of_interest} vs the rest", color="darkorange",)
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nYoungAge vs Rest(OldAge, MiddleAge, SeniorAge )")
plt.legend()
plt.show()

'''
best_model = gb
best_model.fit(x_train, y_train)
rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test))
#rf_roc_auc = roc_auc_score(y_test,best_model.predict(x_test),average='samples', max_fpr=1, multi_class='ovr')
fpr,tpr,thresholds = roc_curve(y_test, best_model.predict_proba(x_test)[:,1])

plt.figure()
plt.plot(fpr,tpr,label = 'Random Forest(area = %0.2f)'% rf_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('LOC_ROC')
plt.show()
'''