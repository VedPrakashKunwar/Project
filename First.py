# import module
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import numpy as np
# read file
train = pd.read_csv('D:/Datasets/Kaggle/Titanic/train.csv')
test = pd.read_csv('D:/Datasets/Kaggle/Titanic/test.csv')

sex_pivot = train.pivot_table(index="Sex",values="Survived")
sex_pivot.plot.bar()
plt.show()

Pclass_pivot = train.pivot_table(index='Pclass', values='Survived')
Pclass_pivot.plot.bar()
plt.show()

survived = train[train["Survived"] == 1]
died = train[train["Survived"] == 0]
survived["Age"].plot.hist(alpha=0.5,color='red',bins=50)
died["Age"].plot.hist(alpha=0.5,color='blue',bins=50)
plt.legend(['Survived','Died'])
plt.show()


def process_age(df,cut_points,label_names):
    df["Age"] = df["Age"].fillna(-0.5)
    df["Age_categories"] = pd.cut(df["Age"],cut_points,labels=label_names)
    return df

cut_points = [-1,0,5, 12,18,35,60,100]
label_names = ["Missing","Infant","Child","Teenager","Young Adult","Adult","Senior"]

train = process_age(train,cut_points,label_names)
test = process_age(test,cut_points,label_names)

Age_Categories_pivot = train.pivot_table(index='Age_categories', values='Survived')
Age_Categories_pivot.plot.bar()
plt.show()

# important variable  till now -  age, sex, pclass
train['Pclass'].value_counts()
# Here Pclass is int64, but it show first, second, third class
# also sex and age_categories both are categorical variable.

def create_dummies(df,column_name):
    dummies = pd.get_dummies(df[column_name],prefix=column_name)
    df = pd.concat([df,dummies],axis=1)
    return df

train = create_dummies(train, 'Pclass')
train = create_dummies(train, 'Sex')
train = create_dummies(train, 'Age_categories')

test = create_dummies(test, 'Pclass')
test = create_dummies(test, 'Sex')
test = create_dummies(test, 'Age_categories')

# make the model
lr = LogisticRegression()
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

lr.fit(train[columns], train['Survived'])

# from now on we will refer to this
# dataframe as the holdout data
holdout = test 

# As test set don't have Survived (target) column we have to devide the train set to train and test set and will make original test set to holdout set.
columns = ['Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex_female', 'Sex_male',
       'Age_categories_Missing','Age_categories_Infant',
       'Age_categories_Child', 'Age_categories_Teenager',
       'Age_categories_Young Adult', 'Age_categories_Adult',
       'Age_categories_Senior']

train_X, test_X, train_y, test_y = train_test_split(train[columns], train['Survived'], test_size=0.2, random_state=0)

# creating the model
lr = LogisticRegression()
lr.fit(train_X, train_y)
predictions = lr.predict(test_X)

# finding the accuracy
# for titanic challenge, kaggle is evaluating model based on accuracy
# se we will also use accuracy from sklearn.metrics
accuracy = accuracy_score(test_y, predictions)

# as the test data is quite small so there is good chance that model is overfitting
# so overcome this situation we will K-Fold Cross Validaion

lr = LogisticRegression()
scores = cross_val_score(lr, train[columns], train['Survived'], cv=10)
accuracy = np.mean(scores)
print(scores)
print(accuracy)

# making prediction on holdout data  (kaggle test dataset)
lr = LogisticRegression()
lr.fit(train[columns], train['Survived'])
holdout_predictions = lr.predict(holdout[columns])

# making submission file
holdout_ids = holdout["PassengerId"]
submission_df = {"PassengerId": holdout_ids,
                 "Survived": holdout_predictions}
submission = pd.DataFrame(submission_df)
submission.to_csv('D:/Datasets/Kaggle/Titanic/submission_lr.csv', index=False)