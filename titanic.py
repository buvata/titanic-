import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

from sklearn import linear_model,neighbors

data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')

full_data=[data,test_data]

print(data.head())
print(test_data.head())
print("**************************")
print(data.columns.values)
print(data.info())
print("****************************")
print(data.describe())
print("******************************")

for dataset in full_data:
	dataset['Age'].fillna(dataset['Age'].mean(),inplace=True)
	dataset['Fare'].fillna(dataset['Fare'].mean(),inplace=True)
	dataset['Age']=dataset['Age'].astype(int)
print(data.describe())

print("**************Sex***************")

survived_sex=data[data['Survived']==1]['Sex'].value_counts()
dead_sex=data[data['Survived']==0]['Sex'].value_counts()
print(data[['Sex','Survived']].groupby(['Sex']).mean())

df=pd.DataFrame([survived_sex,dead_sex])
df.index=['Survived','Dead']
df.plot(kind='bar',stacked=True)
plt.show()

print("****************Age**************")
print(data['Age'].describe())
data['CategoricalAge']=pd.cut(data['Age'],5)
print(data[['CategoricalAge','Survived']].groupby(['CategoricalAge']).mean())


plt.hist([data[data['Survived']==1]['Age'],data[data['Survived']==0]['Age']], stacked=True, color = ['g','r'],
         bins = 5,label = ['Survived','Dead'])
plt.xlabel('Age')
plt.ylabel('Number of passengers')
plt.legend()
plt.show()

print("****************Family***************")

# family
for dataset in full_data:
	dataset['FamilySize']=dataset['SibSp']+dataset['Parch']+1

print(data[['FamilySize','Survived']].groupby(['FamilySize']).mean())

for dataset in full_data:
	dataset['IsAlone']=0
	dataset.loc[dataset['FamilySize']==1,'IsAlone']=1

print(data[['IsAlone','Survived']].groupby(['IsAlone']).mean())


print("****************Fare**************")

data['CategoricalFare']=pd.cut(data['Fare'],4)
print(data[['CategoricalFare','Survived']].groupby(['CategoricalFare']).mean())

print("***************Embarked***********************")

for dataset in full_data:
	dataset['Embarked']=dataset['Embarked'].fillna('S')
print(data[['Embarked','Survived']].groupby(['Embarked']).mean())

# ("********************Transform data******************")

for dataset in full_data:
	dataset['Sex']=dataset['Sex'].map({'female':0,'male':1}).astype(int)

	dataset['Embarked']=dataset['Embarked'].map({'S':0,'C':1,'Q':2}).astype(int)

	dataset.loc[dataset['Fare']<128.1,'Fare']=0
	dataset.loc[(dataset['Fare']>128.1)&(dataset['Fare']<256.17),'Fare']=1
	dataset.loc[(dataset['Fare']>256.17)&(dataset['Fare']<384.25),'Fare']=2
	dataset.loc[dataset['Fare']>384.25,'Fare']=3
	dataset['Fare']=dataset['Fare'].astype(int)

	dataset.loc[dataset['Age']<=16,'Age']=0
	dataset.loc[(dataset['Age']>16)&(dataset['Age']<=32),'Age']=1
	dataset.loc[(dataset['Age']>32)&(dataset['Age']<=48),'Age']=2
	dataset.loc[(dataset['Age']>48)&(dataset['Age']<=64),'Age']=3
	dataset.loc[dataset['Age']>64,'Age']=4
	dataset['Age']=dataset['Age'].astype(int)

# remove unnacessary feature
drop_feature=['PassengerId','Name','Ticket','Cabin','SibSp','Parch','FamilySize']
data_drop=data.drop(drop_feature,axis=1)
data_drop=data_drop.drop(['CategoricalAge','CategoricalFare'],axis=1)

test_data_drop=test_data.drop(drop_feature,axis=1)
trainingSet=data_drop.values
testingSet=test_data_drop.values

print(data_drop.head(10))
print(trainingSet)
print(test_data.columns.values)


X = trainingSet[0::, 1:]
y = trainingSet[0::, 0]

clf=neighbors.KNeighborsClassifier(n_neighbors=50,p=2,weights='distance')
clf.fit(X,y)
y_pred=clf.predict(testingSet)

submission = pd.DataFrame({
     "PassengerId": test_data["PassengerId"],
     "Survived": y_pred
 })
submission
submission.to_csv('submission.csv', index=False)

result=pd.read_csv('submission.csv')
print(result.head())
















