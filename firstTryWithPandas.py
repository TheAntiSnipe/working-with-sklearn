import numpy as np
import pandas as pd
from sklearn import tree

def giveClass(i): #A function for assisting in output display
	if(i==1):
		return "First Class"
	if(i==2):
		return "Second Class"
	if(i==3):
		return "Third Class"


def giveAgeGroup(i): #A function for assisting in output display
	if(i==0):
		return "0-9"
	if(i==1):
		return "10-29"
	if(i==2):
		return "30-49"
	if(i==3):
		return "50-80"


data = pd.read_csv('titanic.csv',usecols = ['Pclass','Sex','Age']) #Accessing data from the target .csv file 
#data.replace(r'^\s*$', np.nan, regex = True)
data = data.fillna(data['Age'].mean()) #We don't want ages throwing off the classifier, hence the mean replacement
data = data.replace(r'^male$', 0, regex = True) #Tokenization of the gender field
data = data.replace(r'^female$', 1, regex = True) 
ageArray = data['Age']
tokenizedAge = [0 for i in range(len(ageArray))]

j=0
for i in ageArray: #Tokenization of the age field
	if(i<10):
		tokenizedAge[j] = 0
	elif(i<30):
		tokenizedAge[j] = 1
	elif(i<50):
		tokenizedAge[j] = 2
	elif(i<=80):
		tokenizedAge[j] = 3
	j+=1
data['Age']=tokenizedAge

X=data
y=pd.read_csv('titanic.csv',usecols = ['Survived'])
clf = tree.DecisionTreeRegressor()
clf = clf.fit(X,y) #Actual fitting

print('''\n\n\nThe table below illustrates the statistical analysis of how probable an individual's 
survival would be under certain conditions. Note that the probabilities observed
are the result of running a decision tree algorithm on the dataset, and that the 
decision tree can also be used to predict whether an individual survived the 
sinking of the Titanic.\n\n\n''')
print("||Class\t\t\t|Gender\t\t|Age Group\t|Survival Probability\t||")
for i in range(1,4): #Gaining an insight as to how various factors came into play in terms of survival odds

	for j in range(0,2):

		for k in range(0,4):

			psgClass = giveClass(i)
			gender = "Male" if j==0 else "Female"
			ageGroup = giveAgeGroup(k)
			probabilityOfSurvival = round(clf.predict([[i,j,k]])[0]*100,3)
			print("||",psgClass," \t|",gender," \t|",ageGroup,"  \t|",probabilityOfSurvival,"  \t\t||")