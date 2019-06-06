from sklearn import svm
from sklearn import datasets
import pickle
classifier = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data[:-5], iris.target[:-5]
classifier.fit (X, y)

#for i in iris.data[-5:]:
j = len(iris.target)-5
score = 0
scores = 5
for i in classifier.predict(iris.data[-5:]):
	if(i==iris.target[j]):
		score+=1
	j+=1
mean = score/scores
accu = mean*100
print("Model accuracy is: ",accu,"%")
print("Pickling model: ")
pickledData = pickle.dumps(classifier)
print("Complete!")
classifier2 = pickle.loads(pickledData)
print("Testing model for entry:",iris.data[-1:])
print("Expected output:",iris.target[-1:])
print("Observed output:",classifier2.predict(iris.data[-1:]))
