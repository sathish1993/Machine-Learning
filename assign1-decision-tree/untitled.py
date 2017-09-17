from __future__ import division
import pandas as pd
import math
import sys

def findCountOfNumbers(colArray):
	count0, count1 = 0, 0
	for value in colArray:
		if value == 1:
			count1 += 1
		else:
			count0 += 1
	return count0, count1			

def determineEntropy(param0, param1):
	if param0 == 0 or param1 == 0:
		return 0
	return -(param1 * math.log(param1, 2)) - (param0 * math.log(param0, 2))

def findOverallEntropy(colArray):
	count0, count1 = findCountOfNumbers(colArray)
	decisonArrayLength = len(colArray)
	param0 = count0/decisonArrayLength
	param1 = count1/decisonArrayLength
	return determineEntropy(param0, param1) 

def findVarianceImpurity(colArray):	
	count0, count1 = findCountOfNumbers(colArray)
	colArrayLength = len(colArray)
	if count0 == 0 or count1 == 0:
		return 0		
	param0 = count0/colArrayLength
	param1 = count1/colArrayLength
	return param0*param1

def findIntermediateCount(colArray, decisionArray):
	count0, count1 = 0, 0
	count00, count01 = 0, 0
	count10, count11 = 0, 0
	for x, y in zip(colArray, decisionArray):
		if x == 1:
			count1 += 1
			if y == 0:
				count10 += 1
			else:
				count11 += 1	
		else:
			count0 += 1	
			if y == 0:
				count00 += 1
			else:
				count01 += 1	
	return count0, count1, count00, count01, count10, count11

def findInformationGain(colArray, decisionArray, overallEntropy):
	count0, count1, count00, count01, count10, count11 = findIntermediateCount(colArray, decisionArray)

	colArrayLength = len(colArray)
	param0 = count0/colArrayLength
	param1 = count1/colArrayLength

	param00Entropy = divideByZeroCheck(count00, count0)
	param01Entropy = divideByZeroCheck(count01, count0)
	param10Entropy = divideByZeroCheck(count10, count1) 
	param11Entropy = divideByZeroCheck(count11, count1)  

	entropy0 = param0 * determineEntropy(param00Entropy, param01Entropy)
	entropy1 = param1 * determineEntropy(param10Entropy, param11Entropy)
	informationGain = overallEntropy - entropy0 - entropy1
	return informationGain	

def findVarianceImpurityGain(data, attribute, colArray, overallVarianceImpurity):	
	dataset0 = getDataset(data, attribute, 0)
	dataset1 = getDataset(data, attribute, 1)
	colArrayLength = len(colArray)
	param0 = len(dataset0)/colArrayLength
	param1 = len(dataset1)/colArrayLength
	varianceImpurity0 = param0 * findVarianceImpurity(dataset0['Class'].values)
	varianceImpurity1 = param1 * findVarianceImpurity(dataset1['Class'].values)
	varianceImpurityGain = overallVarianceImpurity - varianceImpurity0 - varianceImpurity1
	return varianceImpurityGain

def findBestFeatureWithInformationGain(data):	
	columns = data.columns
	decisionArray = data['Class'].values
	overallEntropy = findOverallEntropy(decisionArray)
	featureDictionary = {}
	for col in columns:
		if col == 'Class':
			break
		else:
			attributeArray = data[col].values
			informationGain = findInformationGain(attributeArray, decisionArray, overallEntropy)	
			featureDictionary[col] = informationGain
	return max(featureDictionary, key = featureDictionary.get)	

def findBestFeatureWithVarianceImpurity(data):
	columns = data.columns
	decisionArray = data['Class'].values
	overallVarianceImpurity = findVarianceImpurity(decisionArray)
	featureDictionary = {}
	for col in columns:
		if col == 'Class':
			break
		else:
			attributeArray = data[col].values
			varianceImpurityGain = findVarianceImpurityGain(data, col, attributeArray, overallVarianceImpurity)	
			featureDictionary[col] = varianceImpurityGain
	return max(featureDictionary, key = featureDictionary.get)	

def divideByZeroCheck(input1, input2):
	return input1/input2 if input2 != 0 else 0

def getDataset(data, attribute, value):
	return data.loc[data[attribute] == value]

def generateTreeInformationGain(data):
	columns = data.columns
	#Base condition
	if len(columns) == 1: #Column 'Class' only exists
		dataset0 = getDataset(data, 'Class', 0)
		dataset1 = getDataset(data, 'Class', 1)
		leaf = len(dataset0) if len(dataset0) > len(dataset1) else len(dataset1)
		return node(leaf, None, None)
	if len(getDataset(data, 'Class', 1)) == 0: #Entire datset with only 1, in sub tree
		return node(0, None, None)
	if len(getDataset(data, 'Class', 0)) == 0: #Entire datset with only 0, in sub tree
		return node(1, None, None)		

	attribute = findBestFeatureWithInformationGain(data)
	dataset0 = getDataset(data, attribute, 0)
	dataset1 = getDataset(data, attribute, 1)
	del data[attribute]
	return node(attribute, generateTreeInformationGain(dataset0), generateTreeInformationGain(dataset1))

def generateTreeVarianceImpurity(data):	
	columns = data.columns
	#Base condition
	if len(columns) == 1: #Column 'Class' only exists
		dataset0 = getDataset(data, 'Class', 0)
		dataset1 = getDataset(data, 'Class', 1)
		leaf = len(dataset0) if len(dataset0) > len(dataset1) else len(dataset1)
		return node(leaf, None, None)
	if len(getDataset(data, 'Class', 1)) == 0: #Entire datset with only 1, in sub tree
		return node(0, None, None)
	if len(getDataset(data, 'Class', 0)) == 0: #Entire datset with only 0, in sub tree
		return node(1, None, None)		

	attribute = findBestFeatureWithVarianceImpurity(data)
	print attribute
	dataset0 = getDataset(data, attribute, 0)
	dataset1 = getDataset(data, attribute, 1)
	del data[attribute]
	return node(attribute, generateTreeVarianceImpurity(dataset0), generateTreeVarianceImpurity(dataset1))

def printTree(node, depth):
	#preorder traversal, (root, left, right)
	if node == None:
		return
	if type(node.name) is int:
		print node.name
	if node.left != None:
		print '\n', ('|' * depth), node.name, '=', 0, ':',
		printTree(node.left, depth+1)
	if node.right != None:
		print ('|' * depth), node.name, '=', 1, ':',
		printTree(node.right, depth+1)

def applyModel(record, node):
	if type(node.name) == int:
		return node.name
	if record[node.name] == 0:
		return applyModel(record, node.left)
	if record[node.name] == 1:
		return applyModel(record, node.right)

def testModel(testData, model):
	matchingDecisionCount = 0
	for index, record in testData.iterrows():
		decision = applyModel(record, model)
		if decision == record['Class']:
			matchingDecisionCount += 1 
	return matchingDecisionCount

class node:
	def __init__(self, name, left, right):
		self.name = name
		self.left = left
		self.right = right	
			
def main():
	training_set = sys.argv[1]
	validation_set = sys.argv[2]
	test_set = sys.argv[3]
	to_print = sys.argv[4]
	prune = sys.argv[5]

	data = pd.read_csv(training_set)
	testData = pd.read_csv(test_set)
	informationGainRoot = generateTreeInformationGain(data)
	varianceImpurityRoot = generateTreeVarianceImpurity(data)

	
	if to_print == 'yes':
		print '\nInformation Gain Heuristic Tree'
		printTree(informationGainRoot, 0)
		print '\nVariance Impurity Heuristic Tree'
		printTree(varianceImpurityRoot, 0)
	
	informationGainCount = testModel(testData, informationGainRoot)
	print 'Accuracy on Test Set using Information Gain Heuristic Tree'
	informationGainAccuracy = (informationGainCount/len(testData))*100
	print informationGainAccuracy

	varianceImpurityCount = testModel(testData, varianceImpurityRoot)
	print 'Accuracy on Test Set using Variance Impurity Heuristic Tree'
	varianceImpurityAccuracy = (varianceImpurityCount/len(testData))*100
	print varianceImpurityAccuracy

if __name__ == '__main__':
	main()