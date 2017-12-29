from __future__ import division
import pandas as pd
import math
import sys
import collections
import copy

#Method to count 0's and 1's in a column
def findCountOfNumbers(colArray):
	count0, count1 = 0, 0
	for value in colArray:
		if value == 1:
			count1 += 1
		else:
			count0 += 1
	return count0, count1			

#Helper method to find Entropy
def determineEntropy(param0, param1):
	if param0 == 0 or param1 == 0:
		return 0
	return -(param1 * math.log(param1, 2)) - (param0 * math.log(param0, 2))

#Method to find Entropy
def findOverallEntropy(colArray):
	count0, count1 = findCountOfNumbers(colArray)
	decisonArrayLength = len(colArray)
	param0 = count0/decisonArrayLength
	param1 = count1/decisonArrayLength
	return determineEntropy(param0, param1) 

#Method to find Variance Impurity
def findVarianceImpurity(colArray):	
	count0, count1 = findCountOfNumbers(colArray)
	colArrayLength = len(colArray)
	if count0 == 0 or count1 == 0:
		return 0		
	param0 = count0/colArrayLength
	param1 = count1/colArrayLength
	return param0*param1

#Method to count row entry matches of 2 columns
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

#Method to find Information Gain
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

#Method to find Variance Impurity Gain
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

#Method to choose attribute with best Information Gain
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

#Method to choose attribute with best Variance Impurity
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
			varianceImpurityGain = findVarianceImpurityGain(data, col, decisionArray, overallVarianceImpurity)	
			featureDictionary[col] = varianceImpurityGain
	return max(featureDictionary, key = featureDictionary.get)	

#Method to check divide by zero exception
def divideByZeroCheck(input1, input2):
	return input1/input2 if input2 != 0 else 0

#Method to return dataset of an attribute with a specific value
def getDataset(data, attribute, value):
	return data.loc[data[attribute] == value]

#Method to build tree with Information Gain
def generateTreeInformationGain(data):
	#Base condition
	if len(data.columns) == 1: #Column 'Class' only exists
		dataset0 = getDataset(data, 'Class', 0)
		dataset1 = getDataset(data, 'Class', 1)
		leaf = len(dataset0) if len(dataset0) > len(dataset1) else len(dataset1)
		return node(0, leaf, None, None)
	if len(getDataset(data, 'Class', 1)) == 0: #Entire datset with only 1, in sub tree
		return node(0, 0, None, None)
	if len(getDataset(data, 'Class', 0)) == 0: #Entire datset with only 0, in sub tree
		return node(0, 1, None, None)		

	attribute = findBestFeatureWithInformationGain(data)
	dataset0 = getDataset(data, attribute, 0)
	dataset1 = getDataset(data, attribute, 1)
	del data[attribute]
	del dataset1[attribute]
	del dataset0[attribute]
	return node(0, attribute, generateTreeInformationGain(dataset0), generateTreeInformationGain(dataset1))

#Method to build tree with Variance Impurity
def generateTreeVarianceImpurity(data):	
	#Base condition
	if len(data.columns) == 1: #Column 'Class' only exists
		dataset0 = getDataset(data, 'Class', 0)
		dataset1 = getDataset(data, 'Class', 1)
		leaf = len(dataset0) if len(dataset0) > len(dataset1) else len(dataset1)
		return node(0, leaf, None, None)
	if len(getDataset(data, 'Class', 1)) == 0: #Entire datset with only 1, in sub tree
		return node(0, 0, None, None)
	if len(getDataset(data, 'Class', 0)) == 0: #Entire datset with only 0, in sub tree
		return node(0, 1, None, None)		

	attribute = findBestFeatureWithVarianceImpurity(data)
	dataset0 = getDataset(data, attribute, 0)
	dataset1 = getDataset(data, attribute, 1)
	del data[attribute]
	del dataset1[attribute]
	del dataset0[attribute]
	return node(0, attribute, generateTreeVarianceImpurity(dataset0), generateTreeVarianceImpurity(dataset1))

#Method to print tree
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

#Method to find decision of a node
def applyModel(record, node):
	if type(node.name) == int:
		return node.name
	if record[node.name] == 0:
		return applyModel(record, node.left)
	if record[node.name] == 1:
		return applyModel(record, node.right)

#Method to find total decision matches with Class label
def testModel(testData, root):
	matchingDecisionCount = 0
	for index, record in testData.iterrows():
		decision = applyModel(record, root)
		if decision == record['Class']:
			matchingDecisionCount += 1 
	return matchingDecisionCount

#Method to find accuracy
def findAccuracy(value, total):
	return (value/total)

#Method to create queue
def createQueue():
	return collections.deque()

#Method to index all the nodes in the tree except leaf nodex
def indexNodesInTree(node, dequeue, countOfNodes):
	if node == None:
		return
	if type(node.name) != int:
		countOfNodes += 1
		node.index = countOfNodes
	if node.left != None or node.right != None:
		dequeue.append(node.left)
		dequeue.append(node.right)
	if dequeue:
		indexNodesInTree(dequeue.popleft(), dequeue, countOfNodes)

#Method to find count of non-leaf nodes
def getNonLeafNodesCount(node, dequeue):
	global nonLeafNodeCount
	if node == None:
		return
	if type(node.name) != int:
		nonLeafNodeCount += 1
	if node.left != None or node.right != None:
		dequeue.append(node.left)
		dequeue.append(node.right)
	if dequeue:
		getNonLeafNodesCount(dequeue.popleft(), dequeue)

#Method to find position of node in a tree
def findNodeinTree(node, destinationNode):
	if node == None:
		return 
	if node.index == destinationNode:
		return node
	if node.left != None:
		findNodeinTree(node.left, destinationNode)
	if node.right != None:
		findNodeinTree(node.right, destinationNode)		 	

#Method to count occurences of 0's and 1's for a given node
def countDecisions(node):
	global countDecisions0, countDecisions1
	if node == None:
		return
	if type(node.name) == int:
		if node.name == 0:
			countDecisions0 += 1  
		else:
			countDecisions1	+= 1
	if node.left != None:
		countDecisions(node.left)
	if node.right != None:
		countDecisions(node.right)	

#Method to find max occurence of decisions to replace a node
def findCommonDecisionOfNode(node, destinationNode):
	global countDecisions0, countDecisions1
	countDecisions0 = 0
	countDecisions1 = 0
	currentNode = findNodeinTree(node, destinationNode)
	countDecisions(node)
	return 0 if countDecisions0 >= countDecisions1 else 1

#Method to replace a node(sub-tree) with 0 or 1
def replaceNode(node, source, count):
	if node == None:
		return
	if node.index == source:
		node.name = count
		node.left = None
		node.right = None
		node.index = 0
		return
	if node.left != None:
		replaceNode(node.left, source, count)
	if node.right != None:
		replaceNode(node.right, source, count)

#Method to prune a tree
def pruneTree(tree, accuracy, data):
	global nonLeafNodeCount
	for i in range(25):		
		tempTree = copy.deepcopy(tree)
		queue = createQueue()
		indexNodesInTree(tempTree, queue, 0)
		nonLeafNodeCount = 0
		queue = createQueue()
		getNonLeafNodesCount(tempTree, queue)
		randomNode = nonLeafNodeCount
		maxDecisionCount = findCommonDecisionOfNode(tempTree, randomNode)
		replaceNode(tempTree, randomNode, maxDecisionCount)
		nonLeafNodeCount -= 1
		pruneCount = testModel(data, tempTree)
		tempAccuracy = findAccuracy(pruneCount, len(data))
		if tempAccuracy > accuracy:
			tree = copy.deepcopy(tempTree)
			accuracy = tempAccuracy		
	return tree

class node:
	def __init__(self, index, name, left, right):
		self.index = index
		self.name = name
		self.left = left
		self.right = right	
			
def main():
	#Getting inputs from user
	training_set = sys.argv[1]
	validation_set = sys.argv[2]
	test_set = sys.argv[3]
	to_print = sys.argv[4]
	prune = sys.argv[5]

	#Reading Input csv files
	informationGainData = pd.read_csv(training_set)
	varianceImpurityData = pd.read_csv(training_set)
	trainData = pd.read_csv(training_set)
	testData = pd.read_csv(test_set)
	validationData = pd.read_csv(validation_set)

	#Building tree with training data
	informationGainRoot = generateTreeInformationGain(informationGainData)
	varianceImpurityRoot = generateTreeVarianceImpurity(varianceImpurityData)
	
	#Indexing nodes in the tree
	queue = createQueue()
	indexNodesInTree(informationGainRoot, queue, 0)
	queue = createQueue()
	indexNodesInTree(varianceImpurityRoot, queue, 0)

	#printing tree
	if to_print == 'yes':
		print '\nInformation Gain Heuristic Tree'
		printTree(informationGainRoot, 0)
		print '\nVariance Impurity Heuristic Tree'
		printTree(varianceImpurityRoot, 0)
	
	#Calculating accuracies on input files
	print '\nAccuracy on Training Set:'
	print 'Information Gain Heuristic = %.2f'% findAccuracy(testModel(trainData, informationGainRoot),
		len(trainData))
	print 'Variance Impurity Heuristic = %.2f'% findAccuracy(testModel(trainData, varianceImpurityRoot),
		len(trainData))

	print '\nAccuracy on Validation Set:'
	print 'Information Gain Heuristic = %.2f'% findAccuracy(testModel(validationData, informationGainRoot),
		len(validationData))
	print 'Variance Impurity Heuristic = %.2f'% findAccuracy(testModel(validationData, varianceImpurityRoot),
		len(validationData))

	informationGainCount = testModel(testData, informationGainRoot)
	informationGainAccuracy = findAccuracy(informationGainCount,len(testData))
	print '\nAccuracy on Test Set:'
	print 'Information Gain Heuristic = %.2f'% informationGainAccuracy
	varianceImpurityCount = testModel(testData, varianceImpurityRoot)
	varianceImpurityAccuracy = findAccuracy(varianceImpurityCount,len(testData))
	print 'Variance Impurity Heuristic = %.2f'% varianceImpurityAccuracy

	#Pruning tree
	if prune == 'yes':
		prunnedInformationGainRoot = pruneTree(informationGainRoot, informationGainAccuracy, validationData)	
		prunnedVarianceImpurityRoot = pruneTree(varianceImpurityRoot, informationGainAccuracy, validationData)	

		#Calculating accuracies on pruned tree
		print '\nAccuracy on Training Set after pruning:'
		print 'Information Gain Heuristic = %.2f'% 	findAccuracy(testModel(trainData, informationGainRoot),
			len(trainData))
		print 'Variance Impurity Heuristic = %.2f'% findAccuracy(testModel(trainData, varianceImpurityRoot),
			len(trainData))

		print '\nAccuracy on Validation Set after pruning:'
		print 'Information Gain Heuristic = %.2f'% 	findAccuracy(testModel(validationData, informationGainRoot),
			len(validationData))
		print 'Variance Impurity Heuristic = %.2f'% findAccuracy(testModel(validationData, varianceImpurityRoot),
			len(validationData))

		prunnedInformationGainCount = testModel(testData, prunnedInformationGainRoot)
		prunnedInformationGainAccuracy = findAccuracy(prunnedInformationGainCount,len(testData))

		print '\nAccuracy on Test Set after pruning:'
		print 'Information Gain Heuristic = %.2f'% prunnedInformationGainAccuracy

		prunnedVarianceImpurityCount = testModel(testData, prunnedVarianceImpurityRoot)
		prunnedVarianceImpurityAccuracy = findAccuracy(prunnedVarianceImpurityCount,len(testData))
		print 'Variance Impurity Heuristic = %.2f'% prunnedVarianceImpurityAccuracy

		if to_print == 'yes':
			print '\nInformation Gain Heuristic Tree after pruning'
			printTree(prunnedInformationGainRoot, 0)
			print '\nVariance Impurity Heuristic Tree after pruning'
			printTree(prunnedVarianceImpurityRoot, 0)

if __name__ == '__main__':
	main()