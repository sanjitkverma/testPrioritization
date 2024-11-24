'''
This script performs bug prediction evaluation for our software libraries.
It evaluates bug prediction results against real bugs for specific versions of each project.
The results are written to evaluation files and summarized across versions for each project.
'''

import numpy as np
import pandas as pd
import h5py

def findRowIndex(data, value):
	'''
    Finds the index of a specific value in a dataset.

    Args:
        data (numpy array): The array to search.
        value (str): The value to find.

    Returns:
        int: Index of the value if found, otherwise -1.
    '''

	for i in range(0, data.shape[0]):
		print(str(data[i]),"<>",value)
		if (str(data[i]) == value):
			return i
	return -1

def runBugpredictionEvaluation(project, versionNumber):
	'''
    Runs bug prediction evaluation for a specific project version.

    Reads bug prediction results and compares them against actual bug-fix data.
    Calculates evaluation metrics based on the accuracy of predictions.

    Args:
        project (str): The name of the project.
        versionNumber (int): The version number to evaluate.

    Returns:
        tuple: Evaluation score and probability score.
    '''
	dataPath = "../../WTP-data/%s/%d" % (project, versionNumber)

	h5FileAddress = '%s/TestCoverage.h5' % dataPath

	h5 = h5py.File(h5FileAddress)

	dataSize=h5["data"].shape[0]
	testNames = h5["columnTestNamesArray"]
	unitNames = h5["rowMethodNamesArray"]
	unitNum = unitNames.shape[0]
	testNum = testNames.shape[0]

	print("unitNum: ", unitNum)
	print("testNum: ", testNum)


	print("Reading bug prediction...")
#	bugpred = pd.read_csv('%s/nn_bugprediction.csv' % dataPath, delimiter=',')
	bugpred = pd.read_csv('%s/bugpred.csv' % dataPath, delimiter=',')
	
	print("Reading real bugs...")
	faultClassFile = ("%s/bugfix_sources.txt" % dataPath)
	with open(faultClassFile) as f:
	    faultClasses = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	faultClasses = [x.strip() for x in faultClasses] 

	evaluation = 0
	evaluationProb = 0
	found = 0

	for faultClass in faultClasses:
		row = bugpred[bugpred.LongName==faultClass]
		if (row.empty):
			print("%s not found in bug prediction results" % faultClass)
		else:
			found = found + 1
			print("row.bugpred: ",row["bugpred"].values[0])
			prob = row["bugpred"].values[0]
			print("%s --> %f" % (faultClass, prob))
			if (prob >= 0.1):
				evaluation = evaluation + 1
			evaluationProb = evaluationProb + prob

	if (found==0):
		print("No classes found in bug prediction results")
		return	

	evaluation = evaluation / found
	evaluationProb = evaluationProb / found
	print("evaluation: ", evaluation)
	return (evaluation,evaluationProb)

projects = ['Chart', 'Closure', 'Math', 'Lang', 'Time']
fromVersion = [1, 1, 1, 1, 1]
toVersion = [13, 50, 50, 33, 14]

evaluationSumProjects = []
for index, project in enumerate(projects):
	evaluationSum = 0
	f = open('../../WTP-data/%s/bugprediction_evaluation.csv' % project, "w+")
	f.write("version,evaluation,evaluation_prob\n")	
	for versionNumber in range(fromVersion[index], toVersion[index]+1):
		print("* Version %d" % versionNumber)
		(evaluation,evaluationProb) = runBugpredictionEvaluation(project, versionNumber)
		evaluationSum = evaluationSum+evaluation
		f.write("%d,%f,%f\n"%(versionNumber,evaluation,evaluationProb))
		print()
	f.close()
	evaluationSumProjects.append(evaluationSum)

for index, project in enumerate(projects):
	print("Evaluation sum %s: %d/%d" % (projects[index], evaluationSumProjects[index], toVersion[index]-fromVersion[index]+1))
	index=index+1