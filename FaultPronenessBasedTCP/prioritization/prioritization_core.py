'''
This module implements various prioritization techniques for test cases in software testing.
It includes methods for evaluating ranks, calculating APFD (Average Percentage of Faults Detected),
and applying different prioritization strategies based on coverage and probabilities.
'''

import numpy as np

def rankEvaluation(ranks, failedIds):
  '''
    Evaluates the ranking of test cases using APFD.

    Args:
        ranks (numpy array): Array of test case ranks.
        failedIds (numpy array): Array of IDs of failed test cases.

    Returns:
        float: The APFD score.
    '''
  return evaluateApfd(ranks, failedIds)

def evaluateApfd(ranks, failedIds):
  '''
    Computes the APFD score for test case prioritization.

    Args:
        ranks (numpy array): Array of test case ranks.
        failedIds (numpy array): Array of IDs of failed test cases.

    Returns:
        float: The computed APFD score.
    '''
  sum = 0
  for f in failedIds:
    ind = np.where(ranks==f)
    assert(np.size(ind) == 1)
    sum += (ind[0]+1)

  m = np.size(failedIds)
  n = np.size(ranks)
  APFD = 100*(1 - sum/(m*n) + 1/(2*n))
  return APFD

def rankEvaluationFirstFail(ranks, failedIds):
  '''
    Finds the normalized rank of the first failed test case.

    Args:
        ranks (numpy array): Array of test case ranks.
        failedIds (numpy array): Array of IDs of failed test cases.

    Returns:
        float: Normalized rank of the first failed test case.
    '''
  minVal = -1
  for f in failedIds:
    ind = np.where(ranks==f)
    assert(np.size(ind) == 1)
    if minVal == -1:
      minVal = ind[0]+1
    else:
      minVal = min(minVal, ind[0]+1)

  assert(minVal != -1)
  n = np.size(ranks)
  return minVal/float(n)

def totalPrioritization(coverage, unitProb):
  '''
    Performs total prioritization of test cases based on weighted coverage.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  totalWeightedCoverage = np.matmul(coverage, unitProb)
  return np.flip(np.argsort(totalWeightedCoverage))

def additionalPrioritization(coverage, unitProb):
  '''
    Performs additional prioritization by selecting test cases that maximize incremental coverage.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  testNum = coverage.shape[0]
  unitNum = coverage.shape[1]
  
  additionalSumRanks = np.zeros((testNum, ))
  testUsed = [False] * testNum # none of the tests are used at the beginning
  eps = 1.0e-7  
  
  totalWeightedCoverage = np.matmul(coverage,unitProb)
  additionalWeightedCoverage = np.array(totalWeightedCoverage)
#  print("additionalWeightedCoverage: ", additionalWeightedCoverage, "\n")

  # additionalWeightedCoverage = np.zeros((testNum, )) # the weighted coverage of each of the test cases
  # for u in range(0,unitNum):
  #  additionalWeightedCoverage = unitProb[u]*coverage[:,u]
 
  equal = 0
  unequal = 0

  for rank in range(0,testNum):
    #    print(sprintf("additional (%d/%d)", rank, testNum))
    bestAdditionalWeightedCoverage = -1
    bestTotalWeightedCoverage = -1
    
    for candTest in range(0,testNum):
      if (testUsed[candTest]):
        continue
      
      if (additionalWeightedCoverage[candTest] > bestAdditionalWeightedCoverage+eps or 
          (abs(additionalWeightedCoverage[candTest] - bestAdditionalWeightedCoverage) <= eps 
           and totalWeightedCoverage[candTest] > bestTotalWeightedCoverage)):
        bestTest = candTest
        bestAdditionalWeightedCoverage = additionalWeightedCoverage[candTest]
        bestTotalWeightedCoverage = totalWeightedCoverage[candTest]

      if (abs(additionalWeightedCoverage[candTest] - bestAdditionalWeightedCoverage) <= eps 
           and abs(totalWeightedCoverage[candTest] - bestTotalWeightedCoverage) <= eps):
        equal = equal+1
      else:
        unequal = unequal+1
 
#    print("bestTest: ", bestTest)
#    print("bestAdditionalWeightedCoverage: ", bestAdditionalWeightedCoverage)
#    print("bestTotalWeightedCoverage: ", bestTotalWeightedCoverage)
    additionalSumRanks[rank] = bestTest
    testUsed[bestTest] = True

    newUnitProb = np.maximum(unitProb - coverage[bestTest,:], 0)

#   another way of zeroing the values
#   unitProb[additionalWeightedCoverage>0]=0
    if np.sum(unitProb-newUnitProb) > eps: # ignore changing additionalWeightedCoverage if unit probs have not changed
      additionalWeightedCoverage -= np.matmul(coverage,(unitProb-newUnitProb))
    
    unitProb = newUnitProb

  file = open("../../WTP-data/log.txt","a")
  file.write("%d,%d\n" % (equal, unequal))
  file.close()

#    print("new UnitProb: ", newUnitProb, "\n")
  return additionalSumRanks

def maxNormalizedPrioritization(coverage, unitProb):
  '''
    Performs maximum normalized prioritization based on the highest normalized probability of coverage.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  testNum = coverage.shape[0]
  unitNum = coverage.shape[1]
  
  maxProbCovered = np.zeros((testNum,))
  unitCoverage = np.zeros((unitNum,))

  for u in range(unitNum):
    unitCoverage[u] = np.sum(coverage[:, u])
  
  unitProbNormalized = unitProb*1/(unitCoverage+1)
  
  for t in range(testNum):
    maxProbCovered[t] = np.max(coverage[t, :]*unitProbNormalized)
  
  sortedByMaxIndexes = np.flip(np.argsort(maxProbCovered))
  return sortedByMaxIndexes

def maxPrioritization(coverage, unitProb):
  '''
    Performs maximum prioritization based on the highest probability of coverage.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  testNum = coverage.shape[0]
  unitNum = coverage.shape[1]
  
  maxProbCovered = np.zeros((testNum,))

  for t in range(testNum):
    maxProbCovered[t] = np.max(coverage[t, :]*unitProb)
  
  sortedByMaxIndexes = np.flip(np.argsort(maxProbCovered))
  return sortedByMaxIndexes

def normalizedTotalPrioritization(coverage, unitProb):
  '''
    Performs total prioritization with normalized probabilities.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  unitNum = coverage.shape[1]
  
  unitCoverage = np.zeros((unitNum,))

  for u in range(unitNum):
    unitCoverage[u] = np.sum(coverage[:, u])
  
  unitProbNormalized = unitProb*1/(unitCoverage+1)
  
  return totalPrioritization(coverage, unitProbNormalized)

def normalizedAdditionalPrioritization(coverage, unitProb):
  '''
    Performs additional prioritization with normalized probabilities.

    Args:
        coverage (numpy array): Test case coverage matrix.
        unitProb (numpy array): Probability of each unit being buggy.

    Returns:
        numpy array: Ranked indices of test cases.
    '''
  unitNum = coverage.shape[1]
  
  unitCoverage = np.zeros((unitNum,))

  for u in range(unitNum):
    unitCoverage[u] = np.sum(coverage[:, u])
  
  unitProbNormalized = unitProb*1/(unitCoverage+1)
  
  return additionalPrioritization(coverage, unitProbNormalized)
