'''
This script evaluates and compares test case prioritization strategies using
a coverage matrix and a predefined set of failed test IDs. It computes and prints
the APFD (Average Percentage of Faults Detected) scores for additional prioritization,
total prioritization, and max prioritization methods.

Input:
- coverage.csv: A file containing the test coverage matrix.

Output:
- Prints the prioritization order and APFD scores for each strategy.
'''
import numpy
import prioritization_core as pc

# Load the test coverage matrix from the CSV file.
# The matrix represents which units (columns) are covered by which tests (rows).
coverage = numpy.genfromtxt('coverage.csv', delimiter=';', dtype='float32')
print("coverage: \n", coverage)

# Get the number of tests and units from the dimensions of the coverage matrix.
testNum = coverage.shape[0]
unitNum = coverage.shape[1]

# Initialize unit probabilities (equal likelihood for all units being faulty).
unitProb = numpy.ones((unitNum, ))
print("unitProb: ", unitProb)
# Define the IDs of failed test cases for evaluation.
failedIds = numpy.array([3, 4])
print("failedIds: ", failedIds)

# Perform additional prioritization and calculate the APFD score.
additionalPrioritization = pc.additionalPrioritization(coverage, unitProb)
print("additionalPrioritization: ", additionalPrioritization, " APFD: ", pc.rankEvaluation(additionalPrioritization, failedIds))

# Perform total prioritization and calculate the APFD score.
totalPrioritization = pc.totalPrioritization(coverage, unitProb)
print("totalPrioritization: ", totalPrioritization, " APFD: ", pc.rankEvaluation(totalPrioritization, failedIds))

# Perform max prioritization and calculate the APFD score.
maxPrioritization = pc.maxPrioritization(coverage, unitProb)
print("maxPrioritization: ", maxPrioritization, " APFD: ", pc.rankEvaluation(maxPrioritization, failedIds))