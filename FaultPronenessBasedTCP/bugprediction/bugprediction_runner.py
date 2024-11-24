#!/usr/bin/env python
# coding: utf-8
'''
This script executes the bug prediction analysis for the specified software libraries using the 
`keras_bugprediction` module. It calculates the average elapsed time and prediction time 
for a range of software versions, writing the results to an output file.

The software libraries are specified in the `projects` list, and the version ranges are specified

NOTE FROM CSC 591 Group: It might be beneficial to run each library at a time, our computer's were unable 
to run all libraries at once.

'''
#get_ipython().run_line_magic('matplotlib', 'inline')
import keras_bugprediction
import time

projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
fromVersion = [1, 1, 1, 1, 1]
toVersion = [13, 50, 33, 50, 14]
lastVersions = [26, 133, 65, 106, 27]

# Can modify what you want to run like this

# projects = ['Lang', 'Math']
# fromVersion = [1, 1]
# toVersion = [33, 50]
# lastVersions = [65, 106]

# Loop through each project and execute the prediction analysis
file = open("../../WTP-data/bugprediction_exectime.txt","a")
for index, project in enumerate(projects):
	print('*** Project: %s ***' % project)
	lastVersion = lastVersions[index]
	start_time = time.time()
	sum_prediction_time = 0

	# Iterate through the specified versions for the current project
	for version in range(fromVersion[index], toVersion[index]+1):
		print('Version: %d' % version)
		prediction_time = keras_bugprediction.kerasBugPrediction(project, version, lastVersion)
		sum_prediction_time = sum_prediction_time + prediction_time

	# Calculate and log average execution times
	elapsed_time = time.time() - start_time
	print("elapsed_time: ", elapsed_time)
	mean_elapsed_time = elapsed_time / (toVersion[index]-fromVersion[index]+1)
	mean_sum_prediction_time = sum_prediction_time / (toVersion[index]-fromVersion[index]+1)
	file.write("%s,%f,%f\n" % (project, mean_elapsed_time, mean_sum_prediction_time))
file.close()
