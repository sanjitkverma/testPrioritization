import prioritization as pr

alphaRangeNum = 5
projects = ['Chart', 'Closure', 'Lang', 'Math', 'Time']
fromVersion = [1, 1, 1, 1, 1]
toVersion = [13, 50, 33, 50, 14]

#projects = ['Chart']
#fromVersion = [1]
#toVersion = [13]

sum_additional_elapsed_time = 0
sum_total_elapsed_time = 0

file = open("../../WTP-data/prioritization_exectime.txt","a")

for index, project in enumerate(projects):
	print('*** Project: %s ***' % project)

	versions = 0
	for versionNumber in range(fromVersion[index], toVersion[index]+1):
		print("* Version %d" % versionNumber)
		additional_elapsed_time, total_elapsed_time = pr.runPrioritization(project, versionNumber, alphaRangeNum)
		
		if (additional_elapsed_time == -1):
			continue
		else:
			versions = versions+1

		print("additional_elapsed_time: ", additional_elapsed_time)
		print("total_elapsed_time: ", total_elapsed_time)
		sum_additional_elapsed_time = sum_additional_elapsed_time + additional_elapsed_time
		sum_total_elapsed_time = sum_total_elapsed_time + total_elapsed_time

	mean_additional_elapsed_time = sum_additional_elapsed_time/versions
	mean_total_elapsed_time = sum_total_elapsed_time/versions

	file.write("%s,%f,%f\n" % (project, mean_additional_elapsed_time,mean_total_elapsed_time))
file.close()
