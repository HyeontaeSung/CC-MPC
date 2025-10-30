import os
import re
import subprocess
import dataForCost_ref
# import dataForCost_v9
import matplotlib.pyplot as plt
import numpy as np


# Define a pattern to extract agent number and frame number
pattern = re.compile(r"agent(\d+)_frame(\d+)_cov")

# Get a list of all files in the current directory
all_files = os.listdir("out/data")

# Create a dictionary to store the smallest frame number for each agent
agent_frame_dict = {}

for file in all_files:
    match = pattern.match(file)
    if match:
        agent_num = int(match.group(1))
        frame_num = int(match.group(2))

        # If agent is already in dictionary, update with smaller frame number
        if agent_num in agent_frame_dict:
            agent_frame_dict[agent_num] = min(agent_frame_dict[agent_num], frame_num)
        else:
            agent_frame_dict[agent_num] = frame_num

# create a dictionary to store the largest frame number for  each agent
agent_frame_dict2 = {}
for file in all_files:
    match = pattern.match(file)
    if match:
        agent_num2 = int(match.group(1))
        frame_num2 = int(match.group(2))

        # If agent is already in dictionary, update with smaller frame number
        if agent_num2 in agent_frame_dict2:
            agent_frame_dict2[agent_num2] = max(agent_frame_dict2[agent_num2], frame_num2)
        else:
            agent_frame_dict2[agent_num2] = frame_num2

processTime = []
solveTime = []
solveTimeOV = []
worstTime = []
cost = []
costU = []
i = 0
Ustar = [] # X[1|0], X[2|1], ... , X[planning horizon-1|planning horizon-2]
Xstar = [] # U[0|0], U[1|1], .. , U[planning horizon-2|planninghorizon-2]

for agent, frame in sorted(agent_frame_dict.items()):
    print(f"Agent {agent}: Smallest frame number is {frame}")
    numFrames = int((agent_frame_dict2[agent]-frame)/10 + 1)
    # print(numFrames)
    # print(numFrames)
    # Call data.py with agent and frame as parameters
    # subprocess.run(["python", "data.py", str(agent), str(frame)]) # you don't need it if you are not interested in plots
    #try:
    episode = dataForCost_ref.getData_singleEpisode(agent, frame, numFrames) # how can we get the numFrames ?
    process_time, solve_time, solveTime_OV, worsttime, X_star, U_star, goal, x_init, timeout = episode.get_data() # a sample code.
    # print(timeout)
    if timeout[-1] == False:# and Infeasible[-1] == False: #and len(timeout)>=14: # save the result of no timeout and no infeasible
        i = i +1
        # print(i)
        cost_temp, costU_temp = episode.cost_final(X_star, U_star)
        cost.append(cost_temp)
        costU.append(costU_temp)
        processTime.append(process_time)
        solveTime.append(solve_time)
        solveTimeOV.append(solveTime_OV)
        worstTime.append(worsttime)
        Xstar.append(X_star)
        Ustar.append(U_star)

        # if 10 <= numFrames:
        #    # we can get an error if startOVconstraint > plotFrom
        #    episode.plot(4,10) # plotFrom, plotUntil
    #except:
    #    continue


# print("<<44 runs of [conventional-scheme_star-intersection_ov1_[-43,-38]]>>")
print("Average cost:" , sum(cost)/len(cost), len(cost))
print("Average costU:" , sum(costU)/len(costU))
print("Average processTime from time.time():" , sum(processTime)/len(processTime))
print("Feasibility:" , len(solveTime)/50)# num episode/
print("Average solveTime from cvxpy:" , sum(solveTime)/len(solveTime),len(solveTime) )
print("Average solveTimeOV from cvxpy:" , sum(solveTimeOV)/len(solveTimeOV))
print("Average worstTime from cvxpy:", sum(worstTime)/len(worstTime))
