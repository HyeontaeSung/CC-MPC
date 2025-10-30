import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import utility as util
import sys
import openpyxl



# Extract agent and frame_id from command line arguments

class getData_singleEpisode:
    def __init__(self, Agent, frame_id, numFrames):
        self.Agent = Agent
        self.frame_id = frame_id
        self.numFrames = numFrames
        self.T = 8 # control horizon
        self.path = ""

    def frame_list(self): # make the frame list
        frame = []
        for i in range(self.numFrames):
            frame.append(self.frame_id+i*10)
        return frame
          
    def get_data(self):
        
        loaded_data = np.empty((self.numFrames,),dtype=object).tolist() #{'cov': cov_to_save, 'mean': mean_to_save, 
        process_time = np.empty((self.numFrames,),dtype=object).tolist()
        solve_time = np.empty((self.numFrames,),dtype=object).tolist()
        cost = np.empty((self.numFrames,),dtype=object).tolist()
        X_star = np.empty((self.numFrames,),dtype=object).tolist()
        U_star = np.empty((self.numFrames,),dtype=object).tolist()
        goal = np.empty((self.numFrames,),dtype=object).tolist()
        x_init = np.empty((self.numFrames,),dtype=object).tolist()
        timeout = np.empty((self.numFrames,),dtype=object).tolist()
        ONotZero = np.empty((self.numFrames,),dtype=object).tolist()
        OVconstraint = np.empty((self.numFrames,),dtype=object).tolist()
        v_0_mag = np.empty((self.numFrames,),dtype=object).tolist()

        processTime = []
        solveTime = []
        solveTimeOV = []
        worstTime = []

        # let's save only cov1_save[0][0][t] which means [ov_idx][latent_idx][t]
        for t, frame_name in enumerate(self.frame_list()):
                #filename = str(Agent)+"_frame"+str(frame_name)+"_cov"
            filename = "agent"+str(self.Agent)+"_frame"+str(frame_name)+"_cov"
            path = "out/data/" + self.path
            filepath = os.path.join(path,filename) # write down your path
            with open(filepath,'rb') as f:
                loaded_data[t] = pickle.load(f)
                # print(loaded_data[t]["infeasible"] )
                if loaded_data[t]["infeasible"] == False:
                    process_time[t] = loaded_data[t]['process_time']
                    solve_time[t] = loaded_data[t]['solve_time']
                    cost[t] = loaded_data[t]['cost']
                    X_star[t] = loaded_data[t]["X_star"][0] # X[tau+1|tau]
                    U_star[t] = loaded_data[t]["U_star"][0] # U[tau|tau]
                    goal[t] = loaded_data[t]["goal"]
                    x_init[t] = loaded_data[t]["x_init"] # X[tau|tau]
                    timeout[t] = loaded_data[t]["timeout"]
                    ONotZero[t] = loaded_data[t]["MeanCov"] # there' OV
                    OVconstraint[t] = loaded_data[t]["OVconstraint"] # really consider OV
                    #v_0_mag[t] = loaded_data[t]["v_0_mag"]
        worstTime = 0
        for t in range(self.numFrames):
            if loaded_data[t]["infeasible"] == False:
                if ONotZero[t]:
                    processTime.append(process_time[t])
                    solveTime.append(solve_time[t])
                    worstTime = np.max([worstTime,solve_time[t]])
                    if True: # OVconstraint[t]:
                        solveTimeOV.append(solve_time[t])
        if len(processTime) != 0:
            avrProcessTime = sum(processTime) / len(processTime)
            avrSolveTime = sum(solveTime) / len(solveTime) 
            avrSolveTimeOV = sum(solveTimeOV) / len(solveTimeOV)
        else: # meaningless values
            avrProcessTime = 0
            avrSolveTime = 0
            avrSolveTimeOV = 0                  

        # v_0_mag = pd.DataFrame(v_0_mag).T
        return avrProcessTime, avrSolveTime, avrSolveTimeOV, worstTime, X_star, U_star, goal, x_init, timeout
        #return avrProcessTime, avrSolveTime, X_star, U_star, goal, x_init, timeout
    
    def get_cov_mean(self,plotUntil):
        # load and rearrange convarances and means
        mode =  0 # which mode do you want to plot?
        ov = 0

        cov1_save = np.empty((self.numFrames,),dtype=object).tolist()
        cov2_save = np.empty((self.numFrames,),dtype=object).tolist()
        cov3_save = np.empty((self.numFrames,),dtype=object).tolist()
        cov4_save = np.empty((self.numFrames,),dtype=object).tolist()

        cov1_original = np.empty((self.numFrames,),dtype=object).tolist()
        loadded_cov1_original = np.empty((self.numFrames,),dtype=object).tolist()

        mean1_save = np.empty((self.numFrames,),dtype=object).tolist()
        mean2_save = np.empty((self.numFrames,),dtype=object).tolist()
        mean3_save = np.empty((self.numFrames,),dtype=object).tolist()
        mean4_save = np.empty((self.numFrames,),dtype=object).tolist()

        loaded_data = np.empty((self.numFrames,),dtype=object).tolist() #{'cov': cov_to_save, 'mean': mean_to_save, 'num_mode':num_mode}
        loaded_cov = np.empty((self.numFrames,),dtype=object).tolist()
        loaded_mean = np.empty((self.numFrames,),dtype=object).tolist()
        loaded_num_mode = np.empty((self.numFrames,),dtype=object).tolist() #num_mode[ov_idx] = ovehicle.n_states

        startOVconstraint = 0

        for t, frame_name in enumerate(self.frame_list()):
            filename = "agent"+str(self.Agent)+"_frame"+str(frame_name)+"_cov"
            path = "out/data/" + self.path
            filepath = os.path.join(path,filename) # write down your path
            with open(filepath,'rb') as f:
                loaded_data[t] = pickle.load(f)
                if loaded_data[t]["MeanCov"]:
                    loaded_cov[t] = loaded_data[t]['cov']
                    loaded_mean[t] = loaded_data[t]['mean']
                    loaded_num_mode[t] = loaded_data[t]['num_mode']
                    loadded_cov1_original[t] = loaded_data[t]['cov_original']
                else: 
                    startOVconstraint = startOVconstraint + 1
                    #loaded_cov[t] = [0]
                    #loaded_mean[t] = [0]
                    loaded_num_mode[t] = [0]

        
        Big = 10000
        max_mode = max(max(loaded_num_mode))
        index_list = np.empty((max_mode,plotUntil,),dtype=object).tolist()
        mean_differ = np.empty((max_mode,plotUntil,),dtype=object)
        
        # set up the last mode index
        for mode_idx in range(loaded_num_mode[plotUntil-1][ov]):
            index_list[mode_idx][-1] = mode_idx


        for t in range(plotUntil-2,startOVconstraint - 1, -1):
            mean1_pri, mean2_pri, mean3_pri, mean4_pri = loaded_mean[t]
            mean1, mean2, mean3, mean4 = loaded_mean[t+1]

            for mode_idx in range(loaded_num_mode[plotUntil-1][ov]):
                mean1_temp = np.array(mean1[ov][index_list[mode_idx][t+1]])
                mean2_temp = np.array(mean2[ov][index_list[mode_idx][t+1]])
                for idx in range(loaded_num_mode[t][ov]):
                    mean1_pri_temp = np.array(mean1_pri[ov][idx])
                    mean2_pri_temp = np.array(mean2_pri[ov][idx]) # let's consider one more

                    mean_differ[idx][t] = np.abs(mean1_temp[:self.T-1,2] - mean1_pri_temp[1:self.T,2])
                    mean_differ[idx][t] += np.abs(mean1_temp[:self.T-1,1] - mean1_pri_temp[1:self.T,1])
                    mean_differ[idx][t] += np.abs(mean1_temp[:self.T-1,0] - mean1_pri_temp[1:self.T,0])
                    mean_differ[idx][t] += np.abs(mean2_temp[:self.T-1,2] - mean2_pri_temp[1:self.T,2])
                    mean_differ[idx][t] += np.abs(mean2_temp[:self.T-1,1] - mean2_pri_temp[1:self.T,1])
                    mean_differ[idx][t] += np.abs(mean2_temp[:self.T-1,0] - mean2_pri_temp[1:self.T,0])
                    '''
                    for i in range(len(mean_differ[idx][t])): # decaying, IS IT VALID??
                        mean_differ[idx][t][i] = np.array(i)*mean_differ[idx][t][i]
                    '''
                    mean_differ[idx][t] = np.mean(mean_differ[idx][t])
                for rest in range(loaded_num_mode[t][ov], max_mode):
                    mean_differ[rest][t] = np.mean(Big)
                index_list[mode_idx][t] = np.argmin(mean_differ[:,t])

        for t in  range(startOVconstraint, plotUntil):
            cov1, cov2, cov3, cov4 = loaded_cov[t]
            mean1, mean2, mean3, mean4 = loaded_mean[t] 
                         
            cov1_save[t] = np.array(cov1[ov][index_list[mode][t]])
            cov2_save[t] = np.array(cov2[ov][index_list[mode][t]])
            cov3_save[t] = np.array(cov3[ov][index_list[mode][t]])
            cov4_save[t] = np.array(cov4[ov][index_list[mode][t]])
            mean1_save[t] = np.array(mean1[ov][index_list[mode][t]])
            mean2_save[t] = np.array(mean2[ov][index_list[mode][t]])
            mean3_save[t] = np.array(mean3[ov][index_list[mode][t]])
            mean4_save[t] = np.array(mean4[ov][index_list[mode][t]])
            # print(loadded_cov1_original)
            cov1_original[t] = np.array(loadded_cov1_original[ov][index_list[mode][t]])

        '''
        for t in range(0, startOVconstraint):
            cov1_save[t] = np.array(0)
            cov2_save[t] = np.array(0)
            cov3_save[t] = np.array(0)
            cov4_save[t] = np.array(0)
            mean1_save[t] = np.array(0)
            mean2_save[t] = np.array(0)
            mean3_save[t] = np.array(0)
            mean4_save[t] = np.array(0)
        '''
        return cov1_save, cov2_save, mean1_save, mean2_save, cov1_original
    
    def cost_final(self, X_star, U_star):
        weight = util.AttrDict(
            w_final=6.0, # 3.0
            w_ref = 3.0,
            w_ch_accel=0.5, # 0.5
            w_ch_turning=2.0, # 2.0
            w_ch_joint=0.1, # 0.1
            w_accel=0.5, # 0.5
            w_turning=1.0, # 1.0
            w_joint=0.2, # 0.2
        )
        goal = [167.174698, -81.759842]
        # goal = [-100.5133, -21.2906]
        goal = np.array(goal)
        planning_horizon = len(U_star)

        R1 = [[weight.w_accel, weight.w_joint], [weight.w_joint, weight.w_turning]]
        R2 = [[weight.w_ch_accel, weight.w_ch_joint], [weight.w_ch_joint, weight.w_ch_turning]]
        costX = (
            weight.w_final * np.absolute(X_star[- 1][0] - goal[0])**2
            + weight.w_final * np.absolute(X_star[ - 1][1] - goal[1])**2
        )
        #print(X_star)
        #print(np.absolute(X_star[- 1][0] - goal[0])**2 + np.absolute(X_star[- 1][1] - goal[0])**2)
        #print(costX)
        for tau in range(0, planning_horizon - 1):
            costU = U_star[tau].T @ R1 @ U_star[tau]

        for tau in range(0, planning_horizon - 2):
            costU = costU + (U_star[tau + 1] - U_star[tau]).T @ R2 @ (U_star[tau + 1] - U_star[tau])
        
        cost = costX + costU
        print(cost)
        
        return cost, costU
    
    def plot(self, plotFrom, plotUntil):
        '''
        linestyle_str =[
            'dashdot',
            'dotted',
            'dashed',
            (0,(3,1,1,1,1,1)),
            (0,(5,1)),
            (0,(3,1,1,1)),
        ]
        '''
        linestyle_str =[
            '<-',
            '^-',
            's-',
            'o-',
            ">-",
            'v-',
        ]


        cov1_save, cov2_save, mean1_save, mean2_save, cov1_original= self.get_cov_mean(plotUntil)
        saveplot = True
        plt.rc('font', size = 14)
        plt.figure(figsize=(16.18/1.5,10/1.5))
        cutoff = 0
        m = 6 # makerSize

        for i in range(len(cov1_save)):
            if cov1_save[i] is not None:# and isinstance(cov1_save[i], (int, float)):
                cov1_save[i] = np.sqrt(cov1_save[i])
                cov2_save[i] = np.sqrt(cov2_save[i])
            else:
                cov1_save[i] = 0
                cov2_save[i] = 0

        for i in range(len(cov1_original)):
            if cov1_original[i] is not None:# and isinstance(cov1_save[i], (int, float)):
                cov1_original[i] = np.sqrt(cov1_original[i])
                cov1_original[i] = np.sqrt(cov1_original[i])
            else:
                cov1_original[i] = 0
                cov1_original[i] = 0
        

        # let's make vertical grids
        for tau in range(plotFrom, plotUntil):
            if tau < plotUntil -1:
                c = 'black'
            else:
                c = 'red'
            plt.figure(1)
            plt.subplot(223)
            plt.axvline(x=tau+1-plotFrom,color = c, linestyle = 'dashed',linewidth = 0.9)
            plt.subplot(224)
            plt.axvline(x=tau+1-plotFrom,color = c, linestyle = 'dashed',linewidth = 0.9)
 
            plt.figure(1)
            plt.subplot(221)
            plt.axvline(x=tau+1-plotFrom,color = c, linestyle = 'dashed',linewidth = 0.9)
            plt.subplot(222)
            plt.axvline(x=tau+1-plotFrom,color = c, linestyle = 'dashed',linewidth = 0.9)

        for tau in range(plotFrom, plotUntil):
            time_value = []
            for i in range(self.T):
                time_value.append(tau+1+i-plotFrom) # the prediction is only for future t.
            # make the axis' value
            min_cov1 = min(cov1_save[tau][:self.T-cutoff])
            max_cov1 = max(cov1_save[tau][:self.T-cutoff])
            min_cov2 = min(cov2_save[tau][:self.T-cutoff])
            max_cov2 = max(cov2_save[tau][:self.T-cutoff])
            min_cov_temp = min(min_cov1, min_cov2)
            max_cov_temp = max(max_cov1, max_cov2)
            if tau == plotFrom:
                min_cov = min_cov_temp
                max_cov = max_cov_temp
            else:
                min_cov = min(min_cov,min_cov_temp)
                max_cov = max(max_cov,max_cov_temp)

            min_mean1_temp = min(mean1_save[tau][:,2])
            max_mean1_temp = max(mean1_save[tau][:,2])
            min_mean2_temp = min(mean2_save[tau][:,2])
            max_mean2_temp = max(mean2_save[tau][:,2])


            if tau == plotFrom:
                min_mean1 = min_mean1_temp
                max_mean1 = max_mean1_temp
                min_mean2 = min_mean2_temp
                max_mean2 = max_mean2_temp

            else:
                min_mean1 = min(min_mean1,min_mean1_temp)
                max_mean1 = max(max_mean1,max_mean1_temp)
                min_mean2 = min(min_mean2,min_mean2_temp)
                max_mean2 = max(max_mean2,max_mean2_temp)

            fig = plt.figure(1)

            plt.subplot(223)
            plt.plot(time_value[:self.T-cutoff],cov1_save[tau][:self.T-cutoff],linestyle_str[tau%6],Markersize = m)#,linestyle = linestyle_str[t%6])
            plt.subplot(224)
            plt.plot(time_value[:self.T-cutoff],cov2_save[tau][:self.T-cutoff],linestyle_str[tau%6], 
                     label= f'$\\tau={tau-plotFrom}$',Markersize = m)#,linestyle = linestyle_str[t%6])

            plt.figure(1)
            plt.subplot(221)
            plt.plot(time_value[:self.T-cutoff],mean1_save[tau][:,2][:self.T-cutoff],linestyle_str[tau%6],Markersize = m)#,linestyle = linestyle_str[t%6])
            plt.subplot(222)
            plt.plot(time_value[:self.T-cutoff],mean2_save[tau][:,2][:self.T-cutoff],linestyle_str[tau%6],Markersize = m)#,linestyle = linestyle_str[t%6])
            cutoff = cutoff + 1
        

        range_cov = max_cov - min_cov
        plt.subplot(223)
        plt.title('$\sqrt{||\Sigma^{t|\\tau}||_{F}}$ for Edge 1')
        plt.xlabel('t')
        plt.ylabel('$\sqrt{||\Sigma^{t|\\tau}||_{F}}$ ')
        plt.ylim([min_cov - range_cov/10, max_cov + range_cov/10])
        plt.subplot(224)
        plt.title('$\sqrt{||\Sigma^{t|\\tau}||_{F}}$ for Edge 2')
        plt.xlabel('t')
        #plt.ylabel('$||\Sigma^{t|\\tau}||_{F}$ ')
        plt.ylim([min_cov - range_cov/10, max_cov + range_cov/10])

        range_mean1 = max_mean1 - min_mean1
        mid_mean1 = (max_mean1 + min_mean1)/2
        range_mean2 = max_mean2 - min_mean2
        mid_mean2 = (max_mean2 + min_mean2)/2
        range_mean = max(range_mean1, range_mean2)
        plt.figure(1)
        plt.subplot(221)
        plt.title('$\mu^{t|\\tau} $ for Edge 1')
        plt.xlabel('t')
        plt.ylabel('$mu^{t|\\tau}$ ')
        plt.ylim([mid_mean1 - range_mean/2 - range_mean/10, mid_mean1 + range_mean/2 + range_mean/10])
        plt.ylabel('$\mu^{t|\\tau}$')
        plt.subplot(222)
        plt.title('$\mu^{t|\\tau} $ for Edge 2')
        plt.xlabel('t')
        #plt.ylabel('$\mu^{t|\\tau}$')
        plt.ylim([mid_mean2 - range_mean/2 - range_mean/10, mid_mean2 + range_mean/2 + range_mean/10])

        plt.tight_layout(rect=(0.12,0,1,1)) # recttuple (left, bottom, right, top), default: (0, 0, 1, 1)
        fig.legend(loc = 'center left')
        if saveplot:
            plt.savefig("figures/try/"+str(self.Agent)+"_figure1_"+ str(plotFrom) + "to" + str(plotUntil-1))#  + ".eps", format='eps')
        plt.clf()

        Gamma_constant = np.array(1.6449)
        g_cov1_diff = np.empty((self.numFrames-1,), dtype = object)
        g_cov2_diff = np.empty((self.numFrames-1,), dtype = object)
        h_mean1_diff = np.empty((self.numFrames-1,), dtype = object)
        h_mean2_diff = np.empty((self.numFrames-1,), dtype = object)


        
        '''
        for i in range(len(cov1_save)):
            if cov1_save[i] is not None:# and isinstance(cov1_save[i], (int, float)):
                cov1_save[i] = np.sqrt(cov1_save[i])
                cov2_save[i] = np.sqrt(cov2_save[i])
            else:
                cov1_save[i] = 0
                cov2_save[i] = 0
        '''
        mean1_save = np.array(mean1_save)
        mean2_save = np.array(mean2_save)
        # print(mean1_save[0][1:T,1]) # the time to make predictions, time to predict

        for tau in range(plotFrom,plotUntil-1):
            g_cov1_diff[tau] = Gamma_constant*(cov1_save[tau][1:self.T] - cov1_save[tau+1][0:self.T-1])
            g_cov2_diff[tau] = Gamma_constant*(cov2_save[tau][1:self.T] - cov2_save[tau+1][0:self.T-1])
            h_mean1_diff[tau] = np.linalg.norm(mean1_save[tau][1:self.T]-mean1_save[tau+1][0:self.T-1], axis = 1)
            h_mean2_diff[tau] = np.linalg.norm(mean2_save[tau][1:self.T]-mean2_save[tau+1][0:self.T-1], axis = 1)

        #g_cov_diff1 = g_cov1_diff.tolist()
        #g_cov_diff2 = g_cov2_diff.tolist()
        #h_mean_diff1 = h_mean1_diff.tolist()
        #h_mean_diff2 = h_mean2_diff.tolist()

        g_cov1_diff_tau = np.empty((plotUntil + self.T-1,plotUntil,), dtype = object).tolist()
        g_cov2_diff_tau = np.empty((plotUntil + self.T-1,plotUntil,), dtype = object).tolist()
        h_mean1_diff_tau = np.empty((plotUntil + self.T-1,plotUntil,), dtype = object).tolist()
        h_mean2_diff_tau = np.empty((plotUntil + self.T-1,plotUntil,), dtype = object).tolist()

        ystart = 0
        yend = 5

        xmin = 13
        xmin2 = 14
        xmax2 = plotUntil - 2
        xmax = plotUntil-1 -plotFrom

        plt.rc('font', size=12) 
        plt.figure(figsize=(16.18/1.5,7/1.5))
        
        for t in range(plotFrom+1, plotUntil + self.T - 1):
            for tau in range(np.max([plotFrom, t - self.T + 1]), min([t,plotUntil-1])):
                g_cov1_diff_tau[t][tau] = g_cov1_diff[tau][t-tau-1]
                g_cov2_diff_tau[t][tau] = g_cov2_diff[tau][t-tau-1]
                h_mean1_diff_tau[t][tau] = h_mean1_diff[tau][t-tau-1]
                h_mean2_diff_tau[t][tau] = h_mean2_diff[tau][t-tau-1]

        for t in range(plotFrom+1, plotUntil + self.T - 1):
            tau_value = []
            g_cov1_diff_tau_temp = []
            g_cov2_diff_tau_temp = []
            h_mean1_diff_tau_temp = []
            h_mean2_diff_tau_temp = []
            for tau in range(np.max([plotFrom, t - self.T + 1]), min([t,plotUntil-1])):
                tau_value.append(tau-plotFrom)
                g_cov1_diff_tau_temp.append(g_cov1_diff_tau[t][tau])
                g_cov2_diff_tau_temp.append(g_cov2_diff_tau[t][tau])
                h_mean1_diff_tau_temp.append(h_mean1_diff_tau[t][tau])
                h_mean2_diff_tau_temp.append(h_mean2_diff_tau[t][tau])
                
                fig = plt.figure(2)
                if 2*(t-plotFrom)-1 <= 10:
                    plt.subplot(2,5,t-plotFrom)
                    plt.title(f't = {t+1-plotFrom}')
                    plt.plot(tau_value,h_mean1_diff_tau_temp,'b--',marker = "*",label = '$h^t(\\tau)$', linewidth = 1.0, markersize = 4)
                    plt.yticks(np.arange(ystart,yend,2))
                    plt.subplot(2,5,t-plotFrom)
                    plt.xlim([-1, xmax ])
                    plt.xticks(np.arange(0,xmax,2))
                    plt.plot(tau_value,g_cov1_diff_tau_temp,'ro--',label = '$\Gamma^t g^t(\\tau)$', linewidth = 1.0, markersize = 3)
                    plt.yticks(np.arange(ystart,yend,2))
                    plt.xlabel('$\\tau$')
                    if t-plotFrom == 1:
                        plt.ylabel("(a)", rotation = 0,labelpad=20, fontsize = 14)
        
                    plt.subplot(2,5,t-plotFrom+5)
                    #plt.title(f't = {t}')
                    plt.plot(tau_value,h_mean2_diff_tau_temp,'b--',marker = "*",label = '$h^t(\\tau)$', linewidth = 1.0, markersize = 4)
                    plt.yticks(np.arange(ystart,yend,2))
                    plt.subplot(2,5,t-plotFrom+5)
                    plt.xlim([-1, xmax ])
                    plt.xticks(np.arange(0,xmax,2))
                    plt.plot(tau_value,g_cov2_diff_tau_temp,'ro--',label = '$\Gamma^t g^t(\\tau)$', linewidth = 1.0, markersize = 3)
                    plt.xlabel('$\\tau$')
                    plt.yticks(np.arange(ystart,yend,2))
                    
                    if t-plotFrom == 1:
                        plt.ylabel("(b)", rotation = 0,labelpad=20, fontsize = 14)
                    


        lines, labels = fig.axes[0].get_legend_handles_labels()
        fig.legend(lines, labels, loc = 'lower center',ncol=2,bbox_to_anchor=(0.5, 0))
        #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=1, hspace=10)
        #plt.subplots_adjust( wspace=0.3, hspace=1.5,bottom=0.15)
        #plt.subplots_adjust( hspace=1)
        plt.tight_layout(rect=(0,0.07,1,1))
        #plt.subplots_adjust(hspace=0.4) 
        #plt.subplots_adjust( hspace=1)
        # plt.yticks(np.arange(ystart,yend,3))
        if saveplot:
            plt.savefig("figures/try/"+str(self.Agent)+"_figure2_"+ str(plotFrom) + "to" + str(plotUntil-1) )# + ".eps", format='eps')
        plt.clf()

    