import numpy as np



mean_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
cov_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()

cross_cov = np.empty((np.max(O), np.max(K), T, T-1), dtype=object).tolist()

def save_moments(ovehicles, O, K, T, Tpred, ego_vehicle_id, params):
    """
    Save mean, covariance, and cross-covariance of predictions
    """
    # estimate moments
    mean_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
    cov_p0p1 = np.empty((np.max(O), np.max(K), T), dtype=object).tolist()
    cross_cov = np.empty((np.max(O), np.max(K), T, T-1), dtype=object).tolist()

    for ov_idx, ovehicle in enumerate(ovehicles):
        for latent_idx in range(ovehicle.n_states):
            poseData = np.vstack(ovehicle.pred_positions[latent_idx])
            for t in range(T):
                p0_t = poseData[t::Tpred,0]
                p1_t = poseData[t::Tpred,1]

                p_t = [p0_t, p1_t]
                mean_t = np.mean(p_t, axis = 1)
                cov_t = np.cov(p_t)

                mean_p0p1[ov_idx][latent_idx][t] = mean_t
                cov_p0p1[ov_idx][latent_idx][t] = cov_t

                for tau_future in range(t):
                    p0_tau = poseData[tau_future::Tpred,0]
                    p1_tau = poseData[tau_future::Tpred,1]

                    p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]
                    cov = np.cov(p_t_tau)

                    # predict moments
                    cov_t_tau = cov[0:2, 2:4]   # Sigma^{(t|tau)(tau+1|tau)} = cov(mu^{t|tau+1})
                    cross_cov[ov_idx][latent_idx][t][tau_future] = cov_t_tau
    moments_save = {}
    moments_save['mean_p0p1'] = mean_p0p1
    moments_save['cov_p0p1'] = cov_p0p1
    moments_save['cross_cov'] = cross_cov

    filename = f"agent{ego_vehicle_id}_frame{params.frame}_moments"
    filepath = os.path.join("out/data",filename)
    try:
        with open(filepath, "wb") as f:
            pickle.dump(moments_save, f)
    except Exception as e:
        logging.info(f"Error while saving: {str(e)}")

def predict_ideal(self, ovehicles, T, ego_vehicle_id, params):
    """
        Predict trajectories using saved moments
    """

    # loading moments
    filename = f"agent{ego_vehicle_id}_frame{params.frame-10}_moments"
    filepath = os.path.join("out/data", filename)
    try:
        with open(filepath, 'rb') as f:
            loaded_data = pickle.load(f)
            mean_p0p1 = loaded_data['mean_p0p1']
            cov_p0p1 = loaded_data['cov_p0p1']
            cross_cov = loaded_data['cross_cov']
    except Exception as e:
        logging.info(f"Error while loading: {str(e)}")

    # generate samples
    Tsh = T
    traj_all = {}
    n_samples = 5000
    for ov_idx, ovehicle in enumerate(ovehicles):

        # --- (A) Load the current position
        traj_all.setdefault(ov_idx, {})
        x_cur = ovehicle.past[-1]

        for latent_idx in range(ovehicle.n_states):
            x_cur_batch = x_cur*np.ones((n_samples, 2))
            traj_batch = np.zeros((n_samples, T, 2))
            # traj_batch[:, 0, :] = x_cur_batch

            #  --- (B) Iterate for t=0 ... Tsh-1 to update all samples at once ---
            for t in range(0, Tsh):
                # (B1) Distribution parameters at previous time τ = t
                mean_tau = mean_p0p1[ov_idx][latent_idx][t]    # (2,)
                cov_tau  = cov_p0p1[ov_idx][latent_idx][t]     # (2,2)

                # (B2) Distribution parameters at next time τ+1
                mean_next = mean_p0p1[ov_idx][latent_idx][t+1] # (2,)
                cov_next  = cov_p0p1[ov_idx][latent_idx][t+1]  # (2,2)

                # (B3) Cross‐covariance: Cov(x_{τ+1}, x_{τ})
                C_t_tp1 = cross_cov[ov_idx][latent_idx][t+1][t]  # (2,2)

                # (B4) Inverse of cov_tau
                cov_tau_inv = np.linalg.inv(cov_tau)

                # (B5) Vectorized conditional mean:
                #     cond_mean_i = mean_next + C_t_tp1 @ inv(cov_tau) @ ( x_cur_batch_i - mean_tau )
                diff = x_cur_batch - mean_tau                 # shape (n_samples, 2)
                A = C_t_tp1 @ cov_tau_inv                      # shape (2,2)
                cond_mean = mean_next + (diff @ A.T)           # shape (n_samples, 2)

                # (B6) Conditional covariance (same for all samples)
                cond_cov = cov_next - A @ C_t_tp1.T            # shape (2,2)

                # (B7) cond_cov의 Cholesky 분해: L @ L^T = cond_cov
                L = np.linalg.cholesky(cond_cov)               # shape (2,2)

                # (B8) Draw standard normal noise Z in one shot: shape (n_samples,2)
                rng = np.random.RandomState(seed=None)                  # 혹은 미리 만들어둔 시드
                Z = rng.standard_normal((n_samples, 2))       # shape (n_samples, 2)

                # (B9) Sample next x_cur_batch: cond_mean + Z @ L.T
                x_cur_batch = cond_mean + (Z @ L.T)            # (n_samples, 2)

                # (B10) Store into traj_batch
                traj_batch[:, t, :] = x_cur_batch

            # --- (C) Save results ---
            traj_all[ov_idx][latent_idx] = traj_batch

    return traj_all

## predict

shrinking_horizon_step = Tpred-T
ov_cur_p = 0
for ov_idx, ovehicle in enumerate(ovehicles):
    for latent_idx in range(ovehicle.n_states):
        for t in range(T):
            mean_t = mean_p0p1[ov_idx][latent_idx][t+shrinking_horizon_step]
            cov_t = cov_p0p1[ov_idx][latent_idx][t+shrinking_horizon_step]

            cov_taufuture = mean_p0p1[ov_idx][latent_idx][shrinking_horizon_step]
            mean_taufuture = mean_p0p1[ov_idx][latent_idx][shrinking_horizon_step]

            cross_cov_t_taufuture = \
                cross_cov[ov_idx][latent_idx][t+shrinking_horizon_step][shrinking_horizon_step]

            mean_t_predict = mean_t \
                + cross_cov_t_taufuture @ np.linalg.inv(cov_taufuture) @ (ov_cur_p - mean_taufuture)
            cov_t_predict = cov_t \
                - cross_cov_t_taufuture @ np.linalg.inv(cov_taufuture) @ cross_cov_t_taufuture.T

            # use mean_t_predict and cov_t_predict to make constraints!

            # Minkowski ??
            # Scale ??
            
traj = []

for idx in range(1000):
    for ov_idx, ovehicle in enumerate(ovehicles[7]):
        for latent_idx in range(ovehicle.n_states):
            t = 0
            # 실제로는 측정해서 받아오는 값
            mean_t = mean_p0p1[ov_idx][latent_idx][0]
            cov_t = cov_p0p1[ov_idx][latent_idx][0]
            x_cur = np.random.multivariate_normal(mean_t, cov_t)

            for t in range(Tsh):
                mean_t = mean_p0p1[ov_idx][latent_idx][t+1]
                cov_t = cov_p0p1[ov_idx][latent_idx][t+1]

                mean_taufuture = mean_p0p1[ov_idx][latent_idx][t]
                cov_taufuture = cov_p0p1[ov_idx][latent_idx][t]

                cross_cov_t_taufuture = \
                    cross_cov[ov_idx][latent_idx][t+1][t]

                mean_t_predict = mean_t.T \
                    + cross_cov_t_taufuture @ np.linalg.inv(cov_taufuture) @ (x_cur - mean_taufuture).T
                cov_t_predict = cov_t \
                    - cross_cov_t_taufuture @ np.linalg.inv(cov_taufuture) @ cross_cov_t_taufuture.T

                x_cur = np.random.multivariate_normal(mean_t_predict, cov_t_predict)


                traj.append(x_cur)

def compute_obstacle_constraints_GMM_Minkowski_idealprediction(
    self, params, ovehicles, Delta2, Omicron, temp_x, eps_ura, segments, Tsh, ref_traj     
):
    constraints = []
    M_big = self.__params.M_big
    T = Tsh
    L, K = self.__params.L, params.K
    truck = {'d': np.array([3.7, 1.79])}
    truck_d = truck['d']
    CAR_R = 4.47213 # this is actually a diameter

    if self.road_boundary_constraints:
        # Auxiliary variable for masking and summing
        Z = cp.Variable(Omicron.shape)

        # Constraints for masking
        mask_constraints = [
            Z[i, j] == (Omicron[i, j] if not segments.mask[i] else 0)
            for i in range(Omicron.shape[0])
            for j in range(Omicron.shape[1])
        ]

        # Summing along axis=0
        Z_sum = M_big * cp.sum(Z, axis=0)

        # Variable and constraints for repeating
        S_big_repeated = cp.Variable((L, T))
        repeat_constraints = [S_big_repeated[i, :] == Z_sum for i in range(L)]

        constraints += mask_constraints
        constraints += repeat_constraints
    else:
        S_big_repeated = np.zeros((L, T), dtype=float)

    Tpred = self.__control_horizon

    # ideal prediction
    if T < Tpred:
        ideal_trajs = self.predict_ideal(ovehicles, T, self.__ego_vehicle.id, params)

    # Let's check if the first prediction passes the intersection or not.
    O = params.O
    OV_injuction = np.empty((O,), dtype=object).tolist()
    direct = np.empty((O,), dtype=object).tolist()
    for ov_idx, ovehicle in enumerate(ovehicles):   
        for latent_idx in range(ovehicle.n_states):
            poseData = np.vstack(ovehicle.pred_positions[latent_idx])
            posemean_x = np.mean(poseData[0::Tpred,0])
            posemean_y = np.mean(poseData[0::Tpred,1])
            if posemean_x >= 190 or posemean_y <=-80: # in Town03_scene4 T inersection.
            #if posemean_x <= -98.122 or (posemean_x <= -83.122 and posemean_y <= -21.178):# or posemean_y >= -21.178: # in Town03_scene3 Star intersection
            #    OV_injuction[ov_idx] = False
            #elif (posemean_x >= -80.122 and posemean_y >= 10.178): Star intersection
                OV_injuction[ov_idx] = False 
            else:
                OV_injuction[ov_idx] = True
            #    direct[ov_idx] = "InJunction"
    

    #logging.info(f"OV{OV_injuction}")
    logging.info(f"K:{params.K}")
    #if OV_injuction:
    OVconstraint = False
    for ov_idx, ovehicle in enumerate(ovehicles):
        OVconstraint = OVconstraint or OV_injuction[ov_idx]

    # Save pose, angle [tau+1|tau] of OV
    posemean_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
    posemean_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
    yawmean_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
    posecov_x_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
    posecov_y_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()
    yawcov_save = np.empty((np.max(O), np.max(K)), dtype=object).tolist()

    for ov_idx, ovehicle in enumerate(ovehicles):
        for latent_idx in range(ovehicle.n_states):
            # load data
            poseData = np.vstack(ovehicle.pred_positions[latent_idx])
            yawData = np.vstack(ovehicle.pred_yaws[latent_idx])

            posemean_x_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,0])
            posemean_y_save[ov_idx][latent_idx] = np.mean(poseData[0::Tpred,1])
            yawmean_save[ov_idx][latent_idx] = np.mean(yawData[:,0])
            posecov_x_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,0])
            posecov_y_save[ov_idx][latent_idx] = np.cov(poseData[0::Tpred,1])
            yawcov_save[ov_idx][latent_idx] = np.cov(yawData[:,0])
    
    ovehicles_toSave_moments = []
    for ov_idx, ovehicle in enumerate(ovehicles):
        ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))
    ovehicles_toSave_moments = []          
    for ov_idx, ovehicle in enumerate(ovehicles):
        ovehicles_toSave_moments.append(copy.deepcopy(ovehicle))     
    for ov_idx, ovehicle in enumerate(ovehicles):
        if OV_injuction[ov_idx]:
            for latent_idx in range(ovehicle.n_states):
                
                if T < self.__prediction_horizon:
                    poseData = np.vstack(ideal_trajs[ov_idx][latent_idx])
                    ovehicles_toSave_moments[ov_idx].pred_positions[latent_idx] = ideal_trajs[ov_idx][latent_idx]
                    Tpred = T
                else:
                    poseData = np.vstack(ovehicle.pred_positions[latent_idx])
                
                for t in range(T):
                    p0_t = poseData[t::Tpred,0]
                    p1_t = poseData[t::Tpred,1]

                    for tau_future in range(t):
                        # tau_future = 0
                        p0_tau = poseData[tau_future::Tpred,0]
                        p1_tau = poseData[tau_future::Tpred,1]
                                            
                        p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]

                        # predict moments
                        cov_infer, cov_mu, _ = makeconstraint.predict_moments(p_t_tau)

                        # Minkowski sum
                        eps_ijt = eps_ura[ov_idx, latent_idx] / self.__control_horizon             
                        chi_risk_tol = scipy.stats.chi2.ppf(1-eps_ijt, df = 2)
                        target_p = 0.999
                        chi_target_p = scipy.stats.chi2.ppf(target_p, df = 2)

                        _, cov_mvoe = makeconstraint.compute_mvoe(cov_infer*chi_risk_tol , cov_mu*chi_target_p )

                        mean = np.mean([p0_t, p1_t], axis = 1)
                        
                        ref_x = [ref_traj[t][0], ref_traj[t][1]]
                        # n_star, d_star = makeconstraint.closest_tangent_line(mean, cov_mvoe, 1, ref_x)
                        
                        m = - (ref_traj[t][0] - mean[0]) / (ref_traj[t][1] - mean[1]) # fix tangent
                        n_star, d_star, _ = makeconstraint.choose_closest_tangent(mean , cov_mvoe, 1, m, ref_x)
                        
                        constraints += [
                            n_star.T @ cp.vstack([temp_x[t][0], temp_x[t][1]]) >= d_star # + 0.5 * CAR_R 
                        ]
                        # NOTE: maybe we can optimize the buffer(0.5 * CAR_R ) 
        else:
            constraints += []
    # constraints = []
    vertices = self.__compute_vertices(params, ovehicles)
    A_union, b_union = self.__compute_overapproximations(
        params, ovehicles, vertices
    )

    ovStateMean_tau_1 = (posemean_x_save, posemean_y_save, yawmean_save)
    ovStateCov_tau_1 = (posecov_x_save, posecov_y_save, yawcov_save)

    # save moments
    self.save_moments(ovehicles_toSave_moments, params.O, K, T, self.__control_horizon, self.__ego_vehicle.id, params)

    return constraints, vertices, A_union, b_union, OVconstraint, direct, ovStateMean_tau_1, ovStateCov_tau_1, 0
