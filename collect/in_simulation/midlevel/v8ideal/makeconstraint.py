import numpy as np
import scipy
from scipy.stats import norm



def compute_mvoe(Sigma1, Sigma2, tol=1e-8, maxiter=1000):
    """
        Compute the Minimum-Volume Outer Ellipsoid (MVOE) of the Minkowski sum
        of two centered ellipoids with covariance Sigma1 and Sigma2
        Args:
            Sigma1:
            Sigma2:
            tol:
            maxiter:
        Returns:
            beta_star   : optimal scalar parameter
            Q_star      : shape matrix of the MVOE (so that (x)^T Q_star^{-1} x <= 1)
    """
    # eigenvalues of R = Sigma1^{-1} Sigma2
    lam = np.linalg.eigvals(scipy.linalg.solve(Sigma1, Sigma2)).real

    # fixed-point iteration for beta
    beta = 1.0
    for _ in range(maxiter):
        num = np.sum(1.0 / (1.0 + beta * lam))
        den = np.sum(lam    / (1.0 + beta * lam))
        beta_new = np.sqrt(num / den)
        if abs(beta_new - beta) < tol:
            beta = beta_new
            break
        beta = beta_new

    # compute shape matrix
    Q_star = (1 + 1.0/beta) * Sigma1 + (1 + beta) * Sigma2
    # x*Q_star*x = 1

    return beta, Q_star


def predict_moments(p_t_tau):
    """
        Predict moments using current prediciton

        Args:   
            p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]
            :{t|\tau} {\tau_future+\tau|\tau}
        Returns:
            cov_infer   : covariance of x^{t|\tau+\tau_future|\tau}
            cov_mu      : covariance of mean^{\tau+\tau_future|\tau}
            cov_t       : covariance of  x^{t|\tau}
    """
    # estimate moments
    # mean = np.mean(p_t_tau, axis = 1)
    cov = np.cov(p_t_tau)

    # predict moments
    cov_t_tau = cov[0:2, 2:4]   # Sigma^{(t|tau)(tau+1|tau)} = cov(mu^{t|tau+1})
    cov_t_tau_T = cov[2:4, 0:2] # Sigma^{(t|tau)(tau+1|tau)} = cov(mu^{t|tau+1})

    cov_t = cov[0:2, 0:2]
    cov_tau = cov[2:4, 2:4]

    # mean^{\tau+\tau_future|\tau}
    cov_mu = cov_t_tau @ np.linalg.inv(cov_tau) @ cov_t_tau_T
    
    # Sigma^{t|\tau+\tau_future}
    cov_infer = cov_t - cov_mu

    return cov_infer, cov_mu, cov_t

def closest_tangent_line(mu, Sigma, c, ref_traj,
                         coarse_steps=1000, refine_steps=200,
                         refine_width=0.02):
    """
    Find the tangent line to the ellipse (x-mu)^T Sigma^{-1} (x-mu) = c^2
    whose distance to the point(ref_traj) is minimal

    Returns:
        n_star  : normal vector of the tangent line (shape (2,))
        d_star  : offset so that the line is { x : n_star^T x = d_star }
    """
    # Precompute a matrix S such that S @ S.T = Sigma (Cholesky)
    Sigma_sqrt = np.linalg.cholesky(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    def distance_to_tangent(theta):
        """
        Compute the signed perpendicular distance from point a to the tangent
        line at angle theta.  We return the absolute distance.
        """
        # 1) u(θ) = c * Sigma_sqrt @ [cosθ, sinθ]
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        u = c * (Sigma_sqrt @ np.array([cos_t, sin_t]))  # shape (2,)

        # 2) normal n = Sigma^{-1} u
        n = Sigma_inv @ u  # shape (2,)

        # 3) offset d = u^T Sigma^{-1} mu + c^2
        d = (u @ (Sigma_inv @ mu)) + c**2

        # 4) signed distance = (n^T a - d) / ||n||
        # we only care about absolute value
        num = abs(n @ ref_traj - d)
        denom = np.linalg.norm(n)
        return num / denom
    
    # 1) Coarse grid search over [0, 2π)
    thetas_coarse = np.linspace(0.0, 2*np.pi, coarse_steps, endpoint=False)
    distances_coarse = np.array([distance_to_tangent(t) for t in thetas_coarse])
    idx_min = np.argmin(distances_coarse)
    theta_coarse_best = thetas_coarse[idx_min]

    # 2) Refine around theta_coarse_best ± refine_width
    low = theta_coarse_best - refine_width
    high = theta_coarse_best + refine_width
    thetas_refine = np.linspace(low, high, refine_steps)
    distances_refine = np.array([distance_to_tangent(t) for t in thetas_refine])
    idx_refine = np.argmin(distances_refine)
    theta_star = thetas_refine[idx_refine]
    distance_star = distances_refine[idx_refine]    # the perpendicular distance from traj_ref to that line

    # 3) Recompute final u*, n*, d*, x0*
    cos_t = np.cos(theta_star)
    sin_t = np.sin(theta_star)
    u_star = c * (Sigma_sqrt @ np.array([cos_t, sin_t]))      # shape (2,)
    x0_star = mu + u_star                                      # tangent point
    n_star = Sigma_inv @ u_star                                # normal vector
    d_star = (u_star @ (Sigma_inv @ mu)) + c**2                 # offset

    return  n_star, d_star
    
def tangent_lines_of_slope_m(mu, Sigma, c, m):
    """
    Find the two tangent lines of slope m to the ellipse
       (x-mu)^T Sigma^{-1} (x-mu) = c^2.
    Returns a list of two (n, d) pairs, where each line is:
       n^T x = d,
    with n = [-m, 1] (normal vector) and d = intercept.
    If no real tangents exist, returns an empty list.
    """
    # 1) build the normal vector n = [-m, 1]
    n = np.array([-m, 1.0])   # shape (2,)

    # 2) compute n^T Sigma n  (a positive scalar if Sigma >> 0)
    n_Sigma_n = float(n @ (Sigma @ n))
    if n_Sigma_n <= 0:
        # Should not happen if Sigma is positive-definite, but check anyway.
        return []

    # 3) compute mu_proj = n^T mu
    mu_proj = float(n @ mu)

    # 4) check discriminant: we need (d - n^T mu)^2 = c^2 * (n^T Sigma n)
    #    so d = mu_proj ± c * sqrt(n_Sigma_n).
    delta = c * np.sqrt(n_Sigma_n)
    d1 = mu_proj + delta
    d2 = mu_proj - delta

    # 5) return the two (n,d) pairs
    return [(n, d1), (n, d2)]


def distance_point_to_line(n, d, a):
    """
    Given a line n^T x = d (with n in R^2 and d a scalar) 
    and a point a in R^2, compute the (perpendicular) distance
    from a to that line, i.e. |n^T a - d| / ||n||_2.
    """
    numer = abs(float(n @ a) - d)
    denom = np.linalg.norm(n)
    return numer / denom


def choose_closest_tangent(mu, Sigma, c, m, a, const_idx = None):
    """
    1) Compute the two tangents of slope m to the ellipse 
       (x-mu)^T Sigma^{-1} (x-mu) = c^2.
    2) Among those that are real, return the one whose distance
       to point a is smaller.
    Returns a tuple (n_star, d_star, which_index, dist_star), where:
      - n_star is the normal vector of the chosen line (shape (2,))
      - d_star is the offset so that the line is { x : n_star^T x = d_star }
      - which_index is 0 or 1 indicating whether we picked the “+” or “–” version
      - dist_star is the distance from a to that chosen line
    If no tangent exists, returns (None, None, None, None).
    """
    # Get the two candidate tangents
    candidates = tangent_lines_of_slope_m(mu, Sigma, c, m)
    if len(candidates) == 0:
        return (None, None, None, None)
    
    if const_idx == None:
        best_idx = None
        best_dist = np.inf

        for idx, (n, d) in enumerate(candidates):
            dist = distance_point_to_line(n, d, a)
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
    else: 
        best_idx = const_idx

    n_star, d_star = candidates[best_idx]
    return n_star, d_star, best_idx



# Minkowski sum
"""
    for latent_idx in range(ovehicle.n_states):
        pose_data[ov_idx][latent_idx] = np.vstack(ovehicle.pred_positions[latent_idx])
        poseData = np.vstack(ovehicle.pred_positions[latent_idx])
        for t in range(T):
            p0_t = poseData[t::Tpred,0]
            p1_t = poseData[t::Tpred,1]

            scale = 1.0
            
            for tau_future in range(t):
                p0_tau = poseData[tau_future::Tpred,0]
                p1_tau = poseData[tau_future::Tpred,1]

                p_t_tau = [p0_t, p1_t, p0_tau, p1_tau]

                # predict moments
                cov_infer, cov_mu, _ = predict_moments(p_t_tau)


                # choice: scaling or Minkowski sum

                
                #Minkowski sum

                

                eps_t = 0.05/8                     # predefined
                chi_risk_tol = scipy.stats.chi2.ppf(1-eps_t, df = 2)
                target_p = 0.9999
                chi_target_p = scipy.stats.chi2.ppf(target_p, df = 2)

                _, cov_mvoe = compute_mvoe(cov_infer*chi_risk_tol , cov_mu*chi_target_p )


                
                # Add Constraint using cov_mvoe
                #기울기 고정할 필요 없다!

                # find the tangent line
                
                mean = np.mean([p0_t, p1_t], axis = 1)
                n_star, d_star = closest_tangent_line(mean, cov_mvoe, 1, ref_traj)

                #  n_star^T x  >= d_star + buffer 
                """

def compute_scale(cov_infer, cov_mu, cov_t, Gamma_ijt, target_p = 0.9999):
    """
        Compute the scale factor

        Return:
            scale: scaling factor to satisfy the recursively feasible assumption
    """
    nominator = np.sqrt(np.linalg.norm(cov_t, 'fro'))
    denominator_alpha = np.sqrt(np.linalg.norm(cov_infer, 'fro'))
    denominator_beta = np.sqrt(np.linalg.norm(cov_mu, 'fro'))

    alpha = denominator_alpha/nominator
    beta = denominator_beta/nominator

    # Gamma_ijt = norm.ppf(1-eps_ijt) 
    lamb = target_p               # target prob to satisfy the assumption
    chi_p = scipy.stats.chi2.ppf(lamb, df = 2)
    scale_temp = (np.sqrt(chi_p)*beta/Gamma_ijt+alpha)**2

    # scale = np.max((scale_temp,scale))  # only take the largest one

    return scale_temp

def compute_lower_bound(cov_infer, cov_mu, cov_t, eps_t = 0.05/8):
    """
        Compute the lower bound of the prob. to satisfy the condition for recursively feasibility

        Return:
            prob
    """
    nominator = np.sqrt(np.linalg.norm(cov_t, 'fro'))
    denominator_alpha = np.sqrt(np.linalg.norm(cov_infer, 'fro'))
    denominator_beta = np.sqrt(np.linalg.norm(cov_mu, 'fro'))

    alpha = denominator_alpha/nominator
    beta = denominator_beta/nominator

    Gamma_ijt = norm.ppf(1-eps_t)
    chi_squared = (Gamma_ijt*(1-alpha)/beta)**2
    prob_lower = scipy.stats.chi2.cdf(chi_squared, df = 2)
    # Gamma_ijt = norm.ppf(1-eps_ijt) 

    # scale = np.max((scale_temp,scale))  # only take the largest one

    return prob_lower


