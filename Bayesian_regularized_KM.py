import math
import time
import numpy as np
from types import SimpleNamespace


def gigrnd(p, a, b, sample_size=1):
    """
    Generate random variates from the Generalized Inverse Gaussian (GIG) distribution.
    Density proportional to: x^(p-1) * exp(-0.5 * (a*x + b/x)) for x > 0.
    Implementation uses Devroye's (2014) / Dagpunar's algorithm for GIG sampling.
    """
    if a <= 0 and b <= 0:
        raise ValueError("At least one parameter of a and b must be positive.")

    # Special cases: Gamma and Inverse Gamma
    if a > 0 and b <= 0:
        if p <= 0:
            raise ValueError("Invalid parameters for Gamma limiting distribution.")
        # Gamma(shape=p, scale=2/a)
        res = np.random.gamma(shape=p, scale=2.0 / a, size=sample_size)
        return res[0] if sample_size == 1 else res

    if a <= 0 and b > 0:
        if p >= 0:
            raise ValueError("Invalid parameters for Inverse-Gamma limiting distribution.")
        # Inverse-Gamma(shape=-p, scale=b/2)
        res = 1.0 / np.random.gamma(shape=-p, scale=2.0 / b, size=sample_size)
        return res[0] if sample_size == 1 else res

    # Standardize to GIG(p, lambda_param, omega) where a*x + b/x = omega * (x/eta + eta/x)
    # with eta = sqrt(b/a), omega = sqrt(a*b).
    # Then sample standard GIG(p, omega) and scale by eta.
    eta = math.sqrt(b / a)
    omega = math.sqrt(a * b)
    abs_p = abs(p)

    samples = np.empty(sample_size)

    # Devroye (2014) algorithm for standard GIG(p, omega)
    # with parameterizations depending on p and omega
    for idx in range(sample_size):
        # Case 1: p == 0.5 (Inverse Gaussian)
        if abs_p == 0.5:
            # Wald / Inverse Gaussian generator
            # mu = 1.0, lambda_param = omega
            mu_val = 1.0
            y_val = np.random.standard_normal() ** 2
            x_val = mu_val + (mu_val**2 * y_val) / (2.0 * omega) - (mu_val / (2.0 * omega)) * math.sqrt(4.0 * mu_val * omega * y_val + (mu_val * y_val)**2)
            u_val = np.random.uniform(0.0, 1.0)
            if u_val <= mu_val / (mu_val + x_val):
                z_val = x_val
            else:
                z_val = (mu_val**2) / x_val
            if p < 0:
                z_val = 1.0 / z_val
            samples[idx] = z_val * eta
            continue

        # General GIG sampling via Devroye (2014) / Dagpunar rejection method
        lam = abs_p
        alpha_val = math.sqrt(omega**2 + lam**2) - lam

        # Setup envelope
        t = 0.5 * (math.sqrt(omega**2 + lam**2) - lam)
        if t == 0:
            t = omega / (2.0 * lam)

        x0 = (lam + math.sqrt(lam**2 + omega**2)) / omega
        # Shift and scale parameters for two-sided exponential or ratio-of-uniforms
        # Using Dagpunar's (1989) / Leemis algorithm for GIG:
        psi = -omega - lam * math.log(x0) + lam
        # Rejection method from Devroye (2014), "Random variate generation for the generalized inverse Gaussian distribution"
        # Algorithm based on ratio of uniforms or transformed rejection
        m_val = (lam - 1.0 + math.sqrt((lam - 1.0)**2 + omega**2)) / omega
        if m_val <= 0:
            m_val = omega / (2.0 * (1.0 - lam)) if (1.0 - lam) > 0 else 1.0

        # Short & quick ratio-of-uniforms setup:
        # For general GIG with density f(x) \propto x^{lam-1} exp(-0.5*omega*(x + 1/x))
        # Find mode
        mode_val = (lam - 1.0 + math.sqrt((lam - 1.0)**2 + omega**2)) / omega if lam >= 1.0 else omega / (1.0 - lam + math.sqrt((1.0 - lam)**2 + omega**2))
        log_f_mode = (lam - 1.0) * math.log(mode_val) - 0.5 * omega * (mode_val + 1.0 / mode_val)

        # Ratio of uniforms bounding box:
        # u in [0, sqrt(f(mode))]
        # v in [0, v_plus] where v_plus = sup (x * sqrt(f(x)))
        # For x > 0: g(x) = x * sqrt(f(x)) = x^{(lam+1)/2} exp(-0.25*omega*(x + 1/x))
        # Mode of g(x):
        mode_g = (lam + 1.0 + math.sqrt((lam + 1.0)**2 + omega**2)) / omega
        log_g_mode = ((lam + 1.0) / 2.0) * math.log(mode_g) - 0.25 * omega * (mode_g + 1.0 / mode_g)
        u_max = 1.0  # normalized by sqrt(f(mode))
        v_max = math.exp(log_g_mode - 0.5 * log_f_mode)

        while True:
            u_rnd = np.random.uniform(0.0, u_max)
            v_rnd = np.random.uniform(0.0, v_max)
            candidate = v_rnd / u_rnd
            if candidate <= 0:
                continue
            log_f_cand = (lam - 1.0) * math.log(candidate) - 0.5 * omega * (candidate + 1.0 / candidate)
            if math.log(u_rnd) <= 0.5 * (log_f_cand - log_f_mode):
                z_val = candidate
                break

        if p < 0:
            z_val = 1.0 / z_val
        samples[idx] = z_val * eta

    return samples[0] if sample_size == 1 else samples


def kernel_cov_matrix(z):
    """
    Computes the base kernel matrix Ktmp where:
    Ktmp(i,j) = exp(-|zi - zj|^2)
    such that Ktmp^(phi^2) = exp(-(phi^2)*|zi - zj|^2).
    """
    z_mat = np.atleast_2d(z)
    if z_mat.ndim == 1 or (z_mat.shape[0] == 1 and z_mat.shape[1] > 1 and z.ndim == 1):
        z_mat = z_mat.reshape(-1, 1)
    elif z.ndim == 1:
        z_mat = z.reshape(-1, 1)
    
    # Pairwise squared Euclidean distances
    diff = z_mat[:, np.newaxis, :] - z_mat[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    return np.exp(-dist_sq)


def Est_TVL(y, x, z, burnin, ndraws, alpha0_var):
    """
    Estimate the TVL model:
    yt = xt'*alpha + sigma*f(zt) + ut, ut~N(0,s), f~N(0,K) with K(i,j)=exp(-(phi^2)*|zi-zj|^2)).

    Inputs:
      y: a n-by-1 vector of targets.
      x: a n-by-m matrix of linear regressors.
      z: a n-by-mz matrix of nonlinear regressors. 
      burnin: an integer of the number of burn-ins.
      ndraws: an integer of the number of draws after burn-in.
      alpha0_var: a scalar of the prior variance of the intercept (e.g. 100)
    Outputs:
      draws: a structure (SimpleNamespace) with the following fields:
        draws.alpha: a ndraws-by-m matrix of linear coef.
        draws.sigma: a ndraws-by-1 vector of sigma.
        draws.s: a ndraws-by-1 vector of residual variance s.
        draws.phi: a ndraws-by-1 vector of phi.
        draws.f: a ndraws-by-n matrix of f.
        draws.yfit: a ndraws-by-n matrix of xt'*alpha+sigma*f(zt).
        draws.tau: a ndraws-by-1 vector of global para of [alpha(2:m);sigma].
        draws.tau0: a ndraws-by-1 vector of hyperpara for tau.
        draws.lambda: a ndraws-by-m matrix of local para of [alpha(2:m);sigma].
        draws.lambda0: a ndraws-by-m matrix of hyperpara for lambda.
        draws.phi_lambda: a ndraws-by-1 vector of local para of phi.
        draws.phi_lambda0: a ndraws-by-1 vector of hyperpara for phi_lambda.
        draws.s0: a ndraws-by-1 vector of local para of s.
        draws.rw: a ndraws-by-2 matrix of MH tuning para for [phi sigma].
        draws.ff: a ndraws-by-n matrix of sigma*f.
        draws.xalpha: a ndraws-by-n matrix of x*alpha.
        draws.count_phi: a scalar of MH acceptance rate for phi.
        draws.count_sigma: a scalar of MH acceptance rate for sigma.
    """
    # Ensure inputs are standard 2D / 1D NumPy arrays
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    z = np.asarray(z, dtype=float)

    ntotal = burnin + ndraws
    n, m = x.shape

    alpha0 = math.sqrt(alpha0_var) * np.random.randn()  # prior draw of intercept from N(0,alpha0_var)

    n2 = n**2
    # Matlab gamrnd(a, b) has shape a and scale b
    tau0 = 1.0 / np.random.gamma(0.5, 1.0 / n2)
    tau = 1.0 / np.random.gamma(0.5, tau0)
    lambda0 = 1.0 / np.random.gamma(0.5, 1.0, size=(m, 1))
    lambda_ = 1.0 / np.random.gamma(0.5, lambda0)
    beta = math.sqrt(tau) * np.sqrt(lambda_) * np.random.randn(m, 1)  # prior draw of alpha(2:m) and sigma
    sigma = float(beta[m - 1, 0])
    alpha = np.vstack([[[alpha0]], beta[0 : m - 1]])

    phi_lambda0 = 1.0 / np.random.gamma(0.5, 1.0)
    phi_lambda = 1.0 / np.random.gamma(0.5, phi_lambda0)
    phi = math.sqrt(phi_lambda) * np.random.randn()  # prior draw of phi

    Ktmp = kernel_cov_matrix(z)
    Kcov = Ktmp ** (phi**2)
    Ku, Kd_diag, _ = np.linalg.svd(Kcov, full_matrices=True)
    ftmp = np.sqrt(Kd_diag)[:, np.newaxis] * np.random.randn(n, 1)
    f = Ku @ ftmp  # prior draw of nonlinear part f

    s0 = 1.0 / np.random.gamma(0.5, 1.0)
    s_a0 = 0.5
    s_b0 = 1.0 / s0
    # s = 1/gamrnd(0.5,s0); %prior draw of residual variance s~IG(0.5,1/s0),s0~IG(0.5,1)

    pstar = 0.44  # target acceptance prob of MH
    rw_phi = 0.01
    rw_sigma = 0.01  # stdev of MH steps 
    count_phi = 0
    count_sigma = 0  # MH acceptance counter

    draws = SimpleNamespace()
    draws.alpha = np.zeros((ndraws, m))
    draws.sigma = np.zeros((ndraws, 1))
    draws.s = np.zeros((ndraws, 1))
    draws.phi = np.zeros((ndraws, 1))
    draws.f = np.zeros((ndraws, n))
    draws.yfit = np.zeros((ndraws, n))
    draws.tau = np.zeros((ndraws, 1))
    draws.tau0 = np.zeros((ndraws, 1))
    draws.lambda_ = np.zeros((ndraws, m))
    setattr(draws, 'lambda', draws.lambda_)
    draws.lambda0 = np.zeros((ndraws, m))
    draws.phi_lambda = np.zeros((ndraws, 1))
    draws.phi_lambda0 = np.zeros((ndraws, 1))
    draws.s0 = np.zeros((ndraws, 1))
    draws.rw = np.zeros((ndraws, 2))  # phi,sigma
    draws.ff = np.zeros((ndraws, n))  # sigma*f
    draws.xalpha = np.zeros((ndraws, n))  # x*alpha
    draws.count_phi = 0.0
    draws.count_sigma = 0.0

    tic_time = time.time()
    for drawi in range(1, ntotal + 1):
        # Draw s
        s01 = s_a0 + 0.5 * n
        xx = np.hstack([x, f])
        eps = y - xx @ np.vstack([alpha, [[sigma]]])
        s02 = s_b0 + 0.5 * float(eps.T @ eps)
        s = 1.0 / np.random.gamma(s01, 1.0 / s02)

        # Draw s0
        s0 = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / s))
        s_b0 = 1.0 / s0

        # Draw alpha (integrate out f)
        sigma2 = sigma**2
        tmpu = Ku
        tmpd_diag = s + sigma2 * Kd_diag
        tmp_ux = tmpu.T @ x
        tmp_uy = tmpu.T @ y
        tmpd_inv = np.diag(1.0 / tmpd_diag)
        prior_alpha_var_inv = np.diag(np.concatenate([[1.0 / alpha0_var], (1.0 / (tau * lambda_[0 : m - 1])).flatten()]))
        Binv = prior_alpha_var_inv + tmp_ux.T @ tmpd_inv @ tmp_ux
        Binvb = tmp_ux.T @ tmpd_inv @ tmp_uy
        BinvU, BinvD_diag, _ = np.linalg.svd(Binv, full_matrices=True)
        ub = np.diag(1.0 / BinvD_diag) @ BinvU.T @ Binvb
        alpha_tmp = ub + (np.sqrt(1.0 / BinvD_diag)[:, np.newaxis] * np.random.randn(m, 1))
        alpha = BinvU @ alpha_tmp

        # Draw sigma (integrate out f)
        sigma_old = sigma
        sigma_new = sigma + rw_sigma * np.random.randn()

        sigma2_old = sigma_old**2
        tmp = y - x @ alpha
        tmpu = Ku
        tmpd_diag = s + sigma2_old * Kd_diag
        logdet_tmp1 = np.sum(np.log(tmpd_diag))
        tmp2 = tmpu.T @ tmp
        tmp3 = float(tmp2.T @ np.diag(1.0 / tmpd_diag) @ tmp2)
        loglike_old = -0.5 * logdet_tmp1 - 0.5 * tmp3
        logprior_old = -0.5 * sigma2_old / float(tau * lambda_[m - 1, 0])

        sigma2_new = sigma_new**2
        tmpd_diag = s + sigma2_new * Kd_diag
        logdet_tmp1 = np.sum(np.log(tmpd_diag))
        tmp3 = float(tmp2.T @ np.diag(1.0 / tmpd_diag) @ tmp2)
        loglike_new = -0.5 * logdet_tmp1 - 0.5 * tmp3
        logprior_new = -0.5 * sigma2_new / float(tau * lambda_[m - 1, 0])

        log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old
        if math.log(np.random.rand()) <= log_accpt_prob:
            sigma = sigma_new
            if drawi > burnin:
                count_sigma = count_sigma + 1
        else:
            sigma = sigma_old
        sigma2 = sigma**2

        accpt_prob = min(1.0, math.exp(log_accpt_prob))
        logrw_new = math.log(rw_sigma) + (accpt_prob - pstar) / (drawi * pstar * (1.0 - pstar))
        rw_sigma = math.exp(logrw_new)

        # Draw phi (integrate out f)
        phi_old = phi
        phi_new = phi + rw_phi * np.random.randn()

        Kcov_old = Ktmp ** (phi_old**2)
        Ku_old, Kd_old_diag, _ = np.linalg.svd(Kcov_old, full_matrices=True)
        tmpu = Ku_old
        tmpd_diag = s + sigma2 * Kd_old_diag
        logdet_tmp1 = np.sum(np.log(tmpd_diag))
        tmp2 = tmpu.T @ tmp
        tmp3 = float(tmp2.T @ np.diag(1.0 / tmpd_diag) @ tmp2)
        loglike_old = -0.5 * logdet_tmp1 - 0.5 * tmp3
        logprior_old = -0.5 * phi_old * phi_old / phi_lambda

        Kcov_new = Ktmp ** (phi_new**2)
        Ku_new, Kd_new_diag, _ = np.linalg.svd(Kcov_new, full_matrices=True)
        tmpu = Ku_new
        tmpd_diag = s + sigma2 * Kd_new_diag
        logdet_tmp1 = np.sum(np.log(tmpd_diag))
        tmp2 = tmpu.T @ tmp
        tmp3 = float(tmp2.T @ np.diag(1.0 / tmpd_diag) @ tmp2)
        loglike_new = -0.5 * logdet_tmp1 - 0.5 * tmp3
        logprior_new = -0.5 * phi_new * phi_new / phi_lambda

        log_accpt_prob = logprior_new + loglike_new - logprior_old - loglike_old
        if math.log(np.random.rand()) <= log_accpt_prob:
            phi = phi_new
            Ku = Ku_new
            Kd_diag = Kd_new_diag
            if drawi > burnin:
                count_phi = count_phi + 1
        else:
            phi = phi_old
            Ku = Ku_old
            Kd_diag = Kd_old_diag

        accpt_prob = min(1.0, math.exp(log_accpt_prob))
        logrw_new = math.log(rw_phi) + (accpt_prob - pstar) / (drawi * pstar * (1.0 - pstar))
        rw_phi = math.exp(logrw_new)

        # Draw f
        sigma_s = sigma / s
        sigma2_s = sigma2 / s
        tmp_diag = Kd_diag / (1.0 + sigma2_s * Kd_diag)
        Ku_times_b = sigma_s * (tmp_diag[:, np.newaxis] * (Ku.T @ (y - x @ alpha)))
        ftmp = Ku_times_b + (np.sqrt(tmp_diag)[:, np.newaxis] * np.random.randn(n, 1))
        f = Ku @ ftmp

        # ASIS step for sigma, f
        ff = sigma * (Ku.T @ f)
        sigma_sign = np.sign(sigma) if sigma != 0 else 1.0

        sigma2_p = 0.5 - 0.5 * n
        sigma2_a = 1.0 / float(tau * lambda_[m - 1, 0])
        sigma2_b = float(ff.T @ np.diag(1.0 / Kd_diag) @ ff)
        sigma2_asis = gigrnd(sigma2_p, sigma2_a, sigma2_b, 1)
        sigma = float(math.sqrt(sigma2_asis) * sigma_sign)

        f = (Ku @ ff) / sigma

        # Draw tau, tau0
        beta = np.vstack([alpha[1:m], [[sigma]]])
        beta2 = beta**2
        tau_a = (1.0 + m) / 2.0
        tau_b = 1.0 / tau0 + 0.5 * float(np.sum(beta2 / lambda_))
        tau = 1.0 / np.random.gamma(tau_a, 1.0 / tau_b)

        tau0_a = 1.0
        tau0_b = n2 + 1.0 / tau
        tau0 = 1.0 / np.random.gamma(tau0_a, 1.0 / tau0_b)

        # Draw lambda, lambda0
        lambda_a = 1.0
        lambda_b = 1.0 / lambda0 + 0.5 * beta2 / tau
        lambda_ = 1.0 / np.random.gamma(lambda_a, 1.0 / lambda_b)

        lambda0_a = 1.0
        lambda0_b = 1.0 + 1.0 / lambda_
        lambda0 = 1.0 / np.random.gamma(lambda0_a, 1.0 / lambda0_b)

        # Draw phi_lambda, phi_lambda0
        phi_lambda = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 / phi_lambda0 + 0.5 * phi * phi))
        phi_lambda0 = 1.0 / np.random.gamma(1.0, 1.0 / (1.0 + 1.0 / phi_lambda))

        # Collect draws
        if drawi > burnin:
            draw_idx = drawi - burnin - 1
            draws.alpha[draw_idx, :] = alpha.flatten()
            draws.sigma[draw_idx, 0] = sigma
            draws.tau[draw_idx, 0] = tau
            draws.tau0[draw_idx, 0] = tau0
            draws.lambda_[draw_idx, :] = lambda_.flatten()
            draws.lambda0[draw_idx, :] = lambda0.flatten()
            draws.phi_lambda[draw_idx, 0] = phi_lambda
            draws.phi_lambda0[draw_idx, 0] = phi_lambda0
            draws.s0[draw_idx, 0] = s0
            draws.s[draw_idx, 0] = s
            draws.phi[draw_idx, 0] = phi
            draws.f[draw_idx, :] = f.flatten()
            draws.yfit[draw_idx, :] = (x @ alpha + sigma * f).flatten()
            draws.ff[draw_idx, :] = (sigma * f).flatten()
            draws.xalpha[draw_idx, :] = (x @ alpha).flatten()
            draws.rw[draw_idx, :] = [rw_phi, rw_sigma]
            draws.count_phi = count_phi / ndraws
            draws.count_sigma = count_sigma / ndraws

        if round(drawi / 1000.0) == (drawi / 1000.0):
            print(f"{drawi} draws have been completed!")
            print(f"Elapsed time is {time.time() - tic_time:.6f} seconds.")
            print(" ")

    return draws
