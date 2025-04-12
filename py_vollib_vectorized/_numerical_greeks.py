import numpy as np

from py_vollib_vectorized.util.jit_helper import maybe_jit
from ._model_calls import black, black_scholes, black_scholes_merton

dS = 1e-6

#### BLACK

@maybe_jit()
def numerical_delta_black(flags, Fs, Ks, ts, rs, sigmas):
    deltas = []
    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        if t == 0.0:
            if F == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif F > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (black(F*(1+dS), K, sigma, t, flag) - black(F*(1-dS), K, sigma, t, flag)) / (
                    2*F*dS)
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black(flags, Fs, Ks, ts, rs, sigmas):
    thetas = []
    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        if t <= 1. / 365.:
            theta = black(F, K, sigma, 0.00001, flag) - black(F, K, sigma, t, flag)
        else:
            theta = black(F, K, sigma, t - 1./365., flag) - black(F, K, sigma, t, flag)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black(flags, Fs, Ks, ts, rs, sigmas):
    vegas = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        vega = (black(F, K, sigma + 0.01, t, flag) - black(F, K, sigma - 0.01, t, flag)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black(flags, Fs, Ks, ts, rs, sigmas):
    rhos = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        black(F, K, sigma. t, flag)
        rho = (black(flag, F, K, t, r + 0.01, sigma) - black(flag, F, K, t, r - 0.01, sigma)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black(flags, Fs, Ks, ts, rs, sigmas):
    gammas = []

    for flag, F, K, t, r, sigma in zip(flags, Fs, Ks, ts, rs, sigmas):
        if t == 0:
            gamma = np.inf if F == K else 0.0
        else:
            gamma = (black(flag, F*(1+dS), K, t, r, sigma) - 2. * black(flag, F, K, t, r, sigma) + \
                     black(flag, F*(1-dS), K, t, r, sigma)) / (F*dS) ** 2.

        gammas.append(gamma)
    return gammas

@maybe_jit()
def numerical_vanna_black(flags, Ss, Ks, ts, rs, sigmas, bs):
    vanna = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        price_upS_upSigma = black(flag, S*(1+dS), K, t, r, sigma)
        price_upS_downSigma = black(flag, S*(1+dS), K, t, r, sigma - dSigma)
        price_downS_upSigma = black(flag, S*(1-dS), K, t, r, sigma + dSigma)
        price_downS_downSigma = black(flag, S*(1-dS), K, t, r, sigma - dSigma)

        value = (price_upS_upSigma - price_upS_downSigma - price_downS_upSigma + price_downS_downSigma) / (4 * dS * S * dSigma)
        vanna.append(value)
    return vanna

@maybe_jit()
def numerical_charm_black(flags, Ss, Ks, ts, rs, sigmas, bs):
    charm = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            charm.append(0.0)
            continue

        delta_t_minus = (black(flag, S*(1+dS), K, t - 1./365., r, sigma) -
                         black(flag, S*(1-dS), K, t - 1./365., r, sigma)) / (2 * dS * S)

        delta_t_now = (black(flag, S*(1+dS), K, t, r, sigma) -
                       black(flag, S*(1-dS), K, t, r, sigma)) / (2 * dS * S)

        value = (delta_t_minus - delta_t_now) / (1. / 365.)
        charm.append(value)
    return charm

#### BLACK SCHOLES

@maybe_jit()
def numerical_delta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    deltas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0.0:
            if S == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif S > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (black_scholes(flag, S*(1+dS), K, t, r, sigma) - black_scholes(flag, S*(1-dS), K, t, r, sigma)) / (
                    2 * S*dS)
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    thetas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            theta = black_scholes(flag, S, K, 0.00001, r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        else:
            theta = black_scholes(flag, S, K, t - 1. / 365., r, sigma) - black_scholes(flag, S, K, t, r, sigma)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    vegas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        vega = (black_scholes(flag, S, K, t, r, sigma + 0.01) - black_scholes(flag, S, K, t, r, sigma - 0.01)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    rhos = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        rho = (black_scholes(flag, S, K, t, r + 0.01, sigma) - black_scholes(flag, S, K, t, r - 0.01, sigma)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    gammas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes(flag, S*(1+dS), K, t, r, sigma) - 2. * black_scholes(flag, S, K, t, r, sigma) + \
                     black_scholes(flag, S*(1-dS), K, t, r, sigma)) / (S*dS) ** 2.

        gammas.append(gamma)
    return gammas


@maybe_jit()
def numerical_vanna_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    vanna = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        price_upS_upSigma = black_scholes(flag, S*(1+dS), K, t, r, sigma + dSigma)
        price_upS_downSigma = black_scholes(flag, S*(1+dS), K, t, r, sigma - dSigma)
        price_downS_upSigma = black_scholes(flag, S*(1-dS), K, t, r, sigma + dSigma)
        price_downS_downSigma = black_scholes(flag, S*(1-dS), K, t, r, sigma - dSigma)

        value = (price_upS_upSigma - price_upS_downSigma - price_downS_upSigma + price_downS_downSigma) / (4 * dS * S * dSigma)
        vanna.append(value)
    return vanna


@maybe_jit()
def numerical_charm_black_scholes(flags, Ss, Ks, ts, rs, sigmas, bs):
    charm = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            charm.append(0.0)
            continue

        delta_t_minus = (black_scholes(flag, S*(1+dS), K, t - 1./365., r, sigma) -
                         black_scholes(flag, S*(1-dS), K, t - 1./365., r, sigma)) / (2 * dS * S)

        delta_t_now = (black_scholes(flag, S*(1+dS), K, t, r, sigma) -
                       black_scholes(flag, S*(1-dS), K, t, r, sigma)) / (2 * dS * S)

        value = (delta_t_minus - delta_t_now) / (1. / 365.)
        charm.append(value)
    return charm


### BLACK SCHOLES MERTON

@maybe_jit()
def numerical_delta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    deltas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0.0:
            if S == K:
                if flag > 0:  # call option
                    delta = 0.5
                if flag < 0:  # put option
                    delta = -0.5
            elif S > K:
                if flag > 0:  # call option
                    delta = 1.0
                if flag < 0:  # put option
                    delta = 0.0
            else:
                if flag > 0:  # call option
                    delta = 0.0
                if flag < 0:  # put option
                    delta = -1.0
        else:
            delta = (black_scholes_merton(flag, S*(1+dS), K, t, r, sigma, r - b) - black_scholes_merton(flag, S*(1-dS), K,
                                                                                                      t, r,
                                                                                                      sigma,
                                                                                                      r - b)) / (
                            2 * dS*S)
        deltas.append(delta)
    return deltas


@maybe_jit()
def numerical_theta_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    thetas = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            theta = black_scholes_merton(flag, S, K, 0.00001, r, sigma, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                      sigma,
                                                                                                      r - b)
        else:
            theta = black_scholes_merton(flag, S, K, t - 1. / 365., r, sigma, r - b) - black_scholes_merton(flag, S, K,
                                                                                                            t,
                                                                                                            r, sigma,
                                                                                                            r - b)
        thetas.append(theta)
    return thetas


@maybe_jit()
def numerical_vega_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    vegas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        vega = (black_scholes_merton(flag, S, K, t, r, sigma + 0.01, r - b) - black_scholes_merton(flag, S, K, t, r,
                                                                                                   sigma - 0.01,
                                                                                                   r - b)) / 2.
        vegas.append(vega)
    return vegas


@maybe_jit()
def numerical_rho_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    rhos = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        rho = (black_scholes_merton(flag, S, K, t, r + 0.01, sigma, r - b + 0.01) - black_scholes_merton(flag, S, K, t,
                                                                                                         r - 0.01,
                                                                                                         sigma,
                                                                                                         r - b - 0.01)) / 2.
        rhos.append(rho)

    return rhos


@maybe_jit()
def numerical_gamma_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    gammas = []

    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t == 0:
            gamma = np.inf if S == K else 0.0
        else:
            gamma = (black_scholes_merton(flag, S*(1+dS), K, t, r, sigma, r - b) - 2. * black_scholes_merton(flag, S, K,
                                                                                                           t, r,
                                                                                                           sigma,
                                                                                                           r - b) + \
                     black_scholes_merton(flag, S*(1-dS), K, t, r, sigma, r - b)) / (S*dS) ** 2.

        gammas.append(gamma)
    return gammas


@maybe_jit()
def numerical_vanna_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    vanna = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        price_upS_upSigma = black_scholes_merton(flag, S*(1+dS), K, t, r, sigma + dSigma, r - b)
        price_upS_downSigma = black_scholes_merton(flag, S*(1+dS), K, t, r, sigma - dSigma, r - b)
        price_downS_upSigma = black_scholes_merton(flag, S*(1-dS), K, t, r, sigma + dSigma, r - b)
        price_downS_downSigma = black_scholes_merton(flag, S*(1-dS), K, t, r, sigma - dSigma, r - b)
        
        value = (price_upS_upSigma - price_upS_downSigma - price_downS_upSigma + price_downS_downSigma) / (4 * dS * S * dSigma)
        vanna.append(value)
    return vanna

@maybe_jit()
def numerical_charm_black_scholes_merton(flags, Ss, Ks, ts, rs, sigmas, bs):
    charm = []
    for flag, S, K, t, r, sigma, b in zip(flags, Ss, Ks, ts, rs, sigmas, bs):
        if t <= 1. / 365.:
            charm.append(0.0)  # Too close to expiry — can be unstable or meaningless
            continue

        delta_t_minus = (black_scholes_merton(flag, S*(1+dS), K, t - 1./365., r, sigma, r - b) - 
                         black_scholes_merton(flag, S*(1-dS), K, t - 1./365., r, sigma, r - b)) / (2 * dS * S)

        delta_t_now = (black_scholes_merton(flag, S*(1+dS), K, t, r, sigma, r - b) -
                       black_scholes_merton(flag, S*(1-dS), K, t, r, sigma, r - b)) / (2 * dS * S)

        value = (delta_t_minus - delta_t_now) / (1. / 365.)
        charm.append(value)
    return charm
