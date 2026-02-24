from math import log2, comb, floor
from scipy.optimize import root_scalar
from scipy.special import binom
from numpy import seterr


def dsDOOM(n, k, w, M: int):
    seterr(over="raise")
    nw = log2(comb(n, w))
    m = log2(M)
    em = m + max(0, nw - n + k)  # expected number of solutions
    best_time = float("inf")
    mem = None
    params = None, None, None, None
    for p in range(w + 1):
        try:
            l_best = root_scalar(
                lambda x: x - (log2(M * binom(k + x, p))) / 2,
                bracket=(max(0, p - k), n - k - w + p),
            ).root

            l = round(l_best)
            if comb(k + l, p) < M:
                continue  # Balancing not possible

            # Cost of partial Gaussian elimination:
            gauss = log2(max(1, (n - k - l) * (n - k) * (n + M)))

            # Distribution factor to balance initial lists:
            r = root_scalar(
                lambda x: log2(binom((1 - x) * (k + l), (1 - x) * p))
                - log2(M * binom(x * (k + l), x * p)),
                bracket=(0, 1 / 2),
            ).root

            # Work with the nearest integers:
            p_ = round(r * p)
            kl_ = round(r * (k + l))
            if kl_ < p_ or k + l - kl_ < p - p_:
                continue

            right = comb(kl_, p_)
            left = comb(k + l - kl_, p - p_)

            # log2 probability of good permutation for fixedd solution:
            alpha = log2(comb(n - k - l, w - p) * left * right) - nw

            # List sizes:
            L1 = log2(left)
            L2 = log2(right * M)
            L = log2(left * right * M) - l

            # Costs:
            time = -min(0, em + alpha) + max(gauss, L1, L2, L)
            if time < best_time:
                best_time = time
                mem = max(
                    log2((n - k) * (n + M)),  # Parity-Check matrix + syndromes
                    min(L1, L2) + log2(max(1, l)),  # Hashmap for join
                )
                params = p, l, p_, kl_
        except:
            continue

    return best_time, mem, params


def mmtDOOM(n, k, w, M: int):
    seterr(over="raise")
    nw = log2(comb(n, w))
    best_time = float("inf")
    mem = None
    params = (None, None, None, None, None, None, None)
    for p in range(w + 1):
        for l in range(max(0, p - k), n - k - w + p + 1):
            try:
                ## Distribution factors:
                # If L4 only syndromes, i.e. r = 0:
                p_ = root_scalar(
                    lambda x: log2(binom((k + l) / 2, (p - x) / 2))
                    - log2(binom((k + l) / 2, x / 2)) * 2,
                    bracket=(0, p / 2),
                ).root

                bound = (
                    comb(round((k + l) / 2), round(p / 2) - round(p_ / 2))
                    * comb(round((k + l) / 2), round(p_ / 2))
                ) ** (2 / 3)
                if M <= bound:
                    # Balancing possible
                    MM = M
                    p_ = root_scalar(
                        lambda x: log2(binom((k + l) / 2, (p - x) / 2))
                        - log2(MM * binom(k + l, x)) / 2,
                        bracket=(0, p / 2),
                    ).root

                    r = root_scalar(
                        lambda x: log2(
                            binom((k + l) / 2, p_ / 2)
                            * binom((1 / 2 - x) * (k + l), (1 / 2 - x) * p_)
                        )
                        - log2(MM * binom(x * (k + l), x * p_)),
                        bracket=(0, 1 / 2),
                    ).root
                else:
                    # Balancing not fully possible:
                    MM = floor(bound)
                    r = 0

                mm = log2(MM)
                em = mm + max(0, nw - n + k)  # expected number of solutions

                # Runtime for initial partial gaussian elimination:
                gauss = log2(max(1, (n - k - l) * (n - k) * (n + MM)))

                # probability of good permutation for single instance:
                klhalf = round((k + l) / 2)
                phalf = round(p / 2)
                rkl = round(r * (k + l))
                rp = round(r * p)
                dkl = k + l - klhalf - rkl
                dp = p - phalf - rp
                if klhalf < phalf or rkl < rp or dkl < dp:
                    continue
                alpha = (
                    log2(
                        comb(n - k - l, w - p)
                        * comb(klhalf, phalf)
                        * comb(rkl, rp)
                        * comb(dkl, dp)
                    )
                    - nw
                )

                # Number of representations per well-permuted instance:
                p_half = round(p_ / 2)
                rp_ = round(r * p_)
                dp_ = round(p_) - p_half - rp_
                if phalf < p_half or rp < rp_ or dp < dp_:
                    continue
                reps = log2(comb(phalf, p_half) * comb(rp, rp_) * comb(dp, dp_))

                # Number of bits to match on level 1:
                b = floor(min(l, max(0, mm + alpha) + reps - 2))
                if b < 1:
                    continue

                # List sizes
                ls1 = comb(klhalf, phalf - p_half)
                ls2 = comb(dkl, dp - dp_) * comb(rkl, rp - rp_)
                ls3 = comb(klhalf, p_half) * comb(dkl, dp_)
                ls4 = comb(rkl, rp_)
                L1 = log2(ls1)
                L2 = log2(ls2)
                L3 = log2(ls3)
                L4 = log2(ls4 * MM)

                L12 = log2(ls1 * ls2) - b
                L34 = log2(ls3 * ls4 * MM) - b

                L = log2(ls1 * ls2 * ls3 * ls4 * MM) - l - b

                # Runtime:
                time = -min(0, em + alpha) + max(gauss, L1, L2, L3, L4, L12, L34, L)
                if time < best_time:
                    best_time = time
                    mem = max(
                        log2((n - k) * (n + MM)),  # parity-check matrix + syndromes
                        min(
                            max(max(min(L1, L2), L12), min(L3, L4)),
                            max(max(min(L3, L4), L34), min(L1, L2)),
                        )
                        + log2(max(1, l)),  # Hash-Joins
                    )
                    params = mm, p, l, b, rkl, round(p_), rp_

            except:
                continue

    return best_time, mem, params
