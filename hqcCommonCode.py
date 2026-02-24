from doom import dsDOOM, mmtDOOM
from math import log2
from matplotlib import pyplot as plt
from tqdm import tqdm
from multiprocessing.pool import Pool
from multiprocessing import cpu_count


if __name__ == "__main__":
    plt.style.use("bmh")
    hqc_parameters = [(17669, 66), (35851, 100), (57637, 131)]
    for j, (n, w) in enumerate(hqc_parameters):
        print(f"HQC-{2*j+1} ...")

        # Single-instance:
        print("Single-instance:\nDS: ", end="")
        timeds, memds, (pds, lds, p_ds, kl_ds) = dsDOOM(2 * n, n - 2, 2 * w, n)
        print(timeds, "\nMMT: ", end="")
        timemmt, memmmt, (mmmmt, pmmt, lmmt, bmmt, rkl, p_mmt, rp_) = mmtDOOM(
            2 * n, n - 2, 2 * w, n
        )
        print(timemmt)

        instanceRange = range(int(3 / 4 * max(timeds, timemmt) - log2(n)))
        DSruntimes = dict((i, timeds) for i in instanceRange)
        DSspeedups = dict((i, float(0)) for i in instanceRange)
        DSmemory = dict((i, memds) for i in instanceRange)
        DSsyndromesUsed = dict((i, i + log2(n)) for i in instanceRange)
        DSparameters_p = dict((i, pds) for i in instanceRange)
        DSparameters_l = dict((i, lds) for i in instanceRange)
        DSparameters_p_ = dict((i, p_ds) for i in instanceRange)
        DSparameters_kl_ = dict((i, kl_ds) for i in instanceRange)
        MMTruntimes = dict((i, timemmt) for i in instanceRange)
        MMTspeedups = dict((i, float(0)) for i in instanceRange)
        MMTmemory = dict((i, memmmt) for i in instanceRange)
        MMTsyndromesUsed = dict((i, mmmmt) for i in instanceRange)
        MMTparameters_p = dict((i, pmmt) for i in instanceRange)
        MMTparameters_l = dict((i, lmmt) for i in instanceRange)
        MMTparameters_b = dict((i, bmmt) for i in instanceRange)
        MMTparameters_rkl = dict((i, rkl) for i in instanceRange)
        MMTparameters_p_ = dict((i, p_mmt) for i in instanceRange)
        MMTparameters_rp_ = dict((i, rp_) for i in instanceRange)

        # Multi-instance:
        print("Multi-instance:")
        with Pool(processes=min(64, cpu_count())) as pool:
            input = [(2 * n, n - 2, 2 * w, n * (1 << i)) for i in instanceRange[1:]]
            print("DS:")
            DSresults = list(tqdm(pool.istarmap(dsDOOM, input), total=len(input)))
            print("MMT:")
            MMTresults = list(tqdm(pool.istarmap(mmtDOOM, input), total=len(input)))

        for i in instanceRange[1:]:
            (
                DSruntimes[i],
                DSmemory[i],
                (
                    DSparameters_p[i],
                    DSparameters_l[i],
                    DSparameters_p_[i],
                    DSparameters_kl_[i],
                ),
            ) = DSresults[i - 1]
            if DSruntimes[i] > DSruntimes[i - 1]:
                DSruntimes[i] = DSruntimes[i - 1]
                DSmemory[i] = DSmemory[i - 1]
                DSsyndromesUsed[i] = DSsyndromesUsed[i - 1]
                DSparameters_p[i] = DSparameters_p[i - 1]
                DSparameters_l[i] = DSparameters_l[i - 1]
                DSparameters_p_[i] = DSparameters_p_[i - 1]
                DSparameters_kl_[i] = DSparameters_kl_[i - 1]
            DSspeedups[i] = (DSruntimes[0] - DSruntimes[i]) / i

            (
                MMTruntimes[i],
                MMTmemory[i],
                (
                    MMTsyndromesUsed[i],
                    MMTparameters_p[i],
                    MMTparameters_l[i],
                    MMTparameters_b[i],
                    MMTparameters_rkl[i],
                    MMTparameters_p_[i],
                    MMTparameters_rp_[i],
                ),
            ) = MMTresults[i - 1]
            if MMTruntimes[i] > MMTruntimes[i - 1]:
                MMTruntimes[i] = MMTruntimes[i - 1]
                MMTmemory[i] = MMTmemory[i - 1]
                MMTsyndromesUsed[i] = MMTsyndromesUsed[i - 1]
                MMTparameters_p[i] = MMTparameters_p[i - 1]
                MMTparameters_l[i] = MMTparameters_l[i - 1]
                MMTparameters_b[i] = MMTparameters_b[i - 1]
                MMTparameters_rkl[i] = MMTparameters_rkl[i - 1]
                MMTparameters_p_[i] = MMTparameters_p_[i - 1]
                MMTparameters_rp_[i] = MMTparameters_rp_[i - 1]
            MMTspeedups[i] = (MMTruntimes[0] - MMTruntimes[i]) / i

        # Save the results in a file:
        with open(f"Out/HQCCommonCode/HQC-{2*j+1}.txt", "w") as file:
            file.write("DS-DOOM\n")
            file.write("-------\n")
            file.write(
                "log2 # Instances      :\t"
                + "\t".join(f"{i:20d}" for i in instanceRange)
                + "\n"
            )
            file.write(
                "log2 # syndromes used :\t"
                + "\t".join(f"{entry:20.16f}" for entry in DSsyndromesUsed.values())
                + "\n"
            )
            file.write(
                "Runtime               :\t"
                + "\t".join(f"{entry:20.16f}" for entry in DSruntimes.values())
                + "\n"
            )
            file.write(
                "Speedups              :\t"
                + "\t".join(f"{entry:20.16f}" for entry in DSspeedups.values())
                + "\n"
            )
            file.write(
                "Memory                :\t"
                + "\t".join(f"{entry:20.16f}" for entry in DSmemory.values())
                + "\n"
            )
            file.write(
                "p                     :\t"
                + "\t".join(f"{entry:20d}" for entry in DSparameters_p.values())
                + "\n"
            )
            file.write(
                "l                     :\t"
                + "\t".join(f"{entry:20d}" for entry in DSparameters_l.values())
                + "\n"
            )
            file.write(
                "p_                    :\t"
                + "\t".join(f"{entry:20d}" for entry in DSparameters_p_.values())
                + "\n"
            )
            file.write(
                "rkl                   :\t"
                + "\t".join(f"{entry:20d}" for entry in DSparameters_kl_.values())
                + "\n"
            )

            file.write("\nMMT-DOOM\n")
            file.write("--------\n")
            file.write(
                "log2 # Instances      :\t"
                + "\t".join(f"{i:20d}" for i in instanceRange)
                + "\n"
            )
            file.write(
                "log2 # syndromes used :\t"
                + "\t".join(f"{entry:20.16f}" for entry in MMTsyndromesUsed.values())
                + "\n"
            )
            file.write(
                "Runtime               :\t"
                + "\t".join(f"{entry:20.16f}" for entry in MMTruntimes.values())
                + "\n"
            )
            file.write(
                "Speedups              :\t"
                + "\t".join(f"{entry:20.16f}" for entry in MMTspeedups.values())
                + "\n"
            )
            file.write(
                "Memory                :\t"
                + "\t".join(f"{entry:20.16f}" for entry in MMTmemory.values())
                + "\n"
            )
            file.write(
                "p                     :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_p.values())
                + "\n"
            )
            file.write(
                "l                     :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_l.values())
                + "\n"
            )
            file.write(
                "b                     :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_b.values())
                + "\n"
            )
            file.write(
                "rkl                   :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_rkl.values())
                + "\n"
            )
            file.write(
                "p_                    :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_p_.values())
                + "\n"
            )
            file.write(
                "rp_                   :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_rp_.values())
                + "\n"
            )

        # Plot the results:
        plt.figure()
        plt.plot(DSruntimes.keys(), DSruntimes.values(), label="DS-DOOM")
        plt.plot(MMTruntimes.keys(), MMTruntimes.values(), label="MMT-DOOM")
        plt.legend()
        plt.xlabel("$\\log_2$ number of instances")
        plt.ylabel("bit security")
        plt.savefig(f"Figures/HQCCommonCode/HQC-{2*j+1}/Runtimes")

        plt.figure()
        plt.plot(
            list(DSspeedups.keys())[1:], list(DSspeedups.values())[1:], label="DS-DOOM"
        )
        plt.plot(
            list(MMTspeedups.keys())[1:],
            list(MMTspeedups.values())[1:],
            label="MMT-DOOM",
        )
        plt.legend()
        plt.xlabel("$M$: $\\log_2$ number of instances")
        plt.ylabel("$\\log_M$ speedups")
        plt.savefig(f"Figures/HQCCommonCode/HQC-{2*j+1}/Speedups")

        plt.figure()
        plt.plot(DSmemory.keys(), DSmemory.values(), label="DS-DOOM")
        plt.plot(MMTmemory.keys(), MMTmemory.values(), label="MMT-DOOM")
        plt.legend()
        plt.xlabel("$\\log_2$ number of instances")
        plt.ylabel("$\\log_2$ memory")
        plt.savefig(f"Figures/HQCCommonCode/HQC-{2*j+1}/Memory")
