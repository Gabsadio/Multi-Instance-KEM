from doom import dsDOOM, mmtDOOM
from matplotlib import pyplot as plt
import istarmap
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from tqdm import tqdm


if __name__ == "__main__":
    plt.style.use("bmh")
    mceliece_parameters = [
        # First two not part of ISO spec
        (3488, 12, 64),
        (4608, 13, 96),
        (6688, 13, 128),
        (6960, 13, 119),
        (8192, 13, 128),
    ]
    for n, m, t in mceliece_parameters:
        print(f"mceliece-{n}{t} ...")
        k = n - m * t

        # Single-instance:
        print("Single-instance:\nDS: ", end="")
        timeds, memds, (pds, lds, p_ds, kl_ds) = dsDOOM(n, k - 1, t, 1)
        print(timeds, "\nMMT: ", end="")
        timemmt, memmmt, (mm, pmmt, lmmt, b, rkl, p_mmt, rp_) = mmtDOOM(n, k - 1, t, 1)
        print(timemmt)

        instanceRange = range(int(3 / 4 * max(timeds, timemmt)))
        DSruntimes = dict((i, timeds) for i in instanceRange)
        DSspeedups = dict((i, float(0)) for i in instanceRange)
        DSmemory = dict((i, memds) for i in instanceRange)
        DSsyndromesUsed = dict((i, i) for i in instanceRange)
        DSparameters_p = dict((i, pds) for i in instanceRange)
        DSparameters_l = dict((i, lds) for i in instanceRange)
        DSparameters_p1 = dict((i, p_ds) for i in instanceRange)
        DSparameters_l1 = dict((i, kl_ds) for i in instanceRange)
        MMTruntimes = dict((i, timemmt) for i in instanceRange)
        MMTspeedups = dict((i, float(0)) for i in instanceRange)
        MMTmemory = dict((i, memmmt) for i in instanceRange)
        MMTsyndromesUsed = dict((i, mm) for i in instanceRange)
        MMTparameters_p = dict((i, pmmt) for i in instanceRange)
        MMTparameters_l = dict((i, lmmt) for i in instanceRange)
        MMTparameters_b = dict((i, b) for i in instanceRange)
        MMTparameters_l2 = dict((i, rkl) for i in instanceRange)
        MMTparameters_p1 = dict((i, p_mmt) for i in instanceRange)
        MMTparameters_p12 = dict((i, rp_) for i in instanceRange)

        # Multi-instance:
        print("Multi-Instance:")
        with Pool(processes=cpu_count()) as pool:
            input = [(n, k - 1, t, 1 << i) for i in instanceRange[1:]]
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
                    DSparameters_p1[i],
                    DSparameters_l1[i],
                ),
            ) = DSresults[i - 1]
            if DSruntimes[i] > DSruntimes[i - 1]:
                DSruntimes[i] = DSruntimes[i - 1]
                DSmemory[i] = DSmemory[i - 1]
                DSsyndromesUsed[i] = DSsyndromesUsed[i - 1]
                DSparameters_p[i] = DSparameters_p[i - 1]
                DSparameters_l[i] = DSparameters_l[i - 1]
                DSparameters_p1[i] = DSparameters_p1[i - 1]
                DSparameters_l1[i] = DSparameters_l1[i - 1]
            DSspeedups[i] = (DSruntimes[0] - DSruntimes[i]) / i

            (
                MMTruntimes[i],
                MMTmemory[i],
                (
                    MMTsyndromesUsed[i],
                    MMTparameters_p[i],
                    MMTparameters_l[i],
                    MMTparameters_b[i],
                    MMTparameters_l2[i],
                    MMTparameters_p1[i],
                    MMTparameters_p12[i],
                ),
            ) = MMTresults[i - 1]
            if MMTruntimes[i] > MMTruntimes[i - 1]:
                MMTruntimes[i] = MMTruntimes[i - 1]
                MMTmemory[i] = MMTmemory[i - 1]
                MMTsyndromesUsed[i] = MMTsyndromesUsed[i - 1]
                MMTparameters_p[i] = MMTparameters_p[i - 1]
                MMTparameters_l[i] = MMTparameters_l[i - 1]
                MMTparameters_b[i] = MMTparameters_b[i - 1]
                MMTparameters_l2[i] = MMTparameters_l2[i - 1]
                MMTparameters_p1[i] = MMTparameters_p1[i - 1]
                MMTparameters_p12[i] = MMTparameters_p12[i - 1]
            MMTspeedups[i] = (MMTruntimes[0] - MMTruntimes[i]) / i

        # Save the results in a file:
        with open(f"Out/McElieceDOOM/{n}{t}.txt", "w") as file:
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
                + "\t".join(f"{entry:20d}" for entry in DSparameters_p1.values())
                + "\n"
            )
            file.write(
                "kl_                   :\t"
                + "\t".join(f"{entry:20d}" for entry in DSparameters_l1.values())
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
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_l2.values())
                + "\n"
            )
            file.write(
                "p_                    :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_p1.values())
                + "\n"
            )
            file.write(
                "rp_                   :\t"
                + "\t".join(f"{entry:20d}" for entry in MMTparameters_p12.values())
                + "\n"
            )

        # Plot the results:
        plt.figure()
        plt.plot(DSruntimes.keys(), DSruntimes.values(), label="DS-DOOM")
        plt.plot(MMTruntimes.keys(), MMTruntimes.values(), label="MMT-DOOM")
        plt.legend()
        plt.xlabel("$\\log_2$ number of instances")
        plt.ylabel("bit security")
        plt.savefig(f"Figures/McElieceDOOM/{n}{t}/Runtimes")

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
        plt.savefig(f"Figures/McElieceDOOM/{n}{t}/Speedups")

        plt.figure()
        plt.plot(DSmemory.keys(), DSmemory.values(), label="DS-DOOM")
        plt.plot(MMTmemory.keys(), MMTmemory.values(), label="MMT-DOOM")
        plt.legend()
        plt.xlabel("$\\log_2$ number of instances")
        plt.ylabel("$\\log_2$ memory")
        plt.savefig(f"Figures/McElieceDOOM/{n}{t}/Memory")
