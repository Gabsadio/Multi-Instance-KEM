from doom import dsDOOM, mmtDOOM
from math import log2
from matplotlib import pyplot as plt
import istarmap
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
from tqdm import tqdm


if __name__ == "__main__":
    plt.style.use("bmh")
    bike_parameters = [(12323, 134), (24659, 199), (40973, 264)]
    for j, (r, t) in enumerate(bike_parameters):
        print(f"BIKE-{2*j+1}...")

        # Single-instance:
        print("Single instance:\nDS: ", end="")
        time, mem, (p, l, p_, kl_) = dsDOOM(2 * r, r - 1, t, 1 * r)
        print(time, "\nMMT: ", end="")
        timemmt, memmmt, (mm, pmmt, lmmt, b, rkl, p_mmt, rp_) = mmtDOOM(
            2 * r, r - 1, t, 1 * r
        )
        print(timemmt)

        instanceRange = range(int(3 / 4 * max(time, timemmt)))
        dsruntimes = dict((i, time) for i in instanceRange)
        dsspeedups = dict((i, float(0)) for i in instanceRange)
        dsmemory = dict((i, mem) for i in instanceRange)
        dssyndromesUsed = dict((i, i + log2(r)) for i in instanceRange)
        dsparameters_p = dict((i, p) for i in instanceRange)
        dsparameters_l = dict((i, l) for i in instanceRange)
        dsparameters_p_ = dict((i, p_) for i in instanceRange)
        dsparameters_kl_ = dict((i, kl_) for i in instanceRange)
        mmtruntimes = dict((i, timemmt) for i in instanceRange)
        mmtspeedups = dict((i, float(0)) for i in instanceRange)
        mmtmemory = dict((i, memmmt) for i in instanceRange)
        mmtsyndromesUsed = dict((i, mm) for i in instanceRange)
        mmtparameters_p = dict((i, pmmt) for i in instanceRange)
        mmtparameters_l = dict((i, lmmt) for i in instanceRange)
        mmtparameters_b = dict((i, b) for i in instanceRange)
        mmtparameters_rkl = dict((i, rkl) for i in instanceRange)
        mmtparameters_p_ = dict((i, p_mmt) for i in instanceRange)
        mmtparameters_rp_ = dict((i, rp_) for i in instanceRange)

        # Multi-instance:
        print("Multi-Instance:")
        with Pool(processes=cpu_count()) as pool:
            input = [(2 * r, r - 1, t, (1 << i) * r) for i in instanceRange[1:]]
            print("DS:")
            dsresults = list(tqdm(pool.istarmap(dsDOOM, input), total=len(input)))
            print("MMT:")
            mmtresults = list(tqdm(pool.istarmap(mmtDOOM, input), total=len(input)))

        for i in instanceRange[1:]:
            (
                dsruntimes[i],
                dsmemory[i],
                (
                    dsparameters_p[i],
                    dsparameters_l[i],
                    dsparameters_p_[i],
                    dsparameters_kl_[i],
                ),
            ) = dsresults[i - 1]
            if dsruntimes[i] > dsruntimes[i - 1]:
                dsruntimes[i] = dsruntimes[i - 1]
                dsmemory[i] = dsmemory[i - 1]
                dssyndromesUsed[i] = dssyndromesUsed[i - 1]
                dsparameters_p[i] = dsparameters_p[i - 1]
                dsparameters_l[i] = dsparameters_l[i - 1]
                dsparameters_p_[i] = dsparameters_p_[i - 1]
                dsparameters_kl_[i] = dsparameters_kl_[i - 1]
            dsspeedups[i] = (dsruntimes[0] - dsruntimes[i]) / i

            (
                mmtruntimes[i],
                mmtmemory[i],
                (
                    mmtsyndromesUsed[i],
                    mmtparameters_p[i],
                    mmtparameters_l[i],
                    mmtparameters_b[i],
                    mmtparameters_rkl[i],
                    mmtparameters_p_[i],
                    mmtparameters_rp_[i],
                ),
            ) = mmtresults[i - 1]
            if mmtruntimes[i] > mmtruntimes[i - 1]:
                mmtruntimes[i] = mmtruntimes[i - 1]
                mmtmemory[i] = mmtmemory[i - 1]
                mmtsyndromesUsed[i] = mmtsyndromesUsed[i - 1]
                mmtparameters_p[i] = mmtparameters_p[i - 1]
                mmtparameters_l[i] = mmtparameters_l[i - 1]
                mmtparameters_b[i] = mmtparameters_b[i - 1]
                mmtparameters_rkl[i] = mmtparameters_rkl[i - 1]
                mmtparameters_p_[i] = mmtparameters_p_[i - 1]
                mmtparameters_rp_[i] = mmtparameters_rp_[i - 1]
            mmtspeedups[i] = (mmtruntimes[0] - mmtruntimes[i]) / i

        # Save the results in a file:
        with open(f"Out/BIKEDOOM/BIKE-{2*j+1}.txt", "w") as file:
            file.write("DS-DOOM\n")
            file.write("-------\n")
            file.write(
                "log2 # Instances      :\t"
                + "\t".join(f"{i:20d}" for i in instanceRange)
                + "\n"
            )
            file.write(
                "log2 # syndromes used :\t"
                + "\t".join(f"{entry:20.16f}" for entry in dssyndromesUsed.values())
                + "\n"
            )
            file.write(
                "Runtime               :\t"
                + "\t".join(f"{entry:20.16f}" for entry in dsruntimes.values())
                + "\n"
            )
            file.write(
                "Speedups              :\t"
                + "\t".join(f"{entry:20.16f}" for entry in dsspeedups.values())
                + "\n"
            )
            file.write(
                "Memory                :\t"
                + "\t".join(f"{entry:20.16f}" for entry in dsmemory.values())
                + "\n"
            )
            file.write(
                "p                     :\t"
                + "\t".join(f"{entry:20d}" for entry in dsparameters_p.values())
                + "\n"
            )
            file.write(
                "l                     :\t"
                + "\t".join(f"{entry:20d}" for entry in dsparameters_l.values())
                + "\n"
            )
            file.write(
                "p_                    :\t"
                + "\t".join(f"{entry:20d}" for entry in dsparameters_p_.values())
                + "\n"
            )
            file.write(
                "rkl                   :\t"
                + "\t".join(f"{entry:20d}" for entry in dsparameters_kl_.values())
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
                + "\t".join(f"{entry:20.16f}" for entry in mmtsyndromesUsed.values())
                + "\n"
            )
            file.write(
                "Runtime               :\t"
                + "\t".join(f"{entry:20.16f}" for entry in mmtruntimes.values())
                + "\n"
            )
            file.write(
                "Speedups              :\t"
                + "\t".join(f"{entry:20.16f}" for entry in mmtspeedups.values())
                + "\n"
            )
            file.write(
                "Memory                :\t"
                + "\t".join(f"{entry:20.16f}" for entry in mmtmemory.values())
                + "\n"
            )
            file.write(
                "p                     :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_p.values())
                + "\n"
            )
            file.write(
                "l                     :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_l.values())
                + "\n"
            )
            file.write(
                "b                     :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_b.values())
                + "\n"
            )
            file.write(
                "rkl                   :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_rkl.values())
                + "\n"
            )
            file.write(
                "p_                    :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_p_.values())
                + "\n"
            )
            file.write(
                "rp_                   :\t"
                + "\t".join(f"{entry:20d}" for entry in mmtparameters_rp_.values())
                + "\n"
            )

        # Plot the results:
        plt.figure()
        plt.plot(dsruntimes.keys(), dsruntimes.values(), label="DS")
        plt.plot(mmtruntimes.keys(), mmtruntimes.values(), label="MMT")
        plt.xlabel("$\\log_2$ number of encapsulations")
        plt.ylabel("bit security")
        plt.legend()
        plt.savefig(f"Figures/BIKEDOOM/BIKE-{2*j+1}/Runtimes")

        plt.figure()
        plt.plot(list(dsspeedups.keys())[1:], list(dsspeedups.values())[1:], label="DS")
        plt.plot(
            list(mmtspeedups.keys())[1:], list(mmtspeedups.values())[1:], label="MMT"
        )
        plt.xlabel("$\\log_2$ number of encapsulations")
        plt.ylabel("$\\log_M$ speedups")
        plt.legend()
        plt.savefig(f"Figures/BIKEDOOM/BIKE-{2*j+1}/Speedups")

        plt.figure()
        plt.plot(dsmemory.keys(), dsmemory.values(), label="DS")
        plt.plot(mmtmemory.keys(), mmtmemory.values(), label="MMT")
        plt.xlabel("$\\log_2$ number of encapsulations")
        plt.ylabel("$\\log_2$ memory")
        plt.legend()
        plt.savefig(f"Figures/BIKEDOOM/BIKE-{2*j+1}/Memory")
