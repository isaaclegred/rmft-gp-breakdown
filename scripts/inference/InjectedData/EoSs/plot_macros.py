import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def plot_mr(macro_file, name=None):
    data = pd.read_csv(macro_file)
    plt.plot(data["R"], data["M"], label=name)
def plot_mlambda(macro_file, name=None):
    data = pd.read_csv(macro_file)
    plt.plot(data["Lambda"], data["M"], label=name)
def plot_prho(eos_file, name=None):
    data = pd.read_csv(eos_file)
    plt.plot(data["baryon_density"], data["pressurec2"])
if __name__ == "__main__":
    comptag="rmf_vs_1p7_vs_2p0"
    eos_list= ["rmf-1109-000338-bps", "rmf-1109-000338-bps_1p7_1p4_0p8", "rmf-1109-000338-bps_2p0_1p3_0p7"]
    for file_tag in eos_list:
        plot_mr(f"macro-{file_tag}.csv", name=file_tag)
    plt.xlabel(r"$R\ [\rm{km}]$")
    plt.ylabel(r"$M\ [M_{\odot}]$")
    plt.xlim(7, 14)
    plt.ylim(.1, 2.5)
    plt.legend()
    plt.savefig(f"compare_mr_{comptag}.pdf")

    plt.clf()
    for file_tag in eos_list :
        plot_mlambda(f"macro-{file_tag}.csv", name=file_tag)
    plt.xlabel(r"$\Lambda$")
    plt.ylabel(r"$M\ [M_{\odot}]$")
    plt.xlim(5, 10000)
    plt.xscale("log")
    plt.ylim(.7, 2.5)
    plt.legend()
    plt.savefig(f"compare_mlambda_{comptag}.pdf")

    for file_tag in eos_list :
        plot_prho(f"{file_tag}.csv", name=file_tag)
    plt.xlabel(r"$\rho$")
    plt.ylabel(r"$p$")
    plt.xlim(1e10, 3e15)
    plt.ylim(1e8, 4e15)
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.savefig(f"compare_rhop_{comptag}.pdf")
