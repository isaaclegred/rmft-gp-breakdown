import pandas as pd
import numpy as np
from scipy import interpolate
import temperance as tmpy
import temperance.core.result as result
from temperance.core.result import EoSPosterior
from temperance.sampling.eos_prior import EoSPriorSet

import tqdm
import json
import temperance.plotting.corner as tmcorner 
import temperance.plotting.get_quantiles as get_quantiles
import temperance.plotting.envelope as envelope
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
envelope.get_defaults(mpl, fontsize=32)
import temperance.sampling.branched_interpolator as b_interp
import sys 
sys.path.append("..")
import weigh_rmf_draws

def plot_mr_single_eos(macro_path, central_pressure_column="central_pressurec2", pt_energy_density=None,
                       **kwargs):
    macro = pd.read_csv(macro_path)
    stable = np.where(
        np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0,
                       macro[central_pressure_column] < 10e17))[0][:-1]
    plt.plot(macro["R"][stable], macro["M"][stable], **kwargs)

def cgs_to_per_fm_3(rho):
    return rho / 2.8e14 * .16

def cgs_to_mev_per_fm_3(rho):
    return rho / 2.8e14 * .16 * 938.5
    
def plot_prho_single_eos(eos_path,   pt_energy_density=None,
                       **kwargs):
    eos = pd.read_csv(eos_path)
    plt.plot(cgs_to_per_fm_3(eos["baryon_density"]), cgs_to_mev_per_fm_3(eos["pressurec2"]), **kwargs)

def plot_mr(eos_post, eos_prior,  weight_columns, num_samples=500, central_pressure_column="central_pressurec2",     injected_eos = "./EoSs/macro-abht_qmcrmf2.csv", label=None, transition_pressure=None,label_color="red"):
    eos_samples = eos_post.sample(size=num_samples, weight_columns=weight_columns, replace=True)

    for eos_index in eos_post.samples["eos"][0:1000]:
        
        macro = pd.read_csv(eos_prior.get_macro_path(int(eos_index)))
        if transition_pressure is not None:
            rmft_informed = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, macro[central_pressure_column] < transition_pressure))[0][:]
            #close the gap between lines
            if len(rmft_informed) != 0:
                rmft_informed = np.append(rmft_informed, max(rmft_informed) + 1)
            model_agnostic = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, np.logical_and(2e15>macro[central_pressure_column], macro[central_pressure_column] > transition_pressure)))[0][:-1]
            plt.plot(macro["R"][rmft_informed], macro["M"][rmft_informed], color="black", alpha=.3, lw=0.7)
            plt.plot(macro["R"][model_agnostic], macro["M"][model_agnostic], color="darkgrey", alpha=.3, lw=0.7 )
        else:
            stable = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, macro[central_pressure_column] < 2.6e15))[0][:-1]
            plt.plot(macro["R"][stable], macro["M"][stable], color="black", alpha=.1, lw=0.7)
    for eos_index in eos_samples["eos"]:
        macro = pd.read_csv(eos_prior.get_macro_path(int(eos_index)))
        if transition_pressure is not None:
            rmft_informed = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, macro[central_pressure_column] < transition_pressure))[0][:]
            #close the gap between lines
            if len(rmft_informed) != 0:
                rmft_informed = np.append(rmft_informed, max(rmft_informed) + 1)
            model_agnostic = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, np.logical_and(2e15>macro[central_pressure_column], macro[central_pressure_column] > transition_pressure)))[0][:-1]
            plt.plot(macro["R"][rmft_informed], macro["M"][rmft_informed], color="orange", alpha=.3, lw=0.8)
            plt.plot(macro["R"][model_agnostic], macro["M"][model_agnostic], color="red", alpha=.3, lw=0.8)
        else:
            
            stable = np.where(np.logical_and(np.gradient(macro["M"], macro[central_pressure_column]) > 0, macro[central_pressure_column] < 2.6e15))[0][:-1]
    
            plt.plot(macro["R"][stable], macro["M"][stable], color="orange", alpha=.3, lw=0.7)


    plt.xlim(8, 16)
    plt.ylim(.2, 2.7)
    if label is None:
        label = eos_post.label
    
    plot_mr_single_eos(injected_eos, lw=2.4, alpha=1, color="cyan")

    plt.fill_between(np.linspace(15.85, 16.0, 10), 1.0, np.max(pd.read_csv(injected_eos)["M"]), color="black", alpha=.6)

    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    #plt.fill_between(np.linspace(15.85, 16.0, 10), 1.0, max(pd.read_csv(injection_eos)), color="black", alpha=.6)
    #plt.fill_between(np.linspace(8.0, 16.0, 10), 1.0, 2.08, color="black", alpha=.1)
    plt.xlabel(r"$R\ [\rm{km}]$", fontsize=24)
    plt.ylabel(r"$M\ [M_{\odot}]$", fontsize=24)
    plt.text(8.1, 2.5, f"posterior", color=label_color, fontsize=34)
    plt.text(8.1, 2.30, f"prior", color="black", fontsize=34)
    plt.title(f"{label}", color="black", fontsize=24)    # Injected EoS

   


    plt.savefig(f"./{eos_post.label}-m-r.pdf",bbox_inches="tight")

def get_macro_of_m(macro_filepath, m):
    macro = pd.read_csv(macro_filepath)
    interps =  b_interp.choose_macro_per_m(m, macro, {"R": lambda m : 2 * 1.477 * m, "Lambda":lambda m:np.zeros_like(m)}, 
                                           only_lambda=False)
    return np.concatenate([interps["R"], interps["Lambda"]])


def annotate_nuclear_density(y_level=1.01):
    plt.axvline(.16, color="k", lw=2.0)
    plt.text(.16, y_level, r"$\rho_{\rm nuc}$",{"fontsize":20} )
    plt.axvline(2*.16, color="k", lw=2.0)
    plt.text(2 * .16, y_level, r"$2\rho_{\rm nuc}$", {"fontsize":20})
    plt.axvline(4 * .16, color="k", lw=2.0)
    plt.text(4 * .16, y_level, r"$4\rho_{\rm nuc}$", {"fontsize":20})

def add_second_pressure_axis(ax, cgs_pressure_range):
    secax = ax.secondary_xaxis("top", functions=(lambda x: cgs_to_mev_per_fm_3(x), lambda x: x/cgs_to_mev_per_fm_3(1)))
    secax.set_xlabel(r"$p\ [\rm{MeV}/\rm{fm}^{3}]$", fontsize=18)
    ax.tick_params(top=False)
    # mev_pressure_min = cgs_to_mev_per_fm_3(cgs_pressure_range[0])
    # mev_pressure_max = cgs_to_mev_per_fm_3(cgs_pressure_range[1])
    # print("mev_pressure_min, mev_pressure_max are", mev_pressure_min, mev_pressure_max)
    # major_ticks = []
    # minor_ticks = []
    # scaling = 10**(np.round(np.floor(np.log10(mev_pressure_min))))
    # print("scaling is", scaling)
    # tracker = np.round(np.ceil(mev_pressure_min / scaling))*scaling
    # print("tracker is", tracker)
    # while (tracker < mev_pressure_max):
    #     print("tracker is", tracker)
    #     if np.isclose(np.log10(tracker) % 1,  0):
    #         major_ticks.append(tracker)
    #         tracker *= 2
    #     else:
    #         minor_ticks.append(tracker)
    #         tracker +=  scaling if major_ticks == [] else major_ticks[-1]

    # print("major_ticks",  major_ticks)
    # secax.xaxis.set_major_locator(FixedLocator(major_ticks))
    # secax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
    # secax.xaxis.set_minor_locator(FixedLocator(minor_ticks))
    return secax

def plot_p_rho(
        eos_post, eos_prior,  weight_columns, num_samples=500, central_pressure_column="central_pressurec2",
        injected_eos = "./EoSs/abht_qmcrmf2.csv", label=None, transition_pressure=None):
    """
    Make the p-rho spaghetti plot
    eos_post: EoSPosterior object
    eos_prior: EoSPriorSet object
    weight_columns: list of WeightColumn objects
    num_samples: number of samples to draw from the posterior
    central_pressure_column: name of the column in the macro file that contains the central pressure
    injected_eos: path to the injected EoS file
    label: label for the plot
    transition_pressure: the transition pressure for the EoS prior being used
    """
    # Sample the EoS posterior
    eos_samples = eos_post.sample(size=num_samples, weight_columns=weight_columns, replace=True)
    # Plot the samples from the prior
    for eos_index in eos_post.samples["eos"][0:1000]:
        # read in the EoS 
        eos = pd.read_csv(eos_prior.get_eos_path(int(eos_index)))
        if transition_pressure is not None:
            # split the EoS into parts below and above the rmft-agnostic transtion
            rmft_informed = np.where(eos["pressurec2"] < transition_pressure)[0]
            model_agnostic = np.where(eos["pressurec2"] > transition_pressure)[0]
            plt.plot(cgs_to_per_fm_3(eos["baryon_density"][rmft_informed]),
                     cgs_to_mev_per_fm_3(eos["pressurec2"][rmft_informed]), color="black", alpha=.3, lw=0.7)
            plt.plot(cgs_to_per_fm_3(eos["baryon_density"][model_agnostic]),
                     cgs_to_mev_per_fm_3(eos["pressurec2"][model_agnostic]), color="darkgrey", alpha=.3, lw=0.7 )
        else:
            # Do not split the EoS
            plt.plot(cgs_to_per_fm_3(eos["baryon_density"]),
                     cgs_to_mev_per_fm_3(eos["pressurec2"]), color="black", alpha=.1, lw=0.7)
    # Do the same for the posterior
    for eos_index in eos_samples["eos"]:
        eos = pd.read_csv(eos_prior.get_eos_path(int(eos_index)))
        rmft_informed = np.where(eos["pressurec2"] < transition_pressure)[0]
        model_agnostic = np.where(eos["pressurec2"] > transition_pressure)[0]
        plt.plot(cgs_to_per_fm_3(eos["baryon_density"][rmft_informed]),
                 cgs_to_mev_per_fm_3(eos["pressurec2"][rmft_informed]), color="coral", alpha=.3, lw=0.7)
        plt.plot(cgs_to_per_fm_3(eos["baryon_density"][model_agnostic]),
                 cgs_to_mev_per_fm_3(eos["pressurec2"][model_agnostic]), color="red", alpha=.3, lw=0.7 )
    else:
            
        plt.plot(cgs_to_per_fm_3(eos["baryon_density"]),
                 cgs_to_mev_per_fm_3(eos["pressurec2"]), color="red", alpha=.1, lw=0.7)

    plt.xlim(.16/2, .16*7)
    plt.ylim(5 * .16, 420 * .16 * 7)
    plt.xscale("log")
    plt.yscale("log")
    annotate_nuclear_density(y_level=.55)
    if label is None:
        label = eos_post.label
    
    #plt.yscale("log")
    plt.xlabel(r"$n\ [\rm{fm}^{-3}]$")
    plt.ylabel(r"$p\ [\rm{MeV}/\rm{fm}^3]$")
    plt.text(.09, 180, f"posterior", color="red", fontsize=24)
    plt.text(.09, 140, f"prior", color="black", fontsize=24)
    plt.title(f"{label}", color="black", fontsize=24)    # Injected EoS

    plot_prho_single_eos(injected_eos, lw=2.4, alpha=1, color="royalblue")

    #annotate_nuclear_density()

    plt.savefig(f"./{eos_post.label}-p-rho.pdf", bbox_inches="tight")
        

    

def check_containment(array, other_array):
    return [array[i] in other_array for i, elt in enumerate(array)]

if __name__ == "__main__":
    pt = False

    # The injection used to simulate data
    injection_tag = f"rmf-1109-000338-bps{'_1p7_1p4_0p8' if pt else '' }"
    injection_eos = f"./EoSs/{injection_tag}.csv"
    injection_macro  = f"./EoSs/macro-{injection_tag}.csv"
    
    # Which of the RMFT EoS sets used to build GPs
    rmf_eos_set = "1109"

    
    # Which events to use for the posterior distribution
    rmf_weight_columns = [
                result.WeightColumn(column_name, is_log=True) for column_name in
                [
                    *[f"logweight_xray_{i}" for i in [0, 1,  2, 3]],
                    *[f"logweight_gw_{j}" for j in [i for i in range(2)]],
                    *[f"logweight_radio_{k}" for k in range(2)]                ]   
            ]
    # Make the M-R and p-rho spaghetti plots
    def make_mr_and_prho_plots():
        astro_post = EoSPosterior.from_csv("../../EoSPopulation/Reweighting/EoS/collated_np_all_post.csv", label="astro")
        eos_set = h5py.File("../../EoSPopulation/Reweighting/EoS/gp_mrgagn.h5")
        eoss_used_in_inference = eos_set["id"]
        astro_post = EoSPosterior(astro_post.samples.loc[check_containment(
            astro_post.samples["eos"],  np.array(eoss_used_in_inference))], label="model-agnostic")
        astro_eos_prior = EoSPriorSet.get_default()

        
        rmf_eos_set = "1109"
        for key in ["1e11", "1e12", "3e12", "1e13", "3e13", "1e14"]:
            eos_set_identifier=f"maxpc2-{key}"
            eos_tag = f"{rmf_eos_set}_{eos_set_identifier}_mrgagn_01d000_00d010"
            rmf_dir  = f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{eos_tag}"
            rmf_eos_post = EoSPosterior.from_csv(
                f"./{injection_tag}_{eos_tag}_post.csv", label=injection_tag + "-" + eos_set_identifier + f"-{rmf_eos_set}")
            rmf_eos_prior = EoSPriorSet(
                eos_dir=rmf_dir, eos_column="eos", eos_per_dir=1000, macro_dir=rmf_dir,
                branches_data=None)
            
            transition_pressure = eval(key)
            exponent = int(np.log10(transition_pressure))
            plot_mr(rmf_eos_post, rmf_eos_prior, rmf_weight_columns, num_samples=200, injected_eos=injection_macro,
                    label=rf"$p_{{\rm t}}/c^2 =  {np.round(transition_pressure/10**exponent, 2)}\times 10^{{{exponent}}}\ \rm{{g}}/\rm{{cm}}^3$",
                    transition_pressure=eval(key))
            plt.clf()
            plot_p_rho(rmf_eos_post, rmf_eos_prior, rmf_weight_columns, num_samples=200, injected_eos=injection_eos,
                    label=rf"$p_{{\rm t}}/c^2 =  {np.round(transition_pressure/10**exponent, 2)}\times 10^{{{exponent}}}\ \rm{{g}}/\rm{{cm}}^3$",
                       transition_pressure=eval(key))
            plt.clf()
    
    #make_mr_and_prho_plots()    
    
    # Make the spaghetti plot with the RMFT EoS set as a prior
    def plot_rmf_draws_comparison():
        rmf_eos_set = "1109"
        eos_tag = f"rmf_{rmf_eos_set}"
        rmf_eos_post_path = f"rmf-{rmf_eos_set}-000338-bps{'_1p7_1p4_0p8' if pt else ''}_{eos_tag}_post.csv"
        rmf_eos_post = EoSPosterior.from_csv(rmf_eos_post_path, label=injection_tag + "-"+eos_tag)
        rmf_dir  = f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{eos_tag}"
        rmf_eos_prior = EoSPriorSet(
                  eos_dir=rmf_dir, eos_column="eos", eos_per_dir=1000, macro_dir=rmf_dir,
                  branches_data=None)
        # Use the RMFT EoS the whole way up, transition pressure is just a large number
        plot_mr(rmf_eos_post, rmf_eos_prior, rmf_weight_columns, num_samples=200, injected_eos=injection_macro,
                    label="RMFT",
                    transition_pressure=None, label_color="orange")   
        plt.clf()
        plot_p_rho(rmf_eos_post, rmf_eos_prior, rmf_weight_columns, num_samples=200, injected_eos=injection_eos,
                    label="RMFT",
                       transition_pressure=None)
        plt.clf()
    #plot_rmf_draws_comparison()
    
    # prior_p_of_rho_quantiles = get_quantiles.get_p_of_rho_quantiles(astro_post, weight_columns=[])
    # prior_p_of_rho_quantiles.to_csv("Quantiles/prior_p_of_rho_quantiles.csv", index=False)
    # prior_r_of_m_quantiles = get_quantiles.get_r_of_m_quantiles(astro_post, weight_columns=[])
    # prior_r_of_m_quantiles.to_csv("Quantiles/prior_r_of_m_quantiles.csv", index=False)
    # prior_cs2_of_rho_quantiles = get_quantiles.get_cs2_of_rho_quantiles(astro_post, weight_columns=[])
    # prior_cs2_of_rho_quantiles.to_csv("Quantiles/prior_cs2_of_rho_quantiles.csv", index=False)
        
    # Make the Bayes factor plot
    def estimate_evidence(all_weight_columns=rmf_weight_columns, add_rmf_set=False, condition_on_pulsars=True):
        injected_eos = pd.read_csv(f"./EoSs/{injection_tag}.csv")       
        injected_pt_pressure = interpolate.griddata(injected_eos["energy_densityc2"], injected_eos["pressurec2"], [1.7 * 2.8e14]) if pt else None
        #injected_pt_pressure = interpolate.griddata(injected   _eos["energy_densityc2"], injected_eos["pressurec2"], [1.5 * 2.8e14])
        results = {f"{tag}" :  EoSPosterior.from_csv(
            f"{injection_tag}_1109_maxpc2-{tag}_mrgagn_01d000_00d010_post.csv",
            label=tag) for tag in ["1e11", "1e12", "3e12", "1e13", "3e13", "1e14"]}
        evidences = []
        plt.figure(figsize=(6.472, 4))
        for tag in results.keys():
            evidence_tuple = results[tag].estimate_evidence(
                weight_columns_to_use=all_weight_columns,prior_weight_columns=[
                    result.WeightColumn(column_name, is_log=True) for column_name in ([f"logweight_radio_{k}" for k in range(2)] if condition_on_pulsars else [])] )
            print("transition at: ", tag + ", evidence = ",evidence_tuple[0], r"±",
                np.sqrt(evidence_tuple[1]))
            print("divergence vs prior", results[tag].compute_kl_divergence_vs_prior(
                weight_columns_to_use=all_weight_columns))
            print("neff", results[tag].compute_neff_kish(
                weight_columns_to_use=all_weight_columns) )
            evidences.append(evidence_tuple)
        normalization = evidences[0][0]
        norm_error = evidences[0][1]
        eb = plt.errorbar([1e11, 1e12, 3e12, 1e13, 3e13,  1e14],
                    [evidence_tuple[0]/normalization for evidence_tuple in evidences],
                    yerr= [np.sqrt(evidence_tuple[1] + evidence_tuple[0]/normalization * norm_error)/normalization for evidence_tuple in evidences],
                    linewidth=2.0, color="black", capsize=7, capthick=2, elinewidth=1.0)
        
        if add_rmf_set:
            rmf_post = EoSPosterior.from_csv(f"{injection_tag}_rmf_1109_post.csv")   
            rmf_evidence_tuple = rmf_post.estimate_evidence(
                weight_columns_to_use=all_weight_columns,
                prior_weight_columns=[
                    result.WeightColumn(column_name, is_log=True) for column_name in ([f"logweight_radio_{k}" for k in range(2)] if condition_on_pulsars else [])])
            print("transition at: ", "rmf" + ", evidence = ",rmf_evidence_tuple[0], r"±",
                np.sqrt(rmf_evidence_tuple[1]))
            value = rmf_evidence_tuple[0]/normalization
            error = np.sqrt(rmf_evidence_tuple[1] + rmf_evidence_tuple[0]/normalization * norm_error)/normalization
            plt.fill_between(np.linspace(9e10, 1.1e14, 20), value-error, value+error, alpha=.3, color="k")
        eb[-1][0].set_linestyle('--')
        plt.xlim(9e10, 1.07e14)
        plt.ylim(min([evidence_tuple[0] for evidence_tuple in evidences])/normalization * .9,
                 max([evidence_tuple[0] for evidence_tuple in evidences])/normalization * (5 if value > 1 else 1.8))
        plt.axhline(1, color="black", linestyle="--")
        # plt.fill_between(np.linspace(9e10, 1.1e14, 20), np.ones(20), 200 * np.ones(20), color="deepskyblue", alpha=.4)
        # plt.fill_between(np.linspace(9e10, 1.1e14, 20), np.ones(20), 1e-12* np.ones(20), color="red", alpha=.4)
        plt.xlabel(r"transition pressure $[\rm{g}/\rm{cm^3}]$")
        plt.ylabel(r"$\mathcal B^{\rm{mod.}}_{\rm{agn.}}$")
        plt.xscale("log")
        plt.yscale("log")
        if injected_pt_pressure is not None:
            plt.axvline(injected_pt_pressure, color="red", linestyle="--", linewidth=2)
        if max([evidence_tuple[0] for evidence_tuple in evidences])/normalization * 1.1 < 5:
            plt.yticks(np.arange(2, 12 ,2), labels=[str(i) for i in np.arange(2, 12, 2)])
        ax = plt.gca()
        secax = add_second_pressure_axis(ax, (9e10, 1.07e14))
        plt.savefig(f"{injection_tag}_evidence{'' if condition_on_pulsars else'_no_pulsar_condition'}.pdf", bbox_inches="tight")        
    estimate_evidence(add_rmf_set=True,condition_on_pulsars=True)
    # Make the Bayes factor plot with gravitational waves alone
    def estimate_gw_catalog_evidences(control="3e13", trial="1e14"):
        """
        Compute the Bayes factor between the two GP priors given by "control"
        and "trial" for the simulated gravitational wave catalog.
        """
        plt.clf()
        plt.figure(figsize=(6.472, 4))
        gw_results =  {f"{tag}" :  EoSPosterior.from_csv(
        f"{injection_tag}_1109_maxpc2-{tag}_mrgagn_01d000_00d010_post.csv",
        label=tag) for tag in [control, trial]}
        bfs  = []
        bf_errors = []
        plt.clf()
        # One of the BNS runs didn't finish, so we just ignore it
        events_to_use = [0, 1, 3, 4, 5, 6, 7, 8, 9]
        rmft_event_to_injection = dict(zip(events_to_use,[0,1,3,4,5,6,7,8,9]))
        pt_event_to_injection = dict(zip(events_to_use, [3,4,6,7,8,9,11,12,13]))
        with open("./GW/rmf-1109-000338-bps/found_injections-rmf-1109-000338-bps.json") as rmft_inj_json:
            rmft_injections = json.load(rmft_inj_json)
        with open("./GW/rmf-1109-000338-bps_1p7_1p4_0p8/found_injections-rmf-1109-000338-bps_1p7_1p4_0p8.json") as pt_inj_json:
            pt_injections = json.load(pt_inj_json)
        injections = rmft_injections if not pt else pt_injections
        snrs =  []
        for catalog_size in range(9):
            local_evidences = {}
            prior_weight_columns=[
                result.WeightColumn(column_name, is_log=True) for column_name in [f"logweight_radio_{k}" for k in [0, 1]]]
            #prior_weight_columns = []
            mapper = rmft_event_to_injection if not pt else pt_event_to_injection
            if catalog_size != 0:
                snr = np.sqrt(injections["injections"]["optimal_snr_squared"][mapper[events_to_use[catalog_size-1]]])
            else:
                snr = 0
            snrs.append(int(np.round(snr)))
            
            weight_columns_to_use=[*prior_weight_columns,
                                   *[result.WeightColumn(f"logweight_gw_{j}", is_log=True) for j in events_to_use[:catalog_size]]]
            for tag in [control, trial]:
                evidence_tuple = gw_results[tag].estimate_evidence(
                    weight_columns_to_use=weight_columns_to_use,
                    prior_weight_columns=prior_weight_columns)
                local_evidences[tag] = evidence_tuple
                print("transition at: ", tag + ", evidence = ",evidence_tuple[0], r"±",
                      np.sqrt(evidence_tuple[1]))
                

            normalization = local_evidences[control][0]
            norm_error = local_evidences[control][1]
            bfs.append(local_evidences[trial][0]/normalization)
            bf_errors.append(np.sqrt(local_evidences[trial][1] + local_evidences[trial][0]/normalization * norm_error)/normalization)

        #plt.scatter([catalog_size], [1.0], c=snr/50, cmap="plasma", s=100, edgecolor="black", linewidth=2.0)
        eb = plt.errorbar(np.arange(9), bfs, yerr=bf_errors, linewidth=2.0, color="black", capsize=7, capthick=2, elinewidth=1.0)
        eb[-1][0].set_linestyle('--')
        plt.axhline(1, color="black", linestyle="--")
        # plt.fill_between(np.linspace(0, 10, 20), np.ones(20), 100 * np.ones(20), color="deepskyblue", alpha=.4)
        # plt.fill_between(np.linspace(0, 10, 20), np.ones(20), 1e-12* np.ones(20), color="red", alpha=.4)
        plt.xlabel("GW catalog size")
        plt.ylabel(r"$\mathcal B^{\rm{mod.}}_{\rm{agn.}}$")
        ax = plt.gca()
        secax= ax.secondary_xaxis('top', functions=(lambda x: x, lambda x: x))
        ax.set_xticks(np.arange(0, 9))
        secax.set_xticks(np.arange(0, 9))
        secax.set_xticklabels(snrs)
        plt.yscale("log")
        plt.xlim(0, 8)
        if pt:
            plt.ylim(1e-4, 3.3)
        else:
             plt.yticks([1, 2, 3, 4], labels=["1", "2", "3", "4"])
        plt.text(6.5, 3e-1 if pt else 3, "RMFT+PT" if pt else "RMFT", fontsize=24, color="black", ha="center")
        #if max([evidence_tuple[0] for evidence_tuple in evidences])/normalization * 1.1 < 5:
        #    plt.yticks([1, 2, 3, 4, 5], labels=["1", "2", "3", "4", "5"])

        plt.savefig(f"{injection_tag}_evidence_gws_only.pdf", bbox_inches="tight")
    estimate_gw_catalog_evidences()

    # Make the corner plot of the symmetry parameters
    def plot_symmetry_params():
        rmf_eos_set = "1109"

        rmf_eos_post = EoSPosterior.from_csv(f"{injection_tag}_rmf_1109_post.csv", label="rmf")

        gp_results_comparison_tags = ["1e13", "3e13", "1e14"]
        injected_eos = pd.read_csv(f"./EoSs/{injection_tag}.csv")       
        injected_pt_pressure = interpolate.griddata(injected_eos["energy_densityc2"], injected_eos["pressurec2"], [1.7 * 2.8e14]) if pt else None
        #injected_pt_pressure = interpolate.griddata(injected   _eos["energy_densityc2"], injected_eos["pressurec2"], [1.5 * 2.8e14])
        results = {f"{tag}" :  EoSPosterior.from_csv(
            f"{injection_tag}_1109_maxpc2-{tag}_mrgagn_01d000_00d010_post.csv",
            label=tag) for tag in ["1e11", "1e12", "3e12", "1e13", "3e13", "1e14"]}
        priors = {tag: EoSPriorSet(
            eos_dir=f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{rmf_eos_set}_maxpc2-{tag}_mrgagn_01d000_00d010", 
            eos_column="eos", eos_per_dir=1000, 
            macro_dir=f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{rmf_eos_set}_{tag}_mrgagn_01d000_00d010",
            branches_data=None) for tag in gp_results_comparison_tags}
        priors["rmf"] = f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/rmf_{rmf_eos_set}"
        results["rmf"] = rmf_eos_post
        colors = {"rmf" : "deepskyblue", "1e13": "maroon", "3e13":"red",
        "1e14":"darkorange"}
        label_1e13 =r"$p_{\rm t}/c^2 = 10^{13}\, \rm{g}/\rm{cm}^3$"
        label_3e13 = r"$p_{\rm t}/c^2 = 3\times 10^{13}\, \rm{g}/\rm{cm}^3$"
        label_1e14 = r"$p_{\rm t}/c^2 = 10^{14}\, \rm{g}/\rm{cm}^3$"
        labels={ "1e14":label_1e14, "rmf" : "RMFT", "1e13":label_1e13 }
        prior_linestyles= {"rmf":"--", "1e13":":", "3e13":".", "1e14":"--"}
        weight_columns_to_use=rmf_weight_columns
        params = {tag : pd.read_csv(priors[tag].eos_dir + "/symmetry_data_experimental.csv").rename(columns={"S0":"symmetry_energy", "L":"slope_symmetry_energy", "Ksym" : "second_derivative_symmetry_energy"}) for tag in gp_results_comparison_tags}
        params["rmf"] = pd.read_csv("../gp-rmf-inference/conditioned-priors/many-draws/1109/rmf_1109/saturation_props_len_1109.csv").rename(columns={"gpr_index":"eos"})
        #params["rmf"]["eos"] = np.arange(0, len(params["rmf"]["n_sat"]))
        
        weigh_rmf_draws.plot_multiple_params_direct(labels=labels, weights=results, 
                weight_columns_to_use=weight_columns_to_use,
                params=params, outtag=injection_tag, colors=colors, true_params=params["rmf"],
                prior_linestyles=prior_linestyles)
    plot_symmetry_params()
    # Make diagnostic corner plots of the priors on macroscopic quantities 
    # for a variety of transition pressures
    def plot_priors_diagnostic(eos_posts=[
        EoSPosterior.from_csv(f"{injection_tag}_{rmf_eos_set}_maxpc2-1e13_mrgagn_01d000_00d010_post.csv", label=r"$p_{\rm t} = 1 \times 10^{13}$"),
        EoSPosterior.from_csv(f"{injection_tag}_{rmf_eos_set}_maxpc2-3e13_mrgagn_01d000_00d010_post.csv", label=r"$p_{\rm t} = 3 \times 10^{13}$"),
        EoSPosterior.from_csv(f"{injection_tag}_{rmf_eos_set}_maxpc2-1e12_mrgagn_01d000_00d010_post.csv", label=r"$p_{\rm t} = 10^{12}$")],
                              eos_priors = [
                                  EoSPriorSet(eos_dir=rmf_dir, eos_column="eos", eos_per_dir=1000, macro_dir=rmf_dir, branches_data=None)
                                  for rmf_dir in [f"../gp-rmf-inference/conditioned-priors/many-draws/{rmf_eos_set}/{eos_set}" for eos_set in [
                                          f"{rmf_eos_set}_maxpc2-1e13_mrgagn_01d000_00d010",
                                          f"{rmf_eos_set}_maxpc2-3e13_mrgagn_01d000_00d010",
                                          f"{rmf_eos_set}_maxpc2-1e12_mrgagn_01d000_00d010"]]],
                   weight_columns =[result.WeightColumn(column_name, is_log=True) for column_name in ["logweight_radio_0", "logweight_radio_1"]] , num_samples=8000, central_pressure_column="central_pressurec2",
                   injected_eos =None, colors={
                       r"$p_{\rm t} = 1 \times 10^{13}$":"royalblue",
                       r"$p_{\rm t} = 3 \times 10^{13}$":"maroon",
                       r"$p_{\rm t} = 10^{12}$":"grey"}, **kwargs):
        """
        Make a corner plot of the posterior distributions of the macroscopic quantities
        eos_posts is a list of EoSPosterior objects, each of which has a different transition pressures
        eos_priors is a list of EoSPriorSet objects, each of which has a different transition pressure (corresponding to the eos_posts)
        weight_columns is a list of WeightColumn objects to use for the posterior samples
        num_samples is the number of samples to draw from each posterior
        central_pressure_column is the name of the column in the macro file that contains the central pressure
        injected_eos is the path to the injected EoS file
        colors is a dictionary mapping the labels of the eos_posts to colors
        kwargs are passed to the corner plot function
        """
        plt.clf()
        radii_to_use = ["1.4", "1.8"]
        column_names = [f"R(M={radius})" for radius in radii_to_use]
        column_names += [f"Lambda(M={radius})" for radius in radii_to_use]
        plottable_posts = {}

        for eos_post_index, eos_post in enumerate(eos_posts):
            eos_samples = eos_post.sample(size=num_samples, weight_columns=weight_columns, replace=True)
            post_samples = pd.DataFrame({"eos": eos_samples["eos"]})
            print(len(np.unique(post_samples["eos"])))
            
            
            for column_name in column_names:
                post_samples[column_name] = np.zeros_like(eos_samples["eos"], dtype=np.float64)
            for i, eos_index  in enumerate(tqdm.tqdm(eos_samples["eos"])):
                post_samples.loc[i, column_names] = get_macro_of_m(eos_priors[eos_post_index].get_macro_path(eos_index),
                                        np.array([eval(radius) for radius in radii_to_use]))
            prior_samples = pd.DataFrame({"eos":eos_post.sample(size=num_samples, weight_columns=[], replace=True)["eos"] })
            for column_name in column_names:
                prior_samples[column_name] = np.zeros_like(prior_samples["eos"], dtype=np.float64)
            for i, eos_index  in enumerate(tqdm.tqdm(prior_samples["eos"])):
                prior_samples.loc[i, column_names] = (get_macro_of_m(eos_priors[eos_post_index].get_macro_path(eos_index),
                                              np.array([eval(radius) for radius in radii_to_use])))
            post_samples = pd.merge(post_samples, eos_post.samples[["eos", "Mmax"]], on="eos")
            prior_samples = pd.merge(prior_samples, eos_post.samples[["eos", "Mmax"]], on="eos")
            plottable_posts[eos_post.label + "(astro. data)"] = EoSPosterior(post_samples.copy(deep=True), label=eos_post.label)
            plottable_posts[eos_post.label + "(prior)"] = EoSPosterior(prior_samples.copy(deep=True), label=eos_post.label + "(prior)")
        
        plottable_columns = {}
        plottable_columns["R1p4"] = tmcorner.PlottableColumn(
            name="R(M=1.4)",
            label=tmcorner.get_default_label("R(M=1.4)"),
            plot_range=(10.8, 13.4),
            bandwidth=.2)
        plottable_columns["R1p8"] = tmcorner.PlottableColumn(
            name="R(M=1.8)",
            label=tmcorner.get_default_label("R(M=1.8)"),
            plot_range=(10.8, 13.4),
            bandwidth=.2)

        # plottable_columns["Lambda1p4"] = tmcorner.PlottableColumn(
        #     name="Lambda(M=1.4)",
        #     label=r"$\Lambda_{1.4}$",
        #     plot_range=(1.1e2,7.3e2),
        #     bandwidth=40)
        # plottable_columns["Lambda1p8"] = tmcorner.PlottableColumn(
        #     name="Lambda(M=1.8)",
        #     label=r"$\Lambda_{1.8}$",
        #     plot_range=(10, 190),
        #     bandwidth=50)
        plottable_columns["Mmax"] = tmcorner.PlottableColumn(
            name="Mmax",
            label=r"$M_{\rm TOV}\ [M_{\odot}]$",
            plot_range=(1.8, 2.7),
            bandwidth=.05)

        plottable_samples = {}
        for eos_post in eos_posts:
            print("eos_post samples are", plottable_posts[eos_post.label + "(astro. data)"].samples)
            print("eos_post prior samples are", plottable_posts[eos_post.label + "(prior)"].samples)
            
            plt.clf()
            print("color is", colors[eos_post.label])
            plottable_samples[eos_post.label + "(astro. data)"] = tmcorner.PlottableEoSSamples(
                label=eos_post.label + ("psr"),
                posterior=plottable_posts[eos_post.label + "(astro. data)"],
                weight_columns_to_use=[],
                color=colors[eos_post.label],
                additional_properties=eos_post.samples)

            # plottable_samples[eos_post.label + "(prior)"] = tmcorner.PlottableEoSSamples(   
            #     label=eos_post.label + "(prior)",
            #     posterior=plottable_posts[eos_post.label + "(prior)"],
            #     weight_columns_to_use=[],
            #     color=colors[eos_post.label],
            #     linestyle="--",
            #     additional_properties=eos_post.samples)
        print("about to print all plottable samples", [ps.posterior.samples for ps  in plottable_samples.values()])
        tmcorner.corner_eos(plottable_samples.values(),
                            use_universality=True,
                            columns_to_plot=plottable_columns.values(), **kwargs)

        plt.savefig(f"diagnostic-comparison-mtov_vs_r_corner-radio.pdf", bbox_inches="tight")
    #plot_priors_diagnostic()
    
    


