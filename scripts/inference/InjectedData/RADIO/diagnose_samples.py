import pandas as pd
import numpy as np

def get_stats(samples_path, index):
    samples = pd.read_csv(samples_path)
    m_mean= np.mean(samples["M"])
    m_spread = 1.645 * np.std(samples["M"])
    # print out stats for latex
    print(f"M-{index} injected is",f"{m_mean:.2f}^{{+{m_spread:.2f}}}_{{-{m_spread:.2f}}}")
if __name__ == "__main__":
    for injection_eos  in ["rmf-1109-000338-bps_1p7_1p4_0p8", "rmf-1109-000338-bps"]:
        print("stats for injection eos", injection_eos)
        for samples_index in range(2):
            get_stats(f"{injection_eos}/{injection_eos}_fake_psr_samples_{samples_index}.csv", samples_index)
