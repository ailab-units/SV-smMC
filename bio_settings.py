# filepath, train_filename, val_filename, params_list, math_params_list

case_studies = [
    ["SIR", "SIR_DS_500samples_50obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], [r"$\beta$"]],
    ["SIR", "SIR_DS_700samples_50obs_gamma", "SIR_DS_1000samples_1000obs_gamma", ["gamma"], [r"$\gamma$"]],
    ["SIR", "SIR_DS_2500samples_50obs_betagamma", "SIR_DS_400samples_1000obs_betagamma", ["beta","gamma"], [r"$\gamma$", r"$\beta$"]],
    ["PrGeEx", "PrGeEx_DS_500samples_50obs_k2", "PrGeEx_DS_400samples_1000obs_k2", ["k2"], [r"$k_2$"]],
    ["PrGeEx", "PrGeEx_DS_2500samples_50obs_k2k7", "PrGeEx_DS_400samples_1000obs_k2k7", ["k2","k7"], [r"$k_7$", r"$k_2$"]],
    ["PRDeg", "PRDeg_DS_500samples_50obs_k1", "PRDeg_DS_1000samples_1000obs_k1", ["k1"], [r"$k_1$"]],
    ["PRDeg", "PRDeg_DS_2500samples_50obs_kprodkdeg", "PRDeg_DS_400samples_1000obs_kprodkdeg", ["kp", "kd"], [r"$k_d$", r"$k_p$"]],
    ["PRDeg", "PRDeg_DS_8000_latin_samples_20obs_k1k2k3", "PRDeg_DS_1000_latin_samples_1000obs_k1k2k3", ["k1","k2","k3"], [r"$k_1$",r"$k_2$",r"$k_3$"]],
    ["PRDeg", "PRDeg_DS_10000_latin_samples_20obs_k1k2k3k4", "PRDeg_DS_4096_latin_samples_1000obs_k1k2k3k4", ["k1","k2","k3","k4"], [r"$k_1$",r"$k_2$",r"$k_3$",r"$k_4$"]],
    ["PRDeg", "PRDeg_DS_1000000_latin_samples_20obs_kprodk1k2k3k4kdeg", "PRDeg_DS_4096_latin_samples_1000obs_kprodk1k2k3k4kdeg", ["kp","k1","k2","k3","k4","kd"], [r"$k_p$",r"$k_1$",r"$k_2$",r"$k_3$",r"$k_4$",r"$k_d$"]],

    ["Eco/", "Eco_DS_30000samples_20obs", "Eco_DS_600samples_1000obs",  plist_eco, plist_eco],
    ["MAPK/", "MAPK_DS_50000samples_10obs", "MAPK_DS_1000samples_1000obs", plist_mapk, plist_mapk]
]

data_path = 'data/'
models_path = 'out/models/'
plots_path = 'out/plots/'
