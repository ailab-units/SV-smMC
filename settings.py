# filepath, train_filename, val_filename, params_list, math_params_list

plist2 = ["P0","P1"]
plist3 = ["P0","P1","P2"]
plist4 = ["P0","P1","P2","P3"]
plist8 = ["P0","P1","P2","P3","P4","P5","P6","P7"]
plist12 = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11"]

plist16 = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P15"]
plist20 = ["P0","P1","P2","P3","P4","P5","P6","P7","P8","P9","P10","P11","P12","P13","P14","P15","P16","P17","P18","P19"]


stat_guar_case_studies = [
    ["SIR", "SIR_DS_500samples_10obs_beta", "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 10],
    ["SIR", "SIR_DS_500samples_25obs_beta", "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 25],
    ["SIR", "SIR_DS_500samples_50obs_beta",  "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 50],
    ["SIR", "SIR_DS_500samples_100obs_beta", "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 100],
    ["SIR", "SIR_DS_500samples_200obs_beta", "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 200],
    ["SIR", "SIR_DS_500samples_500obs_beta",  "SIR_DS_200samples_500obs_beta", "SIR_DS_1000samples_1000obs_beta", ["beta"], 500],
    ]


ep_case_studies = [
    ["Dim2/", "NCM0_DS_10000samples_10obs", "NCM0_DS_200samples_1000obs", plist2, plist2],
    ["Dim2/", "NCM1_DS_10000samples_10obs", "NCM1_DS_200samples_1000obs", plist2, plist2],
    ["Dim2/", "NCM2_DS_10000samples_10obs", "NCM2_DS_200samples_1000obs", plist2, plist2],
    
    ["Dim3/", "NCM0_DS_15000samples_10obs", "NCM0_DS_300samples_1000obs", plist3, plist3],
    ["Dim3/", "NCM1_DS_15000samples_10obs", "NCM1_DS_300samples_1000obs", plist3, plist3],
    ["Dim3/", "NCM2_DS_15000samples_10obs", "NCM2_D2_300samples_1000obs", plist3, plist3],
    
    ["Dim4/", "NCM0_DS_20000samples_10obs", "NCM0_DS_400samples_1000obs", plist4, plist4],
    ["Dim4/", "NCM1_DS_20000samples_10obs", "NCM1_DS_400samples_1000obs", plist4, plist4],
    ["Dim4/", "NCM2_DS_20000samples_10obs", "NCM2_DS_400samples_1000obs", plist4, plist4],

    ]


case_studies = [

    ["Dim2/", "NCM0_DS_10000samples_10obs", "NCM0_DS_200samples_1000obs", plist2, plist2], 
    ["Dim2/", "NCM1_DS_10000samples_10obs", "NCM1_DS_200samples_1000obs", plist2, plist2], 
    ["Dim2/", "NCM2_DS_10000samples_10obs", "NCM2_DS_200samples_1000obs", plist2, plist2], 
    
    ["Dim3/", "NCM0_DS_15000samples_10obs", "NCM0_DS_300samples_1000obs", plist3, plist3], 
    ["Dim3/", "NCM1_DS_15000samples_10obs", "NCM1_DS_300samples_1000obs", plist3, plist3], 
    ["Dim3/", "NCM2_DS_15000samples_10obs", "NCM2_DS_300samples_1000obs", plist3, plist3], 
    
    ["Dim4/", "NCM0_DS_20000samples_10obs", "NCM0_DS_400samples_1000obs", plist4, plist4], 
    ["Dim4/", "NCM1_DS_20000samples_10obs", "NCM1_DS_400samples_1000obs", plist4, plist4], 
    ["Dim4/", "NCM2_DS_20000samples_10obs", "NCM2_DS_400samples_1000obs", plist4, plist4], 
    
    ["Dim8/", "NCM5_DS_40000samples_10obs", "NCM5_DS_800samples_1000obs", plist8, plist8], 
    ["Dim8/", "NCM0_DS_40000samples_10obs", "NCM0_DS_800samples_1000obs", plist8, plist8], 
    ["Dim8/", "NCM1_DS_40000samples_10obs", "NCM1_DS_800samples_1000obs", plist8, plist8], 
    
    ["Dim12/", "NCM5_DS_60000samples_10obs", "NCM5_DS_1200samples_1000obs", plist12, plist12], 
    ["Dim12/", "NCM0_DS_60000samples_10obs", "NCM0_DS_1200samples_1000obs", plist12, plist12], 
    ["Dim12/", "NCM1_DS_60000samples_10obs", "NCM1_DS_1200samples_1000obs", plist12, plist12], 
    
    ["Dim16/", "NCM0_DS_80000samples_10obs", "NCM0_DS_1600samples_1000obs", plist16, plist16], 
    ["Dim16/", "NCM1_DS_80000samples_10obs", "NCM1_DS_1600samples_1000obs", plist16, plist16], 
    ["Dim16/", "NCM2_DS_80000samples_10obs", "NCM2_DS_1600samples_1000obs", plist16, plist16], 
    
    ["Dim20/", "NCM0_DS_100000samples_10obs", "NCM0_DS_2000samples_1000obs", plist20, plist20],
    ["Dim20/", "NCM1_DS_100000samples_10obs", "NCM1_DS_2000samples_1000obs", plist20, plist20],
    ["Dim20/", "NCM2_DS_100000samples_10obs", "NCM2_DS_2000samples_1000obs", plist20, plist20],


    ]

novel_case_studies = [

    ["Eco/", "Eco_DS_30000samples_20obs", "Eco_DS_600samples_1000obs",  plist_eco, plist_eco],
    ["MAPK/", "MAPK_DS_50000samples_10obs", "MAPK_DS_1000samples_1000obs", plist_mapk, plist_mapk]

]
data_path = 'data/'
models_path = 'out/models/'
plots_path = 'out/plots/'
