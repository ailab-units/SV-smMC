def sir_pac_config_details(list_params, n_points, n_obs):
    D = {}

    D["model"] = "SIR.psc"
    D["variables"] =  ["S","I","R"]
    D["parameters"] = ["beta","gamma"]
    D["state_space_dim"] = 3
    D["populationsize"] = 100
    D["param_space_dim"] = len(list_params)
    D["params"] = list_params
    D["time_step"] = 0.5
    D["n_steps"] = 240

    D["params_min"] = [0.005]
    D["params_max"] = [0.3]

    D["n_steps_param"] = n_points
    D["n_trajs"] = n_obs

    D["n_combination"] = D["n_steps_param"]**D["param_space_dim"]
    
    return D
