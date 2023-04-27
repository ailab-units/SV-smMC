import numpy as np

def gen_rnd_reactions(max_n_species, n_reactions):
    reaction_IOs = [(2,1), (1,2), (1,1), (1,0), (0,1)]
    react_prob = [0.25,   0.25,   0.25,  0.125, 0.125] # production and degradation reactions happen less frequently 

    onehot_reacts = []

    reactions = []
    selected_species = []
    j = 0
    while j < n_reactions:
        type_react_j = np.random.choice(len(reaction_IOs), p=react_prob)
        reacts_j = np.random.choice(max_n_species, size=reaction_IOs[type_react_j][0])
        prods_j = np.random.choice(max_n_species, size=reaction_IOs[type_react_j][1])
        # needed to check uniqueness of the reactions
        onehot_j = np.zeros(max_n_species*2)
        onehot_j[reacts_j] = 1
        onehot_j[prods_j+max_n_species] = 1
        
        pair_j = (reacts_j,prods_j)
        open_pair_j = [*reacts_j,*prods_j]

        if not (reaction_IOs[type_react_j][0] == reaction_IOs[type_react_j][1] == 1 and reacts_j[0]==prods_j[0]):
            #print("reaction {}/{}".format(j+1,n_reactions))
            if not np.any([np.array_equiv(onehot_j,_) for _ in onehot_reacts]): # check that reactions are non-redondant
                onehot_reacts.append(onehot_j)
                reactions.append(pair_j)
                for s in open_pair_j:
                    selected_species.append(s)
                j += 1

    return reactions, np.unique(selected_species) # returns the list of selected species (sorted)