




def create_params(depths, num_circuits, num_generations):
    # zip num_generations, depths, num_circuits as params
    params = []
    for num_generation in num_generations:
        for depth in depths:
            for num_circuit in num_circuits:
                params.append((depth, num_circuit, num_generation))
    return params        
