import numpy as np
import json


def main():
    method = "SLSQP"
    interation = 10000
    depth = 19
    distance = np.linspace(0.25, 2.5, 10)
    file = open(f"result_H4_{depth}depth_{method}_{str(interation)}.txt", "w")
    for i, dis in enumerate(distance):
        f = open(f"result_H4_{depth}depth_SLSQP_{dis}_{interation}.json")
        data = json.load(f)
        minimum_energy = min(data["energy"])
        minimum_position = data["energy"].index(minimum_energy)
        minimum_seed = data["seed"][minimum_position]
        print(minimum_energy)
        print(minimum_position)
        print(minimum_seed)
        file.write(
            f"{str(dis)}  {' '*(7-len(str(dis)))}{str(minimum_energy)} {' '*(20-len(str(minimum_energy)))} {str(minimum_seed)} \n"
        )  # Corrected formatting and removed comment
        # file.write(f"{str(dis)}  {str(minimum_energy)} \n") #{' '*(15-len(str(minimum_energy)))} {str(minimum_seed)} \n")
    file.close()


if __name__ == "__main__":
    main()

