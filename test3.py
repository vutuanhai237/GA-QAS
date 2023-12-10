import concurrent.futures
import numpy as np
def bypass_compile(circuit):
    print(np.sum(circuit))
    return np.sum(circuit)
def multiple_compile(circuits):
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(bypass_compile, circuits)
        return results
if __name__ == '__main__':
    results = multiple_compile([[1,2,3], [1,2,3,4]])
# for result in results:
#     print(result)