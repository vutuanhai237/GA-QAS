<<<<<<< HEAD
import time
import numpy as np
import concurrent.futures
import os

def main():

    file = open("Test_HPC.txt", "w")
    file.write("Number of CPU: {} \n" ,os.cpu_count())

    def square(x):
        return x**2
    
    number = 10**3
    start_time = time.time()

    executor = concurrent.futures.ProcessPoolExecutor()
    results = executor.map(square,  range(1, number))

    file.write("Parrallel: {} \n", time.time() - start_time)


    results = [square(x) for x in range(1, number)]  #46s, 56s

    start_time = time.time()

    file.write("Serial: {} \n", time.time() - start_time)


    file.write(str(list(results)))

    file.close()

if __name__ == '__main__':
    main()
=======
def main():

    file = open("Test_HPC.txt", "w")
    file.write("Test") 
    file.close()

if __name__ == '__main__':
    main()
>>>>>>> 641d69c914a41f23ea39d1b5eb55e0d121203180
