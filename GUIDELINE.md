# **How to contribute?**

Instruction:
1. Install [Git](https://git-scm.com/) or [GitHub Desktop](https://desktop.github.com/), and an editor such as Visual Studio Code.
2. Send your GitHub username to haivt@uit.edu.vn, after that, I will add you as a collaborator in this repository.
3. Clone the codes to your computer (**qsee** is a core package of our various repository)
```
git clone https://github.com/vutuanhai237/EvolutionalQuantumCircuit.git
cd EvolutionalQuantumCircuit
git clone https://github.com/vutuanhai237/qsee.git
```

Note that the **qsee** folder must be on the same level as the **codes** folder.

4. Create a new branch (set name as your username) for your work and switch to your branch.
5. Make sure that you have installed python 3+ version. After that, install all needed packages.
```
pip install -r requirements.txt
```
6. Test

Run all test case
```
cd tests
pytest
```
or reading details in the codes folder, especially in the file codes/qevocircuit_QC.ipynb or codes/qevocircuit_VQE.ipynb.

- For the genetic algorithm project, we will mainly add/modify code in the qsee/evolution/ folder (such as adding more selection/mutation/crossover functions).

- Whenever you develop your code, you should use the jupyter notebook format, it will save you time.

- Frequently committing/pushing your code onto the git repository, if your code passes all tests, you can open a pull request which I can review and merge it into the main branch.

# **Testing and merging pull requests**

Your pull request will be automatically tested by qsee Github Action (testing status can be checked here: https://github.com/vutuanhai237/EvolutionalQuantumCircuit/actions). If any builders have failed, you should fix the issue. No need to close pull request and open a new one! Once all the builders are "green", I will review your code. 