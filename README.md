# GFN-Check: Diverse Valid Test Inputs with Generative Flow Networks

[Kanghyeon Kim](https://github.com/KAIST19), [Seungwoo Lee](https://github.com/tfoseel), and [Ayhan Suleymanzade](https://github.com/MisakiTaro0414)

## Project Proposal
A variety of methods have been used for property-based testing, a powerful method for testing programs expecting highly-structured inputs. Reinforcement learning (RL) is often used to find such inputs which maximize rewards. However, RL-based methods tend to generate only high-reward inputs, limiting diversity.

To address this issue, [Reddy et al. [1]](https://dl.acm.org/doi/abs/10.1145/3377811.3380399?casa_token=5rEzwOGdaCgAAAAA%3AoN8jOYAq3MzT8dBPyiY2tgz6xO7yytVls-kiNSkGZF4HmQyz96ZNkajavSqI_RHydN_e5hRZ10tnCg) proposed RLCheck, an on-policy RL approach to generating diverse inputs for property-based testing. It basically assigns high rewards to the “unique” valid inputs and uses Monte Carlo Control for sampling. However, this approach is still prone to overfitting to a certain space of valid inputs as RL itself is not designed for sampling. Moreover, in this approach, we should store all the inputs we have generated so far to incentivize unique inputs, requiring a large amount of space and time.

We believe that [GFlowNets (GFN) [2]](https://arxiv.org/abs/2111.09266), a family of probabilistic techniques that sample objects in a way that the generation probability is proportional to its reward, is better suited to provide the guidance in RLCheck. As illustrated in the figure below, while RL only aims to find the global maximum, GFN samples diverse inputs including the ones with low rewards. Moreover, since GFN itself is designed for sampling diverse inputs, there is no need for us to keep track of all the inputs that have been generated. That is, we can generate diverse inputs in a more time- and space-efficient manner.

![A figure showing the difference of RL and GFN](https://i.esdrop.com/d/f/96jV6yefYa/DcRyOEVh7S.jpg)

Based on the work of [Reddy et al. [1]](https://dl.acm.org/doi/abs/10.1145/3377811.3380399?casa_token=5rEzwOGdaCgAAAAA%3AoN8jOYAq3MzT8dBPyiY2tgz6xO7yytVls-kiNSkGZF4HmQyz96ZNkajavSqI_RHydN_e5hRZ10tnCg), we aim to find out if replacing the RL with GFN actually leads to a better performance for property-based testing. First, we start with a synthetic task where the goal is generating valid Binary Search Trees (BST). The existing work has conducted an experiment with RL utilizing the set of inputs generated before with diverse rewards, and we are going to expand this work with three additional experiments: 1) GFN with uniform rewards for all valid BSTs, 2) GFN utilizing the set of inputs generated before with diverse rewards, and 3) RL with uniform rewards for all valid BSTs. Also, we plan to compare our method to other baselines with four real-world Java benchmarks, used in the original evaluation of the state-of-the-art tool, [Zest [3]](https://rohan.padhye.org/files/zest-issta19.pdf), on its valid test-input generation ability.


## Running Codes
The version of Python is ```3.11.3```.
To clone the project, run the following commands:
```bash
git clone https://github.com/tfoseel/gfn-check
```
Please create conda environment and install the requirements:
```bash
conda create -n gfn-check python=3.11
conda activate gfn-check
python3 -m pip install -r requirements.txt
```
We have 3 tasks. The structure of each task resembles that of python implementation of BST task in [Reddy et al. [1]](https://github.com/sameerreddy13/rlcheck).

To run the fuzzer of each task, run the following commands:
```bash
cd Task  # One of BST, POM, Student
python -m fuzz
```


## Implementation
### Generator implementation
To implement a fuzzer using random, RL, or GFN, go to a ```generators``` folder in each tasks,  implement two functions ```select()``` and ```reward()``` in ```Oracle``` class. The input parameters should be same throughout all Oracles, so that they can be tested together in ```fuzz.py```. (See ```BST/fuzz.py``` for example.)

### Task specification
For modifying current XML tasks or making new one, you have to write ```schema_name.xsd``` and ```config.json```. For now, ```config.json``` has two fields: ```tags``` and ```texts```. The fuzzer selects a tag and a text inside the tag. Texts are placeholders.

If you want to introduce some attributes of XMLs, then you can specify available attributes list in ```config.json``` and modify ```generate_*()``` function in ```fuzz.py```.