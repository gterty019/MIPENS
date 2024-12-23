# Code for "Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach"
This repository contains the code and instructions to reproduce the experiments presented in our paper "Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach" authored by Georgios Tertytchny, Georgios L. Stavrinides, and Maria K. Michael. 
The paper will be published in the *Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)*. 
You can access the paper on [arXiv](https://arxiv.org/abs/2412.13439).

# Abstract
To address the challenges of imbalanced multi-class datasets typically used for rare event detection in critical cyber-physical systems, we propose an optimal, efficient, and adaptable mixed integer programming (MIP) ensemble weighting scheme. Our approach leverages the diverse capabilities of the classifier ensemble on a granular per class basis, while optimizing the weights of classifier-class pairs using elastic net regularization for improved robustness and generalization. Additionally, it seamlessly and optimally selects a predefined number of classifiers from a given set. We evaluate and compare our MIP-based method against six well-established weighting schemes, using representative datasets and suitable metrics, under various ensemble sizes. The experimental results reveal that MIP outperforms all existing approaches, achieving an improvement in balanced accuracy ranging from 0.99% to 7.31%, with an overall average of 4.53% across all datasets and ensemble sizes. Furthermore, it attains an overall average increase of 4.63%, 4.60%, and 4.61% in macro-averaged precision, recall, and F1-score, respectively, while maintaining computational efficiency.

# Source Code Availability
The complete source code used in this research, including scripts for training, weight computation, and inference, is available in this repository to ensure the reproducibility of our results and provide a practical foundation for further research and development based on our work.

# Prerequisites
1. **Bash** **Shell**: Ensure that you have a Bash-compatible shell installed to execute the script.
2. **Python 3.x:** The shell script calls several Python scripts. Ensure that Python 3 is installed and accessible via python3.
3. **Python Dependencies:** Install any required libraries for the Python scripts. Dependencies include:
   - numpy (https://numpy.org/)
   - pandas (https://pandas.pydata.org/)
   - scikit-learn (https://scikit-learn.org/stable/)
   - gurobipy (https://pypi.org/project/gurobipy/) 
4. **Gurobi Optimizer**: Ensure that [Gurobi Optimizer](https://www.gurobi.com) (version 11 or later) is installed and properly licensed.

# Datasets
1. **LeakDB**: https://zenodo.org/records/13985057 (Check also the **DATASETS** folder for the .csv file we used)
2. **NSL-KDD**: https://raw.githubusercontent.com/HoaNP/NSL-KDD-DataSet/refs/heads/master/KDDTrain%2B_20Percent.txt
3. **SG-MITM**: https://zenodo.org/records/8375657 (Check also the **DATASETS** folder for the .csv file we used)
4. **CIC-IDS2017**: https://www.unb.ca/cic/datasets/ids-2017.html

# Execution
To execute the code, run the RUN.sh script. If you need to change permissions for execution, use the following command:

```bash
chmod +x RUN.sh
```

# Workflow of RUN.sh
**Step 1:** Training & Accuracy Matrix Computation
1. Input dataset (e.g., D1_LEAKDB.csv) is trained for a set of classifiers using stratified k-folds (in our case **k=5**). The following classifiers are used: MLR, J48, JRIP, REPTree, MLP, SVM, GNB, and IBk. If desired, you can change them in script **0_1_STRATIFIED_KFOLDS.py**
3. Unnecessary lines are removed to produce the accuracy matrix for the classifiers

**Step 2:** Weight Computation

The following weighting schemes are used, as explained in our paper:
1.   Uniform Weights Per Classifier (UW, UW-PC in our paper)
2.   Uniform Weights Per Classifier-Class (UW_PCA, UW-PCC in our paper)
3.   Weighted Average Based on Normalized Accuracy Per Classifier (WA, WA-PC in our paper)
4.   Weighted Average Based on Normalized Accuracy Per Classifier-Class (WA_PCA, WA-PCC in our paper)
5.   Differential Evolution (DE)
6.   Bayesian Model Averaging (BMA)
7.   **MIP incorporating Elastic Net Regularization (MIPEN, MIP in our paper)**

**Step 3:** Inference

**Uncomment and customize the inference lines to use generated weights for evaluation:**

```bash
python3 "$inference_scripts_folder/2_INFERENCE_ALGORITHM.py" --dataset $input_file --weights 0.12,0.12,... --output "$results_folder/1_NSL_UW.csv"
```

All generated weights and inference results are saved in respective folders:

**WEIGHTS**: Contains weight files, such as UW_D1_LEAKDB.txt

**RESULTS**: Contains inference outputs, such as 1_NSL_UW.csv

**Running Example:**

By setting in RUN.sh the **dataset_code_number="D2_NSL_KDD"**

**The script will:**
1. Process **D2_NSL_KDD.csv**
2. Generate intermediate files **D2_NSL_KDD_ACC.csv** and **D2_NSL_KDD_ACCURACIES.csv**
3. Compute classifier weights and save them in the **WEIGHTS** folder
4. Run inference, compute results and save them in the **RESULTS** folder

Upon successful execution of the script, you will see the following message:

**Script execution completed.**

# Citation
If you use our code, methodology, and/or MIP-based weighting scheme, please cite our paper:
```bibtex
@misc{tertytchnyAAAI25,
      title={Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach}, 
      author={Georgios Tertytchny and Georgios L. Stavrinides and Maria K. Michael},
      year={2025},
      eprint={2412.13439},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.13439}, 
}
```

# Contact
For any queries, feel free to raise an issue in this GitHub repository or contact the maintainers directly.
