# Code for "Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach"
This repository contains the code we used to conduct the experiments presented in our paper "Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach" authored by Georgios Tertytchny, Georgios L. Stavrinides, and Maria K. Michael, published in the *Proceedings of the 39th AAAI Conference on Artificial Intelligence (AAAI-25)*. 

Our paper is available at:
- **AAAI-25 Proceedings**: [DOI: 10.1609/aaai.v39i19.34300](https://doi.org/10.1609/aaai.v39i19.34300)
- **arXiv** (includes supplementary material): [arXiv: 2412.13439](https://arxiv.org/abs/2412.13439)

# Abstract
To address the challenges of imbalanced multi-class datasets typically used for rare event detection in critical cyber-physical systems, we propose an optimal, efficient, and adaptable mixed integer programming (MIP) ensemble weighting scheme. Our approach leverages the diverse capabilities of the classifier ensemble on a granular per class basis, while optimizing the weights of classifier-class pairs using elastic net regularization for improved robustness and generalization. Additionally, it seamlessly and optimally selects a predefined number of classifiers from a given set. We evaluate and compare our MIP-based method against six well-established weighting schemes, using representative datasets and suitable metrics, under various ensemble sizes. The experimental results reveal that MIP outperforms all existing approaches, achieving an improvement in balanced accuracy ranging from 0.99% to 7.31%, with an overall average of 4.53% across all datasets and ensemble sizes. Furthermore, it attains an overall average increase of 4.63%, 4.60%, and 4.61% in macro-averaged precision, recall, and F1-score, respectively, while maintaining computational efficiency.

# Source Code Availability
The source code used in our work is available in this repository as a single script (`main.py`) that performs all phases of our methodology. 
Specifically:
1. **Phase 1 - Training & Validation:** Each base classifier is trained using stratified cross-validation.
2. **Phase 2 - Weight Calculation:** The classifier weights are calculated based on the examined weighting scheme.
3. **Phase 3 - Test & Evaluation:** The calculated weights are assigned to the classifiers and the performance of the resulting ensemble model is evaluated. 

# Prerequisites
1. **Python 3.x:** Ensure that Python 3 is installed.
2. **Python Dependencies:** Ensure that the following python libraries are installed:
   - pandas
   - numpy
   - scikit-learn
   - scipy
   - gurobipy

You can install the required libraries using pip:
```bash
pip install pandas numpy scikit-learn scipy gurobipy
```
3. **Gurobi Optimizer:** Ensure that [Gurobi Optimizer](https://www.gurobi.com) (version 11 or later) is installed and properly licensed.

# Datasets
The datasets used in our experiments are publicly available and are described in our paper.

# Running the Script
To run the script, execute:
```bash
python main.py
```

# Execution Overview
The script evaluates different ensemble weighting schemes for multi-class classification. It supports multiple base classifiers and weighting approaches, including uniform weighting, weighted averaging, differential evolution, Bayesian model averaging, and mixed-integer programming (MIP) with elastic net regularization (proposed approach).

# Execution Details
The script expects a dataset in CSV format, where the last column contains class labels. The user is prompted for the dataset path and whether pre-trained classifiers should be used. The execution workflow consists of:

- **Preprocessing:** Encodes categorical labels, standardizes features, and splits data into training/validation and test sets.
- **Phase 1 - Training & Validation:** Base classifiers are trained using stratified 5-fold cross-validation unless pre-trained models are provided.
- **Phase 2 - Weight Calculation:** Weights are computed for the selected ensemble size using each weighting scheme.
- **Phase 3 - Test & Evaluation:** The performance of the ensemble is assessed using balanced accuracy, macro-averaged precision, macro-averaged recall, and macro-averaged F1-score.

**Outputs:** The script generates the following:
1. A `<Dataset_filename>_class_counts_<RunID>.csv` file with the class distribution of the provided dataset.
2. A `Trained_Fold_Models_<RunID>/fold_models.joblib` directory containing the trained models for each fold.
3. A `<Dataset_filename>_accuracies_matrix_<RunID>.csv` file with classifier mean validation accuracies.
4. A `<Dataset_filename>_weight_matrices.csv` file with the calculated classifier weights for each ensemble size (and classifier combination) for all weighting schemes.
5. A `<Dataset_filename>_gurobi_log_<RunID>.log` file containing the Gurobi log (concerns the execution of MIP).
6. A `<Dataset_filename>_gurobi_results.csv` file summarizing the results of Gurobi (concerns the execution of MIP).
7. A `<Dataset_filename>_results.csv` file summarizing the final evaluation metrics for each ensemble size (and classifier combination) for all weighting schemes.

**Note:** Each execution generates a unique `RunID` (timestamp-based), which is either included in the generated filename or recorded within the file to ensure traceability of results.

# Citation
If you use our code, methodology, and/or MIP-based weighting scheme, please cite our paper:
```bibtex
@article{TertytchnyAAAI2025, 
	title={Rare Event Detection in Imbalanced Multi-Class Datasets Using an Optimal MIP-Based Ensemble Weighting Approach},
	volume={39},
	url={https://ojs.aaai.org/index.php/AAAI/article/view/34300},
	DOI={10.1609/aaai.v39i19.34300},
	number={19},
	journal={Proceedings of the AAAI Conference on Artificial Intelligence},
	author={Tertytchny, Georgios and Stavrinides, Georgios L. and Michael, Maria K.},
	year={2025},
	month={Apr.},
	pages={20867--20875}
}
```

# Contact
For any queries, feel free to raise an issue in this GitHub repository or contact the maintainers directly.
