# Fault-Proneness in Coverage-Based Test Case Prioritization

## Introduction
#### Project Overview
This project explores test case prioritization (TCP) methods aimed at enhancing software regression testing. By prioritizing test cases to minimize the time required to detect failing test cases, our implementation integrates fault-proneness estimations derived from a neural network defect predictor into coverage-based TCP methods. Specifically, we replicate and validate findings from the study by Mahdieh et al. (2020), focusing on the benefits of incorporating fault-proneness into traditional TCP strategies. 

#### Objectives
1. **Replicating Findings**: Validate the claims from Mahdieh et al. (2020) regarding fault-proneness and clustering-based methods in improving test case prioritization.
2. **Evaluation Metrics**: Utilize Average Percentage of Faults Detected (APFD) as the primary metric to compare traditional and modified TCP strategies.
3. **Scope**: Analyze total and additional TCP strategies, including their fault-proneness-modified counterparts, on real-world software libraries.

#### Key Findings
1. **Modified Strategies Performance**:
   - Modified strategies incorporating fault-proneness outperformed traditional ones in APFD, validating the claims of the original study.
   - Results showed that the modified strategies detected faults earlier in the testing process, especially in larger libraries.

2. **Library Size Dependency**:
   - Larger libraries yielded more reliable fault-proneness predictions, leading to better performance of modified strategies.
   - Smaller libraries showed inconsistent results due to insufficient data for accurate defect predictions.

3. **Alignment with Research**:
   - Our findings closely align with those of Mahdieh et al. (2020), affirming the effectiveness of fault-proneness estimations in improving TCP.

#### Future Work
1. **Scalability Analysis**:
   - Expand testing to libraries of various sizes to evaluate the generalizability and scalability of modified TCP strategies.

2. **Cross-Project Defect Prediction**:
   - Investigate leveraging cross-project defect prediction methods to improve fault-proneness estimation across different domains and software systems.

3. **Broader Algorithm Comparisons**:
   - Extend the evaluation to other modified TCP algorithms to understand how fault-proneness impacts different prioritization techniques.

4. **Enhancing Defect Predictors**:
   - Optimize neural network models for fault-proneness predictions, especially in smaller libraries, to address limitations due to insufficient data.

#### Contributors
- **Sanjit Verma**
- **Arul Sharma**
- **Keith Tran**
- **Alex Taylor**

## Usage
This package is used in multiple steps: defect prediction, prioritization and result aggregation. The neccesary steps in order to execture the whole package once are listed below:

1. Get the code:
    ```
    git clone https://github.com/mostafamahdieh/FaultPronenessBasedTCP.git
    ```
2. Get the [Defects4J+M](https://github.com/khesoem/Defects4J-Plus-M) repository in the same main directory, naming it WTP-data:
    ```
    git clone https://github.com/khesoem/Defects4J-Plus-M.git WTP-data
    ```
   Keith's NOTE: Make sure this outside the folder, not inside of FaultPronenessBasedTCP
3. Install python and neccesary packages: **(CAN SKIP THIS STEP)**
    ```
    sudo apt-get install python3 python3-pip python3-venv
    ```
4. Create a python virtual environment and install neccesary pip packages:
    ```
    cd FaultPronenessBasedTCP
    python3 -m venv venv
    source ./venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
5. Defect prediction: The defect prediction step can be executed using the bugprediction_runner.py script as follows. This script runs the bug prediction step for the specific versions of all projects.
    ```
    cd bugprediction
    python3 -u bugprediction_runner.py
    ```

6. Test case prioritization: The prioritization_runner.py script is used to execute the traditional and fault-proneness based TCP methods. The total and additional strategies are executed in both the traditional and fault-proneness based methods.
    ```
    cd ../prioritization
    python3 -u prioritization_runner.py
    ```

7. Aggregating the results: The results are aggregated using the aggregate_results.py script:
    ```
    cd ../results
    python3 -u aggregate_results.py
    ```




