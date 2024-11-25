# Fault-Proneness in Coverage-Based Test Case Prioritization (CSC 591 Team: Sanjit Verma, Alex Taylor, Keith Tran, Arul Sharma)

## Introduction
Test case prioritization (TCP) methods aim to benefit testing of a software (specifically regression testing), by prioritizing test cases in an order that minimizes the expected time of executing failing test cases. This project implements a test case prioritization method, based on estimating the fault proneness of code units using a neural network defect predictor, and incorporating it into coverage based TCP methods.

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




