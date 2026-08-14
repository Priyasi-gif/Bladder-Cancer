import os
import json
import time
import random
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from scipy.stats import mode

# ============================================================
# QISKIT
# ============================================================

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


# ============================================================
# PARAMETERS
# ============================================================

DATASET_FILE = "Top50_Expression_Dataset.csv"

N_QUBITS = 6

N_UP_GENES = 3

N_DOWN_GENES = 3

N_ITERATIONS = 100

N_ENSEMBLE = N_ITERATIONS

TEST_SIZE = 0.20

RANDOM_STATE = 42


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

MODEL_DIR = "QSVC_Ensemble_100"

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# RANDOM SEED
# ============================================================

np.random.seed(
    RANDOM_STATE
)

random.seed(
    RANDOM_STATE
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

df = pd.read_csv(
    DATASET_FILE
)

print(
    "Original Dataset Shape:",
    df.shape
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

if "gene" not in df.columns:

    raise ValueError(
        "The dataset must contain a 'gene' column."
    )


if "Label" not in df.columns:

    raise ValueError(
        "The dataset must contain a 'Label' column."
    )


# ============================================================
# FIND REGULATION ROW
# ============================================================

regulation_rows = df[
    df["gene"]
    .astype(str)
    .str.strip()
    .str.lower()
    == "regulation"
]


if len(regulation_rows) == 0:

    raise ValueError(
        "Regulation row was not found."
    )


if len(regulation_rows) > 1:

    raise ValueError(
        "More than one Regulation row was found."
    )


regulation_row = regulation_rows.iloc[0]


# ============================================================
# IDENTIFY UPREGULATED AND DOWNREGULATED GENES
# ============================================================

upregulated_genes = []

downregulated_genes = []


for column in df.columns:

    if column in [
        "gene",
        "Label"
    ]:
        continue


    regulation = str(
        regulation_row[column]
    ).strip().lower()


    if regulation == "upregulated":

        upregulated_genes.append(
            column
        )


    elif regulation == "downregulated":

        downregulated_genes.append(
            column
        )


# ============================================================
# PRINT GENE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("GENE REGULATION INFORMATION")
print("=" * 70)

print(
    "Total Upregulated Genes:",
    len(upregulated_genes)
)

print(
    "Total Downregulated Genes:",
    len(downregulated_genes)
)


print("\nFirst 10 Upregulated Genes:")

print(
    upregulated_genes[:10]
)


print("\nFirst 10 Downregulated Genes:")

print(
    downregulated_genes[:10]
)


# ============================================================
# CHECK GENE COUNT
# ============================================================

if len(upregulated_genes) < N_UP_GENES:

    raise ValueError(
        f"At least {N_UP_GENES} "
        "upregulated genes are required."
    )


if len(downregulated_genes) < N_DOWN_GENES:

    raise ValueError(
        f"At least {N_DOWN_GENES} "
        "downregulated genes are required."
    )


# ============================================================
# REMOVE REGULATION ROW
# ============================================================

df = df[
    df["gene"]
    .astype(str)
    .str.strip()
    .str.lower()
    != "regulation"
].copy()


print(
    "\nDataset Shape After Removing Regulation Row:",
    df.shape
)


# ============================================================
# LABEL CONVERSION
# ============================================================

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
)


df["Label"] = df["Label"].map({

    "Cancer": 1,

    "Normal": 0

})


# Remove unknown labels

df = df.dropna(
    subset=["Label"]
)


df["Label"] = df[
    "Label"
].astype(int)


# ============================================================
# FEATURES
# ============================================================

all_gene_columns = (

    upregulated_genes

    +

    downregulated_genes

)


X = df[
    all_gene_columns
].copy()


y = df[
    "Label"
].copy()


# ============================================================
# CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(
    pd.to_numeric,
    errors="coerce"
)


X = X.fillna(0)


# ============================================================
# DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print(
    "Samples:",
    X.shape[0]
)

print(
    "Genes:",
    X.shape[1]
)

print(
    "Cancer Samples:",
    int((y == 1).sum())
)

print(
    "Normal Samples:",
    int((y == 0).sum())
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

    stratify=y

)


print(
    "\nTraining Samples:",
    X_train.shape[0]
)

print(
    "Testing Samples:",
    X_test.shape[0]
)


# ============================================================
# GENERATE UNIQUE GENE COMBINATIONS
#
# EACH ITERATION:
#
# 3 UPREGULATED
# +
# 3 DOWNREGULATED
#
# Genes CAN be reused in different iterations.
#
# But the complete 6-gene combination will not
# be repeated.
# ============================================================

rng = np.random.default_rng(
    RANDOM_STATE
)


selected_combinations = []

used_combinations = set()


print("\n" + "=" * 70)
print("GENERATING UNIQUE GENE COMBINATIONS")
print("=" * 70)


iteration = 0


while iteration < N_ITERATIONS:

    # --------------------------------------------------------
    # SELECT 3 UPREGULATED GENES
    # --------------------------------------------------------

    selected_up = tuple(
        sorted(
            rng.choice(
                upregulated_genes,
                size=N_UP_GENES,
                replace=False
            )
        )
    )


    # --------------------------------------------------------
    # SELECT 3 DOWNREGULATED GENES
    # --------------------------------------------------------

    selected_down = tuple(
        sorted(
            rng.choice(
                downregulated_genes,
                size=N_DOWN_GENES,
                replace=False
            )
        )
    )


    # --------------------------------------------------------
    # COMPLETE 6-GENE COMBINATION
    # --------------------------------------------------------

    combination = (
        selected_up
        +
        selected_down
    )


    # --------------------------------------------------------
    # CHECK DUPLICATE
    # --------------------------------------------------------

    if combination in used_combinations:

        continue


    used_combinations.add(
        combination
    )


    selected_combinations.append({

        "upregulated":
            list(selected_up),

        "downregulated":
            list(selected_down),

        "all_genes":
            list(combination)

    })


    iteration += 1


print(
    f"Generated {N_ITERATIONS} unique gene combinations."
)


# ============================================================
# SAVE SELECTED GENES FOR ALL 100 ITERATIONS
# ============================================================

gene_selection_records = []


for i, combination in enumerate(
    selected_combinations,
    start=1
):

    gene_selection_records.append({

        "Iteration":
            i,

        "Model_Name":
            f"Model_QSVC_Model_{i:03d}",

        "Upregulated_Genes":
            ", ".join(
                combination["upregulated"]
            ),

        "Downregulated_Genes":
            ", ".join(
                combination["downregulated"]
            ),

        "Selected_Genes":
            ", ".join(
                combination["all_genes"]
            )

    })


gene_selection_df = pd.DataFrame(
    gene_selection_records
)


gene_selection_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_Selected_Genes_100_Iterations.csv"

    ),

    index=False

)


print(
    "\nSaved:"
)

print(
    "Model_Selected_Genes_100_Iterations.csv"
)


# ============================================================
# ENSEMBLE STORAGE
# ============================================================

ensemble_models = []

ensemble_predictions = []

iteration_metrics = []

training_times = []


# ============================================================
# TRAIN 100 QSVC MODELS
# ============================================================

for iteration in range(
    1,
    N_ITERATIONS + 1
):

    print("\n")
    print("=" * 70)

    print(
        f"QSVC MODEL {iteration}/{N_ITERATIONS}"
    )

    print("=" * 70)


    # ========================================================
    # GET GENE COMBINATION
    # ========================================================

    combination = selected_combinations[
        iteration - 1
    ]


    selected_up = combination[
        "upregulated"
    ]


    selected_down = combination[
        "downregulated"
    ]


    selected_genes = combination[
        "all_genes"
    ]


    print(
        "\nUpregulated Genes:"
    )

    print(
        selected_up
    )


    print(
        "\nDownregulated Genes:"
    )

    print(
        selected_down
    )


    print(
        "\nSelected Genes:"
    )

    print(
        selected_genes
    )


    # ========================================================
    # SELECT TRAINING FEATURES
    # ========================================================

    X_train_group = X_train[
        selected_genes
    ]


    X_test_group = X_test[
        selected_genes
    ]


    # ========================================================
    # SCALING
    # ========================================================

    scaler = MinMaxScaler(

        feature_range=(
            0,
            np.pi
        )

    )


    X_train_scaled = scaler.fit_transform(

        X_train_group

    )


    X_test_scaled = scaler.transform(

        X_test_group

    )


    # ========================================================
    # QUANTUM FEATURE MAP
    # ========================================================

    feature_map = ZZFeatureMap(

        feature_dimension=N_QUBITS,

        reps=2,

        entanglement="linear"

    )


    # ========================================================
    # QUANTUM KERNEL
    # ========================================================

    quantum_kernel = FidelityQuantumKernel(

        feature_map=feature_map

    )


    # ========================================================
    # QSVC
    # ========================================================

    qsvc = QSVC(

        quantum_kernel=quantum_kernel

    )


    # ========================================================
    # TRAIN
    # ========================================================

    start_time = time.time()


    qsvc.fit(

        X_train_scaled,

        y_train

    )


    training_time = (

        time.time()

        -

        start_time

    )


    training_times.append(
        training_time
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    prediction = qsvc.predict(

        X_test_scaled

    )


    ensemble_predictions.append(

        prediction

    )


    # ========================================================
    # ITERATION METRICS
    # ========================================================

    accuracy = accuracy_score(

        y_test,

        prediction

    )


    precision = precision_score(

        y_test,

        prediction,

        zero_division=0

    )


    recall = recall_score(

        y_test,

        prediction,

        zero_division=0

    )


    f1 = f1_score(

        y_test,

        prediction,

        zero_division=0

    )


    # ========================================================
    # ITERATION CLASSIFICATION REPORT
    # ========================================================

    report = classification_report(

        y_test,

        prediction,

        output_dict=True,

        zero_division=0

    )


    # ========================================================
    # MODEL NAME
    # ========================================================

    model_name = (

        f"Model_QSVC_Model_{iteration:03d}"

    )


    # ========================================================
    # SAVE INDIVIDUAL MODEL
    # ========================================================

    model_package = {

        "model_name":
            model_name,

        "qsvc":
            qsvc,

        "scaler":
            scaler,

        "selected_genes":
            selected_genes,

        "upregulated_genes":
            selected_up,

        "downregulated_genes":
            selected_down,

        "n_qubits":
            N_QUBITS,

        "feature_map_reps":
            2,

        "entanglement":
            "linear",

        "train_time":
            training_time

    }


    model_path = os.path.join(

        MODEL_DIR,

        f"{model_name}.joblib"

    )


    joblib.dump(

        model_package,

        model_path

    )


    # ========================================================
    # SAVE INDIVIDUAL MODEL PREDICTIONS
    # ========================================================

    prediction_df = pd.DataFrame({

        "Actual":
            y_test.values,

        "Predicted":
            prediction,

        "Correct":
            y_test.values == prediction

    })


    prediction_df.to_csv(

        os.path.join(

            MODEL_DIR,

            f"{model_name}_Predictions.csv"

        ),

        index=False

    )


    # ========================================================
    # SAVE ITERATION METRICS
    # ========================================================

    iteration_metrics.append({

        "Iteration":
            iteration,

        "Model":
            model_name,

        "Selected_Genes":
            ", ".join(
                selected_genes
            ),

        "Upregulated_Genes":
            ", ".join(
                selected_up
            ),

        "Downregulated_Genes":
            ", ".join(
                selected_down
            ),

        "Accuracy":
            accuracy,

        "Precision":
            precision,

        "Recall":
            recall,

        "F1_score":
            f1,

        "Training_Time":
            training_time,

        "Cancer_Precision":
            report["1"]["precision"],

        "Cancer_Recall":
            report["1"]["recall"],

        "Cancer_F1":
            report["1"]["f1-score"],

        "Normal_Precision":
            report["0"]["precision"],

        "Normal_Recall":
            report["0"]["recall"],

        "Normal_F1":
            report["0"]["f1-score"],

        "Number_of_0_Predictions":
            np.sum(
                prediction == 0
            ),

        "Number_of_1_Predictions":
            np.sum(
                prediction == 1
            )

    })


    # ========================================================
    # STORE MODEL
    # ========================================================

    ensemble_models.append(

        model_package

    )


    # ========================================================
    # PRINT ITERATION RESULTS
    # ========================================================

    print(
        "\nModel:",
        model_name
    )


    print(
        "Accuracy:",
        round(
            accuracy,
            6
        )
    )


    print(
        "Precision:",
        round(
            precision,
            6
        )
    )


    print(
        "Recall:",
        round(
            recall,
            6
        )
    )


    print(
        "F1 Score:",
        round(
            f1,
            6
        )
    )


    print(
        "Training Time:",
        round(
            training_time,
            3
        ),
        "seconds"
    )


# ============================================================
# CONVERT ALL PREDICTIONS TO ARRAY
# ============================================================

all_predictions = np.array(

    ensemble_predictions

)


print("\n")
print("=" * 70)
print("ALL QSVC MODELS TRAINED")
print("=" * 70)


print(
    "Prediction Array Shape:",
    all_predictions.shape
)


# ============================================================
# SAVE 100 ITERATION PREDICTIONS
#
# ROW = SAMPLE
# COLUMN = ITERATION
# ============================================================

prediction_results = pd.DataFrame(

    all_predictions.T,

    columns=[

        f"Iteration_{i+1}"

        for i in range(
            N_ITERATIONS
        )

    ]

)


prediction_results.insert(

    0,

    "Actual",

    y_test.values

)


prediction_results.to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_100_iteration_predictions.csv"

    ),

    index=False

)


print(
    "Saved: Model_QSVC_100_iteration_predictions.csv"
)


# ============================================================
# SAVE ITERATION CLASSIFICATION REPORT
# ============================================================

iteration_report_df = pd.DataFrame(

    iteration_metrics

)


iteration_report_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_100_iteration_classification_report.csv"

    ),

    index=False

)


print(
    "Saved: Model_QSVC_100_iteration_classification_report.csv"
)


# ============================================================
# MAJORITY VOTING
# ============================================================

final_prediction = mode(

    all_predictions,

    axis=0

).mode.flatten()


# ============================================================
# SAVE MAJORITY VOTE PREDICTIONS
# ============================================================

majority_prediction = pd.DataFrame({

    "Actual":

        y_test.values,


    "Majority_Prediction":

        final_prediction,


    "Correct":

        y_test.values == final_prediction

})


majority_prediction.to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_majority_vote_prediction.csv"

    ),

    index=False

)


print(
    "Saved: Model_QSVC_majority_vote_prediction.csv"
)


# ============================================================
# MAJORITY CLASSIFICATION REPORT
# ============================================================

majority_report = classification_report(

    y_test,

    final_prediction,

    output_dict=True,

    zero_division=0

)


print(
    "\n========================================"
)

print(
    "QSVC Majority Voting Result"
)

print(
    "========================================\n"
)


print(

    classification_report(

        y_test,

        final_prediction,

        target_names=[

            "Normal",

            "Cancer"

        ],

        zero_division=0

    )

)


# ============================================================
# MAJORITY REPORT VALUES
# ============================================================

majority_accuracy = accuracy_score(

    y_test,

    final_prediction

)


majority_precision = precision_score(

    y_test,

    final_prediction,

    zero_division=0

)


majority_recall = recall_score(

    y_test,

    final_prediction,

    zero_division=0

)


majority_f1 = f1_score(

    y_test,

    final_prediction,

    zero_division=0

)


# ============================================================
# SAVE MAJORITY REPORT CSV
# ============================================================

majority_summary = {

    "Model":

        "Model_QSVC_Ensemble_Majority_Voting",


    "Accuracy":

        majority_accuracy,


    "Precision":

        majority_precision,


    "Recall":

        majority_recall,


    "F1_score":

        majority_f1,


    "Ensemble_Size":

        N_ITERATIONS,


    "Average_Training_Time":

        np.mean(
            training_times
        ),


    "Total_Training_Time":

        np.sum(
            training_times
        ),


    "No_of_0":

        np.sum(
            final_prediction == 0
        ),


    "No_of_1":

        np.sum(
            final_prediction == 1
        ),


    "Cancer_Precision":

        majority_report["1"]["precision"],


    "Cancer_Recall":

        majority_report["1"]["recall"],


    "Cancer_F1":

        majority_report["1"]["f1-score"],


    "Normal_Precision":

        majority_report["0"]["precision"],


    "Normal_Recall":

        majority_report["0"]["recall"],


    "Normal_F1":

        majority_report["0"]["f1-score"]

}


pd.DataFrame(

    [majority_summary]

).to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_MajorityVoting_Classification_Report.csv"

    ),

    index=False

)


print(
    "Saved: Model_QSVC_MajorityVoting_Classification_Report.csv"
)


# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

print(
    "\n========================================"
)

print(
    "Final QSVC Ensemble Performance"
)

print(
    "========================================"
)


print(
    "\nAccuracy:",

    accuracy_score(

        y_test,

        final_prediction

    )

)


print(
    "Precision:",

    precision_score(

        y_test,

        final_prediction,

        zero_division=0

    )

)


print(
    "Recall:",

    recall_score(

        y_test,

        final_prediction,

        zero_division=0

    )

)


print(
    "F1 Score:",

    f1_score(

        y_test,

        final_prediction,

        zero_division=0

    )

)


print(
    "Average Training Time:",

    np.mean(
        training_times
    )
)


# ============================================================
# ADD FINAL ENSEMBLE TO ITERATION RESULTS
# ============================================================

iteration_metrics.append({

    "Iteration":
        "FINAL",

    "Model":
        "Model_QSVC_Ensemble_100_Majority_Voting",

    "Selected_Genes":
        "3 Upregulated + 3 Downregulated per model",

    "Upregulated_Genes":
        "Different combinations",

    "Downregulated_Genes":
        "Different combinations",

    "Accuracy":
        majority_accuracy,

    "Precision":
        majority_precision,

    "Recall":
        majority_recall,

    "F1_score":
        majority_f1,

    "Training_Time":
        np.sum(
            training_times
        ),

    "Cancer_Precision":
        majority_report["1"]["precision"],

    "Cancer_Recall":
        majority_report["1"]["recall"],

    "Cancer_F1":
        majority_report["1"]["f1-score"],

    "Normal_Precision":
        majority_report["0"]["precision"],

    "Normal_Recall":
        majority_report["0"]["recall"],

    "Normal_F1":
        majority_report["0"]["f1-score"],

    "Number_of_0_Predictions":
        np.sum(
            final_prediction == 0
        ),

    "Number_of_1_Predictions":
        np.sum(
            final_prediction == 1
        )

})


# ============================================================
# SAVE FINAL COMPLETE RESULTS CSV
#
# THIS FILE CONTAINS:
#
# MODEL
# ITERATION
# SELECTED GENES
# ACCURACY
# PRECISION
# RECALL
# F1
# CANCER METRICS
# NORMAL METRICS
# TRAINING TIME
# ============================================================

final_results_df = pd.DataFrame(

    iteration_metrics

)


final_results_df.to_csv(

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_Final_Complete_Results.csv"

    ),

    index=False

)


print(
    "\nSaved: Model_QSVC_Final_Complete_Results.csv"
)


# ============================================================
# SAVE FULL ENSEMBLE MODEL
# ============================================================

full_ensemble = {

    "ensemble_name":
        "Model_QSVC_Ensemble_100_Majority_Voting",

    "number_of_models":
        N_ITERATIONS,

    "models":
        ensemble_models,

    "gene_selection":
        selected_combinations,

    "upregulated_gene_list":
        upregulated_genes,

    "downregulated_gene_list":
        downregulated_genes,

    "n_qubits":
        N_QUBITS,

    "n_upregulated":
        N_UP_GENES,

    "n_downregulated":
        N_DOWN_GENES,

    "random_state":
        RANDOM_STATE,

    "test_size":
        TEST_SIZE,

    "ensemble_method":
        "Majority Voting"

}


joblib.dump(

    full_ensemble,

    os.path.join(

        MODEL_DIR,

        "Model_QSVC_Ensemble_100_Full.joblib"

    )

)


print(
    "\nSaved: Model_QSVC_Ensemble_100_Full.joblib"
)


# ============================================================
# SAVE METADATA
# ============================================================

metadata = {

    "ensemble_name":
        "Model_QSVC_Ensemble_100_Majority_Voting",

    "number_of_models":
        N_ITERATIONS,

    "qubits":
        N_QUBITS,

    "genes_per_model":
        6,

    "upregulated_genes_per_model":
        3,

    "downregulated_genes_per_model":
        3,

    "gene_reuse_between_iterations":
        True,

    "unique_gene_combinations":
        True,

    "ensemble_method":
        "Majority Voting",

    "feature_map":
        "ZZFeatureMap",

    "feature_map_reps":
        2,

    "entanglement":
        "linear",

    "test_size":
        TEST_SIZE,

    "random_state":
        RANDOM_STATE,

    "accuracy":
        float(
            majority_accuracy
        ),

    "precision":
        float(
            majority_precision
        ),

    "recall":
        float(
            majority_recall
        ),

    "f1_score":
        float(
            majority_f1
        ),

    "average_training_time":
        float(
            np.mean(
                training_times
            )
        ),

    "total_training_time":
        float(
            np.sum(
                training_times
            )
        )

}


with open(

    os.path.join(

        MODEL_DIR,

        "Model_ensemble_metadata.json"

    ),

    "w"

) as f:

    json.dump(

        metadata,

        f,

        indent=4

    )


print(
    "Saved: Model_ensemble_metadata.json"
)


# ============================================================
# FINAL FILE LIST
# ============================================================

print("\n")
print("=" * 70)
print("ALL SAVED FILES")
print("=" * 70)


print(
    "\nMain Ensemble Files:"
)


print(
    "1. Model_QSVC_Ensemble_100_Full.joblib"
)


print(
    "2. Model_QSVC_Ensemble_100_Results.csv"
)


print(
    "3. Model_QSVC_Ensemble_100_Final_Predictions.csv"
)


print(
    "4. Model_QSVC_100_iteration_predictions.csv"
)


print(
    "5. Model_QSVC_100_iteration_classification_report.csv"
)


print(
    "6. Model_QSVC_majority_vote_prediction.csv"
)


print(
    "7. Model_QSVC_MajorityVoting_Classification_Report.csv"
)


print(
    "8. Model_QSVC_Final_Complete_Results.csv"
)


print(
    "9. Model_Selected_Genes_100_Iterations.csv"
)


print(
    "10. Model_ensemble_metadata.json"
)


print(
    "\nIndividual Model Files:"
)


print(
    "Model_QSVC_Model_001.joblib"
)

print(
    "Model_QSVC_Model_002.joblib"
)

print(
    "Model_QSVC_Model_003.joblib"
)

print(
    "..."
)

print(
    "Model_QSVC_Model_100.joblib"
)


print(
    "\nIndividual Prediction Files:"
)


print(
    "Model_QSVC_Model_001_Predictions.csv"
)

print(
    "Model_QSVC_Model_002_Predictions.csv"
)

print(
    "Model_QSVC_Model_003_Predictions.csv"
)

print(
    "..."
)

print(
    "Model_QSVC_Model_100_Predictions.csv"
)


# ============================================================
# FINAL PERFORMANCE
# ============================================================

print("\n")
print("=" * 70)
print("FINAL PERFORMANCE")
print("=" * 70)


print(
    "\nAccuracy:",
    majority_accuracy
)


print(
    "Precision:",
    majority_precision
)


print(
    "Recall:",
    majority_recall
)


print(
    "F1 Score:",
    majority_f1
)


print(
    "Average Training Time:",
    np.mean(
        training_times
    ),
    "seconds"
)


print(
    "Total Training Time:",
    np.sum(
        training_times
    ),
    "seconds"
)


print(
    "\nFinished Successfully!"
)


print(
    "=" * 70
)