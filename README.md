# mammoSGM

## How to Run

1. **Install dependencies**

   Make sure you have Python (recommended >=3.8) and pip installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the training/testing script**

   The main entry point is in the `src.trainer.train_based` module. Use the following command to run in test mode:

   ```bash
   python -m src.trainer.train_based \
      --mode test \
      --config config.yaml \
      --data_folder /content/SGM815_cropbbx \
      --model_type resnet50 \
      --batch_size 16 \
      --num_epochs 10 \
      --output /content/runs \
      --img_size 224x224 \
      --pretrained_model_path /content/runs/models/SGM815_cropbbx_224x224_based_resnet50_7255.pth
   ```

   Adjust the arguments as needed for your environment.
3. **Run train_patch_with_metrics.py (to report metrics on test data)**

import os, shutil
from pathlib import Path

# %cd /mnt/data/SGM_PROJECT # cd to the project directory

PROJECT_DIR = os.getcwd()  # same as %pwd but pure Python
TARGET_COLUMN = "cancer"
DATA_FOLDER = "/mnt/data/SGM_PROJECT/OPTIMAM/optimam_v3"  # Update this path to your data folder

RESULTS_FOLDER = "/mnt/data/SGM_PROJECT/OPTIMAM/results"
SETTING = "mil" 

print(os.listdir(Path(f"{RESULTS_FOLDER}/{SETTING}")))

for weights_folder_name in os.listdir(Path(f"{RESULTS_FOLDER}/{SETTING}")):
    weights_folder = os.path.join(RESULTS_FOLDER, SETTING, weights_folder_name)
    if TARGET_COLUMN.lower() in str(weights_folder).lower():
        
        # Determine BACKBONE from weights_folder name
        if weights_folder.endswith("resnet34") or weights_folder.endswith("resnet50"):
            BACKBONE = weights_folder.split("/")[-1]
            BACKBONE = BACKBONE.split("_")[-1]
        else:
            BACKBONE = "convnextv2_tiny" # choose from "resnet34", "resnet50", "convnextv2_tiny"



        # Create output directory for results
        output_dir = os.path.join(PROJECT_DIR, "final_results")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        list_of_pretrained_model_paths = []

        for root, dirs, files in os.walk(weights_folder):
            for file in files:
                if file.endswith(".pth"):
                    PRETRAINED_MODEL_PATH = os.path.join(root, file)
                    list_of_pretrained_model_paths.append(PRETRAINED_MODEL_PATH)



        # cd into the repo
        %cd mammoSGM

        for PRETRAINED_MODEL_PATH in list_of_pretrained_model_paths:
            print(f"Testing with pretrained model: {PRETRAINED_MODEL_PATH}")

            # Run the test and capture the output
            !python -m src.trainer.train_patch_with_metrics \
                --mode test \
                --config config.yaml \
                --data_folder "{DATA_FOLDER}" \
                --model_type "{BACKBONE}" \
                --batch_size 16 \
                --num_epochs 50 \
                --output "{output_dir}" \
                --img_size 224x224 \
                --arch_type "mil_v4" \
                --target_column "{TARGET_COLUMN}" \
                --pretrained_model_path "{PRETRAINED_MODEL_PATH}" \
                --setting "{SETTING}" \
                --backbone_name "{BACKBONE}"
                
4. **Project structure**

   ```
   mammoSGM/
   ├── src/
   │   ├── trainer/
   │   │   └── train_based.py
   │   └── ...existing code...
   ├── requirements.txt
   └── README.md
   ```

6. **Notes**

   - If there are configuration files or sample data, please place them in the correct location as indicated in the source code.
   - Read the comments in each file for more usage details.

## Contact

If you encounter any issues running the code, please contact the development team.
