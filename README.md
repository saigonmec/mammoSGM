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

3. **Project structure**

   ```
   mammoSGM/
   ├── src/
   │   ├── trainer/
   │   │   └── train_based.py
   │   └── ...existing code...
   ├── requirements.txt
   └── README.md
   ```

4. **Notes**

   - If there are configuration files or sample data, please place them in the correct location as indicated in the source code.
   - Read the comments in each file for more usage details.

## Contact

If you encounter any issues running the code, please contact the development team.
