# Transformer Training Guide (Image Classification)

This project includes:
- source images: 01.원천데이터
- label JSON files: 02.라벨링데이터
- training script: train_vit_defect_classifier.py

The script trains a Vision Transformer (ViT) classifier and exports:
- training graph PNG
- evaluation summary CSV
- classification report CSV

## 1) Install libraries

Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt

## 2) Run training

Basic run:

python train_vit_defect_classifier.py

CPU-only recommended run (no GPU):

python train_vit_defect_classifier.py --cpu-optimized --epochs 2 --batch-size 2

Recommended quick test (small subset first):

python train_vit_defect_classifier.py --max-samples 300 --epochs 1 --batch-size 4

Full options example:

python train_vit_defect_classifier.py --source-dir "01.원천데이터" --label-dir "02.라벨링데이터" --model-name google/vit-base-patch16-224 --epochs 3 --batch-size 8 --lr 5e-5 --val-ratio 0.2 --output-dir outputs_vit

CPU fine-tuning example (more stable on i7/16GB):

python train_vit_defect_classifier.py --source-dir "01.원천데이터" --label-dir "02.라벨링데이터" --cpu-optimized --epochs 3 --batch-size 2 --dataloader-workers 0 --output-dir outputs_vit_cpu

## 3) Output files

After training, outputs are saved to outputs_vit:
- outputs_vit/training_curves.png
- outputs_vit/evaluation_summary.csv
- outputs_vit/classification_report.csv
- outputs_vit/training_log.csv
- outputs_vit/best_model/

## 4) Notes

- If GPU is available, PyTorch uses it automatically.
- First run downloads pretrained model weights from Hugging Face.
- If class imbalance is severe, increase --max-samples and epochs, or add weighted loss later.
- On CPU-only PCs, start with --cpu-optimized and keep batch size 2-4.
