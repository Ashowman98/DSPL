# DSPL: Dual-Space Prompt Learning with Context Bias Decoupling for Open Set Recognition

This repository is the official implementation of **DSPL**.
---

## 📂 Project Structure

| File/Folder | Description |
| :--- | :--- |
| `osr_train.sh` | Bash script to execute the full training pipeline from scratch. |
| `config.py` | Central configuration file for dataset paths. |
| `main.py` | Main entry point for training and evaluation. |
| `dspl_model` | Core Architecture. Implementation of the proposed DSPL modules. |
| `data/` | Data Management. Directory for importing, storing, and preprocessing datasets. |

---

## 🛠 Installation
      pip install -r requirements.txt

## ⚙️ Set Dataset Path
      Before training, please setup dataset directories in `config.py` :
      ```
      cifar_10_root = '../data'                                   # path for cifar10
      tin_train_root_dir = '../data/tiny-imagenet-200/train'    # path for tinyimagenet train
      tin_val_root_dir = '../data/tiny-imagenet-200/val1'       # path for ood datasets val
      ```
##  🚀 Training from Scratch
       To train models from scratch, `osr_train.sh`