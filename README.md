# fruit CNN training
Chạy trên Google Colab

# Clone repo

!git clone https://github.com/gadeptrai145/fruit-cnn-project.git

%cd fruit-cnn-project

# Kết nối Google Drive

from google.colab import drive

drive.mount('/content/drive')

# Chạy train

!python train_fruit.py --data_dir /content/drive/MyDrive/dataset --save_dir /content/model.h5 --epochs 10
