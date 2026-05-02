# fruit CNN training
Chạy trên Google Colab
1. Clone repo

!git clone https://github.com/gadeptrai145/fruit-cnn-project.git
%cd fruit-cnn-project

3. Kết nối Google Drive

from google.colab import drive
drive.mount('/content/drive')

4. Chạy train
!python train_fruit.py --data_dir /content/drive/MyDrive/dataset --save_dir /content/model.h5 --epochs 10
