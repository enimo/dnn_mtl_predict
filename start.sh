# 1 安装

# 提前安装好conda 或 venv
# 然后执行：

CONDA_SUBDIR=osx-arm64 conda create -n predict_traffic python=3.11

conda activate predict_traffic

# conda deactivate


# 2 依赖
pip install -r requirements.txt


#3 train
python train.py

# test
# python test.py

