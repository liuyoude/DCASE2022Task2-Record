# set tsinghua source
#pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
# install package
#pip install -r requirements.txt
# run
python run.py
# evaluator
cd evaluator||exit
python evaluator.py
cd ..||exit
# tensorboard
tensorboard --logdir=runs

