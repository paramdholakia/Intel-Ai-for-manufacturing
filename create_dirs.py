import os

os.makedirs('app/static', exist_ok=True)
os.makedirs('app/templates', exist_ok=True)
os.makedirs('data_preprocessing', exist_ok=True)
os.makedirs('training', exist_ok=True)
os.makedirs('models', exist_ok=True)

with open('app/main.py', 'a') as f:
    pass
with open('data_preprocessing/preprocess.py', 'a') as f:
    pass
with open('training/train.py', 'a') as f:
    pass
with open('requirements.txt', 'a') as f:
    pass
