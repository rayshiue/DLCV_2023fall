# Flipped-VQA
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install fairscale
pip install fire
pip install sentencepiece
pip install transformers
pip install timm
pip install pandas
pip install setuptools==59.5.0
pip install pysrt
pip install ftfy
# # LLaMA-Adapter-V2
# pip innstall pillow
# pip opencv-python
# pip install gradio
# Custom
pip install tqdm
pip install tensorboard
pip install gdown
pip install wget

# download data
if [ ! -d "./data" ]; then
    gdown 1CQc4_-AHlefSCvakO9Vc2cL5Dm7i0bl1 -O data.zip
    unzip ./data.zip
fi

# download clipvitl140.pth
gdown 1Vy82a_Rtn6JUaL0AfE3t41mzNJbU5tYv -O clipvitl140.pth
mv clipvitl140.pth data/star/clipvitl140.pth

# download pretrained model
if [ ! -d "./pretrained" ]; then
    mkdir pretrained
    mkdir pretrained/llama
    mkdir pretrained/llama/7B
    wget "https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/consolidated.00.pth"
    mv consolidated.00.pth pretrained/llama/7B/consolidated.00.pth
    wget "https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/params.json"
    mv params.json pretrained/llama/7B/params.json
    wget "https://huggingface.co/nyanko7/LLaMA-7B/resolve/main/tokenizer.model"
    mv tokenizer.model pretrained/llama/tokenizer.model
fi

# download model
gdown 1IkFWIvE1fGyVrBzHh95V7djoH29C7xT6 -O checkpoint_epoch=9_best_acc=0.6834319526627219_acc=0.6834319526627219.pth