The repository for ECEN766 course work


git clone 2 repository

```shell
git clone https://github.com/ai4protein/VenusPLM.git

git clone https://github.com/ai4protein/ProSST.git

```

For yining's environment(cuda127):

```shell
conda create -n prosst
conda activate prosst
```

```shell
cd ProSST

## I have met some problem for certain packages in the requirements.txt, I skipped some of them and pip install individually.
pip install -r requirements.txt

# install some pip packages individually
pip install torch-scatter torch-sparse torch-spline-conv  -f https://data.pyg.org/whl/torch-2.1.2+cu121.html


pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```