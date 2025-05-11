bash -c '

apt update;

apt install -y wget;

mkdir -p /workspace;

DEBIAN_FRONTEND=noninteractive apt-get install openssh-server -y;

mkdir -p ~/.ssh;

chmod 700 ~/.ssh;

echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys;

chmod 700 ~/.ssh/authorized_keys;

service ssh start;

apt install python3-pip -y;

pip3 install -U --no-cache-dir jupyterlab jupyterlab_widgets ipykernel ipywidgets;

jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.terminado_settings="{"shell_command":["/bin/bash"]}" --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --FileContentsManager.preferred_dir=/workspace;

apt install git -y;

cd /root;

rm -rf ./embedding-generation;

git clone https://github.com/nudin10/embedding-generation.git;

mkdir -p ./embedding-generation/data;

ls;

wget https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Magazine_Subscriptions.jsonl.gz;

apt install gzip -y;

gzip -d Magazine_Subscriptions.jsonl.gz;

mv Magazine_Subscriptions.jsonl ./embedding-generation/data/Magazine_Subscriptions.jsonl;

cd embedding-generation;

apt install -y python3-venv;

python3 -m venv venv;

source venv/bin/activate;

pip install -r requirements.txt;

# To properly install the correct CUDA compatible PyTorch
CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $11}')
echo "Detected CUDA Version: $CUDA_VERSION"
if [[ "$CUDA_VERSION" == *"11.8"* ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
elif [[ "$CUDA_VERSION" == *"12."* ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 # Or the specific 12.x version
else
  echo "Warning: CUDA version not explicitly handled in script. Attempting default PyTorch install."
  pip install torch torchvision torchaudio
fi

sleep infinity

'