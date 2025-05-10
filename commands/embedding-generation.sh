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

pip install -r requirements.txt

sleep infinity

'