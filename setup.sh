pip install -r requirements.txt
echo 'export PATH="$PATH:/home/cc/.local/bin"' >> ~/.bashrc
source  ~/.bashrc
rm -rf ~/Heimdall/integration/client-level/data/
cd ~/Heimdall/integration/client-level/
gdown https://drive.google.com/uc?id=1MoyVqNjnvJINlRnvfZWRwqzIDbLYQ8dA
tar -xvf data.tar.gz
rm -rf data.tar.gz