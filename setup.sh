pip install -r requirements.txt
echo 'export PATH="$PATH:/home/cc/.local/bin"' >> ~/.bashrc
source  ~/.bashrc
rm -rf /mnt/heimdall-exp/Heimdall/integration/client-level/data/
cd /mnt/heimdall-exp/Heimdall/integration/client-level/
gdown https://drive.google.com/uc?id=149SkDXwpAEchrBTRL1zx8IxUo4E7Gwq6
tar -xvf data.tar.gz
rm -rf data.tar.gz