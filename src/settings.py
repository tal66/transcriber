from pathlib import Path

# paths
TEMP_FILES_DIR = './temp'
UPLOAD_DIR = './uploads'

Path(UPLOAD_DIR).mkdir(exist_ok=True)
Path(TEMP_FILES_DIR).mkdir(exist_ok=True)

# devices
LOOPBACK_DEVICE_ID = 16
MIC_DEVICE_ID = 1

# db
CONN_STRING = "mongodb://localhost:27017/"
