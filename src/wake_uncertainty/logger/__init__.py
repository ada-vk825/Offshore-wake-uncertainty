import os
import logging
from datetime import datetime

file_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

file_path = os.path.join("logs", file_name)
os.makedirs("logs", exist_ok=True)

logging.basicConfig(filename=file_path,
                    level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s")