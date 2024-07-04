import os
from pathlib import Path

package_name = "components"

list_of_files = [
    "src/__init__.py",
    f"src/{package_name}/__init__.py",
    f"src/{package_name}/data_ingestion.py",
    f"src/{package_name}/data_transformation.py",
    f"src/{package_name}/data_evaluation.py",
    f"src/{package_name}/data_trainer.py",
    "src/exception/__init__.py",
    "src/exception/exception.py",
    "src/logger/__init__.py",
    "src/logger/logging.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/prediction_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "tests/__init__.py",
    "tests/unit/__init__.py",
    "tests/integration/__init__.py",
    "init_setup.sh",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "pyproject.toml",
    "tox.ini",
    "experiments/experiments.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass   # create an empty file