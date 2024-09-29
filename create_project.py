import os
PROJECT_NAME = "sentiment-classification-NLP"
DIR_STRUCTURE = {
    "data": {},  
    "logs": {}, 
    "scripts": {  
        "preprocess.py": "",  
        "model.py": "",  
        "train.py": "",  
        "predict.py": ""  
    }
}

def create_structure(base_dir, structure):
    for key, value in structure.items():
        path = os.path.join(base_dir, key)
        if isinstance(value, dict):
            
            os.makedirs(path, exist_ok=True)
           
            create_structure(path, value)
        else:
            
            file_dir = os.path.dirname(path)
            if not os.path.exists(file_dir):
                os.makedirs(file_dir)
            with open(path, 'w') as file:
                pass

create_structure(PROJECT_NAME, DIR_STRUCTURE)

print(f"Project '{PROJECT_NAME}' has been created.")
