import os

# Định nghĩa cấu trúc thư mục và file
project_structure = {
    "configs": ["params.yaml"],
    "data/raw": [],
    "data/processed": [],
    "notebooks": [
        "01_eda.ipynb",
        "02_preprocessing_feature.ipynb",
        "03_mining.ipynb",
        "04_modeling.ipynb",
        "04b_semi_supervised.ipynb",
        "05_evaluation_report.ipynb"
    ],
    "src/data": ["__init__.py", "loader.py", "cleaner.py"],
    "src/features": ["__init__.py", "builder.py"],
    "src/mining": ["__init__.py", "association.py", "clustering.py"],
    "src/models": ["__init__.py", "supervised.py", "semi_supervised.py"],
    "src/evaluation": ["__init__.py", "metrics.py", "report.py"],
    "src/visualization": ["__init__.py", "plots.py"],
    "scripts": ["run_pipeline.py", "run_papermill.py"],
    "outputs/figures": [],
    "outputs/tables": [],
    "outputs/models": [],
    "outputs/reports": []
}

files_at_root = ["requirements.txt", "README.md", ".gitignore"]

def create_structure():
    for folder, files in project_structure.items():
        # Tạo thư mục
        os.makedirs(folder, exist_ok=True)
        # Tạo các file trong thư mục
        for file in files:
            file_path = os.path.join(folder, file)
            with open(file_path, 'w', encoding='utf-8') as f:
                if file.endswith('.py') and '__init__' not in file:
                    f.write(f"# Logic for {file}\n")
                elif file == 'params.yaml':
                    f.write("# Project Parameters\nraw_data_path: 'data/raw/ai4i2020.csv'\n")
    
    # Tạo các file ở thư mục gốc
    for file in files_at_root:
        with open(file, 'w', encoding='utf-8') as f:
            f.write(f"# {file}")

    print("✅ Đã tạo xong toàn bộ cấu trúc dự án chuẩn GitHub Repo!")

if __name__ == "__main__":
    create_structure()