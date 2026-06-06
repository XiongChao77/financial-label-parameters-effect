import os
import shutil
from pathlib import Path
current_work_dir = os.path.dirname(__file__) 

def organize_files(directory_path):
    """
    将文件名以'-'分隔、前缀为年份的文件自动归类到对应年份文件夹
    """
    directory = Path(directory_path)
    
    for file in directory.iterdir():
        if not file.is_file():
            continue
        
        parts = file.stem.split('-')  # stem获取不带扩展名的文件名
        if parts and parts[0].isdigit() and len(parts[0]) == 4:
            year = parts[0]
            target_dir = directory / year
            target_dir.mkdir(exist_ok=True)
            
            dest = target_dir / file.name
            if dest.exists():
                raise RuntimeError(f"dest already exist")
            
            shutil.move(str(file), str(dest))
            print(f"✓ {file.name} -> {year}/")

if __name__ == "__main__":
    organize_files(os.path.join(current_work_dir,'massive','minute'))