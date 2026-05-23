import os
import subprocess

def main():
    # 1. git mv folders from day1 -> day01, etc.
    for i in range(1, 10):
        old_dir = f"day{i}"
        new_dir = f"day0{i}"
        if os.path.exists(old_dir):
            print(f"Renaming {old_dir} to {new_dir}...")
            subprocess.run(["git", "mv", old_dir, new_dir])

    # 2. Update references in root README.md
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        for i in range(1, 10):
            # Replace markdown folder links like [day1/README.md](./day1/README.md)
            content = content.replace(f"day{i}/", f"day0{i}/")
            # Replace file links like [day1.py](./day1/day1.py)
            content = content.replace(f"day{i}.py", f"day0{i}.py")
            
        with open("README.md", "w", encoding="utf-8") as f:
            f.write(content)
        print("Updated README.md references.")

    # 3. Update path reference in day10 notebook
    nb_path = os.path.join("day10", "day10_feature_engineering.ipynb")
    if os.path.exists(nb_path):
        with open(nb_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        content = content.replace("../day9/cleaned_store_transactions.csv", "../day09/cleaned_store_transactions.csv")
        
        with open(nb_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("Updated day10 notebook reference.")

if __name__ == '__main__':
    main()
