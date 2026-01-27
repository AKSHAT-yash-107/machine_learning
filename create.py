import os
import sys

if len(sys.argv) != 2:
    print("Usage: python create_project.py <project_name>")
    sys.exit(1)

project = sys.argv[1]

structure = {
    "": ["README.md", "requirements.txt", ".gitignore"],
    "data": [],
    "src": ["main.py"],
    "outputs": []
}

os.makedirs(project, exist_ok=True)

for folder, files in structure.items():
    path = os.path.join(project, folder)
    os.makedirs(path, exist_ok=True)
    for file in files:
        open(os.path.join(path, file), "w").close()

# sensible defaults
with open(os.path.join(project, ".gitignore"), "w") as f:
    f.write("__pycache__/\n.venv/\n.idea/\n")

with open(os.path.join(project, "README.md"), "w") as f:
    f.write(f"# {project}\n\nProject description.\n")

print(f"Project '{project}' created successfully.")
