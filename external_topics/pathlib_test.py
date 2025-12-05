import os
from pathlib import Path

# Get the current working Directory:
print(Path.cwd())

# Getting all the files inside the current directory:
# for p in Path().iterdir():
#     print(f"file name : {p} , respective file extension : {p.suffix}")

my_dir = Path("external_topics")
print(my_dir)

new_file = my_dir / "test.txt"
print(new_file)

# Getting the parent directory:
print(my_dir.parent.resolve())

# doing it dynamically:

# this gives the absolute path of the current file we are working in:
p = Path(__file__).resolve()

# gives the parent
p = Path(__file__).resolve().parent

print("p : ",p)

# Return a new path pointing to the user's home directory (as returned by os.path.expanduser('~')).
dotfiles = Path.home()

# for p in dotfiles.rglob("*vscode*"):
#     print(p)