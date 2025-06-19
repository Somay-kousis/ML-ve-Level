
---

### 📥 Clone (Repo to Local)

```bash
git clone <repo-url>
```

---

### 🚀 Push (Local to Repo)

```bash
git push origin main
```

- `main` → branch name  
- `origin` → default remote repository

---

### 📊 `git status` – Displays current state of the code

1. **Untracked** – New file added (not yet staged)  
2. **Modified** – File changed (before staging)  
3. **Staged** – Ready to be committed

---

## 🗂️ Basic Terminal Commands

```bash
cd FolderName       # Go into folder  
cd ..               # Move out of folder  
ls                  # List files  
ls -a               # List all files, including hidden
```

---

### ✅ Adding Files

```bash
git add <file-name>   # Add specific file  
git add .             # Add all changes
```

---

### 💬 Commit Changes

```bash
git commit -m "Your message here"
```

---

## 🏁 Starting from Local → GitHub

### 🛠️ Initial Setup

```bash
mkdir <directory-name>   # Create new folder  
cd <directory-name>      # Move into the folder
```

```bash
git init                 # Initialize Git repo (creates .git folder)
```

---

### 🔗 Connect to Remote

```bash
git remote add origin <repo-url>   # Link to GitHub repo  
git remote -v                      # Check remote URL
```

---

### 🌿 Branch Setup

```bash
git branch                # Check current branch  
git branch -M main        # Rename branch to 'main'
```

---

## 🔁 General Workflow

```text
GitHub Repo → Clone → Make Changes → Add → Commit → Push
```




## 🌿 Git Branch Commands

---

### 🔍 Check Current Branch

```bash
git branch
```

---

### ✏️ Rename Current Branch

```bash
git branch -M main
```

---

### 🚀 Switch to Another Branch

```bash
git checkout <branch-name>
```

---

### 🌱 Create and Switch to New Branch

```bash
git checkout -b <new-branch-name>
```

---

### ❌ Delete a Branch

```bash
git branch -d <branch-name>
```

---

> 🧠 Tip: Use `-D` (capital D) instead of `-d` if you want to force delete a branch that hasn't been merged.

---

## 🔀 Git Merge

### 📥 Merge a Branch into Current Branch

```bash
git merge <branch-name>
```

---

### 🟢 Fast-Forward Merge

* Happens when no new commits in current branch
* Branch pointer moves forward (no merge commit)

---

### 🧩 3-Way Merge

* Happens when both branches have new commits
* Git creates a **merge commit** to combine changes

---

## ⚠️ Merge Conflicts

* Occurs when same lines are changed in both branches
* Git shows conflict markers in files:

```plaintext
<<<<<<< HEAD
Your version
=======
Their version
>>>>>>> branch-name
```

---

### 🛠️ Resolving Merge Conflicts

1. Edit the file manually
2. Remove conflict markers
3. Choose what to keep
4. Mark as resolved:

```bash
git add <file-name>
git commit -m "Resolved merge conflict"
```

---

### 🧪 Practice Flow

```bash
git checkout -b feature         # Create feature branch  
# make changes  
git commit -m "Some changes"

git checkout main
git merge feature               # Merge feature into main
```

---

---

## 🍴 Git Fork (GitHub)

### 🌐 Fork a Repo

* Go to the repo on GitHub
* Click `Fork` → This creates a copy under **your account**

---

### 📥 Clone Forked Repo

```bash
git clone <your-forked-repo-url>
```

---

### 🔗 Add Original Repo as Upstream

```bash
git remote add upstream <original-repo-url>
git remote -v     # Confirm remotes
```

---

### 🔄 Sync Fork with Original

```bash
git fetch upstream
git merge upstream/main
```

---

> 🧠 Tip: Use `pull` instead of `fetch + merge` for short-hand:

```bash
git pull upstream main
```

---

## ♻️ Undoing Changes

---

### 🔄 Undo `git add` (Unstage a file)

```bash
git restore --staged <file-name>
```

---

### 🗑️ Discard Local Changes

```bash
git restore <file-name>
```

---

### 🕓 Undo Last Commit (Keep changes)

```bash
git reset --soft HEAD~1
```

---

### 🧽 Undo Last Commit (Discard changes)

```bash
git reset --hard HEAD~1
```

---

### 🪓 Delete All Local Changes (Careful!)

```bash
git reset --hard
```

---

> 🧠 Tip: Use with caution — `--hard` deletes uncommitted work!

---


