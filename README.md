# AMAT 5315 - Modern Scientific Computing: Homeworks

![Course](https://img.shields.io/badge/Course-AMAT%205315-blue) ![Term](https://img.shields.io/badge/Term-2025%20Fall-green) ![Language](https://img.shields.io/badge/Language-Julia-purple)

This repository contains homework assignments for AMAT 5315 - Modern Scientific Computing (Fall 2025). Follow the setup instructions below to get started with submitting your assignments.

**Supported Platforms:** macOS, Linux, Windows (please install WSL with `wsl --instsall`, see [instructions](https://learn.microsoft.com/en-us/windows/wsl/install#install-wsl-command))  
**Need Help?** Contact us via [GitHub Issues](https://github.com/CodingThrust/AMAT5315-2025Fall-Homeworks/issues) or [Zulip](https://zulip.hkust-gz.edu.cn)

---

## 🚀 Initial Setup (One-Time Only)

### Step 1: Install Git and Create GitHub Account

1. **Install Git**: Follow the instructions [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git)
2. **Create GitHub Account**: Sign up for a free account [here](https://github.com/signup)

### Step 2: Generate SSH Key

1. **Generate the key (Replace with your email)**:
   ```bash
   ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
   ```
   - Press Enter to use the default location (`~/.ssh/id_rsa`)
   - Press Enter twice to skip passphrase (or set one if preferred)

2. **Copy your public key**:
   ```bash
   cat ~/.ssh/id_rsa.pub
   ```
   Copy the entire output (starts with `ssh-rsa`)

3. **Add to GitHub**:
   - Go to GitHub → Click your avatar → Settings → SSH and GPG keys
   - Click "New SSH key"
   - Paste your key and give it a descriptive title
   - Click "Add SSH key"

4. **Test the connection**:
   ```bash
   ssh -T git@github.com
   ```

📖 **More details**: [GitHub SSH Documentation](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh)

### Step 3: Fork and Clone Repository

1. **Fork the repository**:
   - Visit the [course repository](https://github.com/CodingThrust/AMAT5315-2025Fall-Homeworks)
   - Click the "Fork" button in the top-right corner
   - This creates your own copy with write permissions

2. **Clone your forked repository**:
   ```bash
   # Navigate to where you want to store the repository
   cd ~/Documents  # or your preferred location
   
   # Clone your fork (replace YOUR_USERNAME with your GitHub username)
   git clone git@github.com:YOUR_USERNAME/AMAT5315-2025Fall-Homeworks.git
   
   # Enter the repository directory
   cd AMAT5315-2025Fall-Homeworks
   ```

3. **Add upstream remote**:
   ```bash
   # Add the original repository as upstream for future updates
   git remote add upstream https://github.com/CodingThrust/AMAT5315-2025Fall-Homeworks.git
   
   # Verify your remotes
   git remote -v
   ```

> ✅ **Setup Complete!** You only need to do these steps once. Now you're ready to submit homework.

---

## 📝 Homework Submission Workflow

> **📋 Quick Summary**: Create branch → Complete work → Commit → Push → Create PR → Address feedback (if any)

Follow these steps for **every** homework assignment:

### 1. 🌿 Prepare Your Working Branch

```bash
# Ensure you're on main branch and up-to-date
git checkout main
git pull upstream main

# Create a new branch (replace 'hw1' and 'yourname' accordingly)
git checkout -b hw1/yourname

# Create your working directory
mkdir hw1/yourname
cd hw1/yourname
```

> 💡 **Naming Convention**: Use `hwX/yourname` where `X` is the homework number and `yourname` is your name in lowercase (e.g., `hw1/zhongyini`)

### 2. 📝 Complete Your Assignment

- Read the homework description in `hwX/README.md`
- Complete all required tasks in your working directory
- Test your solutions thoroughly

### 3. 💾 Save Your Work

```bash
# Add all your changes
git add -A

# Commit with a descriptive message
git commit -m "Complete hw1: Add CPU info and Julia installation"

# Push to your fork
git push -u origin hw1/yourname
```

### 4. 📤 Create Pull Request

1. Go to your forked repository on GitHub
2. Click **"Contribute"** → **"Open pull request"**
   - Or click **"Compare & pull request"** if available
3. **Title**: Use your folder name (e.g., `hw1/yourname`)
4. **Description**: Briefly describe what you completed
5. Click **"Create pull request"**

### 5. 🔄 Address Feedback (If Needed)

If you receive feedback from instructors or TAs:

```bash
# Make your changes in the same branch
# ... edit your files ...

# Commit and push updates
git add -A
git commit -m "Address feedback: Fix error handling"
git push
```

The pull request will automatically update with your new changes.

### 6. 🎯 Grading & Deadlines

- **⏰ Deadline**: 2 weeks after homework release
- **🏆 Early Bird Bonus**: First correct submission gets extra credit
- **✅ Grading**: Homework is graded after PR is merged

---

## 🔧 Common Issues & Quick Fixes

**SSH Connection Problems:**
```bash
# Test your SSH connection
ssh -T git@github.com

# If it fails, check if your SSH key is added to GitHub
# and try adding it to your SSH agent:
ssh-add ~/.ssh/id_rsa
```

**Can't Push Changes:**
```bash
# Make sure you're on the right branch and pull first
git status
git pull origin hwX/yourname
git push origin hwX/yourname
```

**Need to Update from Course Repository:**
```bash
git checkout main
git pull upstream main
git checkout hwX/yourname
git merge main  # if you need the latest changes in your branch
```

---

## 📚 Additional Resources

- 📖 [Git Handbook](https://guides.github.com/introduction/git-handbook/) - Learn Git basics
- 🔄 [GitHub Flow Guide](https://guides.github.com/introduction/flow/) - Understanding pull requests
- 📝 [Markdown Guide](https://www.markdownguide.org/basic-syntax/) - Format your documentation  
- 🔬 [Julia Documentation](https://docs.julialang.org/) - Official Julia docs
