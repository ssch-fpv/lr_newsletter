# How to Run the Project

This document provides step-by-step instructions to set up and execute the project.

---

## Prerequisites

Before running the project, ensure the following are installed on your system:

- Python 3.8 or newer
- `pip` (Python's package manager)

---

## Setting Up the Project

### 1. Clone the Repository

Clone the project repository to your local machine:

```bash
git clone https://github.com/ssch-fpv/lr_newsletter.git
cd lr_newsletter
```

---
### 2: Create a Virtual Environment

Create a virtual environment to isolate project dependencies:
```
python -m venv venv
````

---
### 3: Activate the Virtual Environment

Activate the virtual environment to prepare for dependency installation:

Windows:

```
.\venv\Scripts\activate
````

macOS/Linux:

```
source venv/bin/activate
```
---
### 4: Install Dependencies

Install all required Python libraries using the requirements.txt file:

```
pip install -r requirements.txt
````
---
### 5: Run the Main Script

Once the setup is complete, execute the main script:

```
python nl_main.py
````

What the Script Does:

    Generates synthetic data for training and testing.
    Trains the reinforcement learning agent.
    Evaluates the agent's performance.
    Produces visualizations, including:
        Confusion Matrix
        PCA plots


