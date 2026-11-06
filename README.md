# 🩺 DiabeTech AI - Clinical Diabetes Prediction & ML Platform

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask%203.x-lightgrey.svg?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Bootstrap](https://img.shields.io/badge/UI-Bootstrap%205.3%20%2B%20Glassmorphism-purple.svg?logo=bootstrap&logoColor=white)](https://getbootstrap.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An intelligent, end-to-end clinical machine learning web platform designed to analyze patient physiological biomarkers, train ensemble Random Forest classifiers, evaluate real-time diabetes risk, and generate exploratory data visualizations.

Crafted with a **Bio-Sunset Coral & Smoked Obsidian Glassmorphism** design system, secure authentication, resilient database fallbacks, and a decision-support workstation interface.

---

## 📑 Table of Contents

- [Key Features](#-key-features)
- [System Architecture & Workflow](#-system-architecture--workflow)
- [Biomarkers & Clinical Dataset](#-biomarkers--clinical-dataset)
- [Project Directory Structure](#-project-directory-structure)
- [Technology Stack](#-technology-stack)
- [Installation & Quickstart](#-installation--quickstart)
- [Database Configuration](#-database-configuration)
- [Application Route Map](#-application-route-map)
- [Clinical Assessment & Presets](#-clinical-assessment--presets)
- [Author & Credits](#-author--credits)

---

## ✨ Key Features

### 1. 🎛️ Clinical Command Center & Executive Dashboard
- **Live KPI Status Cards**: Quick overview of the active ML algorithm, loaded biomarkers, target variable, and serialized model artifact.
- **Biomarker Reference Guide**: Built-in clinical table showing normal fasting ranges, metric units, and diagnostic significance for every physiological parameter.
- **Workflow Action Pipeline**: 4-stage guided cards covering ingestion, training, inference, and exploratory visual analysis.

### 2. 🔐 User Authentication & Session Security
- **Bcrypt Encryption**: Secure password hashing with salt generation.
- **Resilient Database Layer**: Automatic detection and graceful fallback between MySQL (`mysql+pymysql`) and local standalone SQLite (`ml_app.db`).
- **Interactive Auth Views**: Centered glassmorphic login and registration cards with show/hide password visibility toggles.

### 3. 📂 Dataset Ingestion & Validation
- **Drag & Drop Dropzone**: Seamless CSV file uploads with client-side file selection indicators and dragover highlights.
- **Schema & Data Preview**: Responsive table displaying columns, sample rows, and color-coded status badges for binary target outcomes (`1: Positive` vs. `0: Negative`).
- **Workspace Auto-Preloading**: Detects and loads pre-existing clinical datasets (`diabetes.csv`) automatically on startup.

### 4. 🧠 Random Forest Model Training Engine
- **Ensemble Classifier**: 100 decision trees trained with an 80/20 train-test split and fixed random seed for reproducible generalization.
- **Model Serialization**: Automatically creates and persists model binaries to `models/model.pkl`.
- **Pipeline Integration**: One-click retraining directly from the web interface.

### 5. ⚡ Real-Time Patient Risk Inference
- **Structured 2-Column Clinical Form**: Logically partitioned into Glycemic/Primary Biomarkers and Metabolic/Genetic Metrics.
- **Interactive Quick-Fill Presets**: Single-click demo buttons (**Sample: Low Risk** and **Sample: High Risk**) for instant testing and presentation.
- **Helpful Clinical Clues**: Units (`mg/dL`, `mm Hg`, `kg/m²`, `µU/mL`) and normal healthy boundary badges for every field.

### 6. 📄 Diagnostic Clinical Assessment Report
- **Risk Level Status Banners**: Distinct visual styling for **High Risk (Elevated Probability)** and **Low Risk (Normal Profile)**.
- **Evaluated Biomarkers Table**: Compares patient input values directly against standard clinical reference ranges with dynamic status chips (*Within Range*, *Overweight*, *Obese*, *Elevated*, *Borderline High*).
- **Print & Export Ready**: Integrated browser print functionality (`window.print()`) with print-optimized CSS that cleans up navigation and background styles.

### 7. 📊 High-Resolution Visual Analytics
- **Dark-Themed Correlation Scatter Plot**: Matched to the smoked obsidian theme, analyzing reproductive history (`Pregnancies`) against serum glucose concentration (`Glucose`).
- **Outcome Hue Differentiation**: Emerald points represent non-diabetic cases (`0`), while Coral points highlight positive risk cases (`1`).
- **Clinical Trend Cards**: Explanatory insights on glycemic trends, gestational risk factors, and multi-dimensional split dynamics.

---

## 🔄 System Architecture & Workflow

```mermaid
flowchart LR
    A["Patient / CSV Data Ingestion"] --> B["Data Validation & Preview"]
    B --> C["Train Random Forest Classifier (100 Trees)"]
    C --> D["Persist Artifact (models/model.pkl)"]
    D --> E["Patient Risk Inference (/predict)"]
    E --> F["Diagnostic Report (/result)"]
    B --> G["Exploratory Data Analysis (/visualize)"]
```

1. **Ingest**: Clinician uploads a CSV file or uses the default workspace dataset.
2. **Train**: Scikit-Learn splits features ($X$) and target ($y$) to fit 100 decision trees.
3. **Persist**: The fitted model is stored as a portable artifact in `models/model.pkl`.
4. **Predict**: Real-time patient physiological metrics are passed through the model.
5. **Diagnose**: Diagnostic outcomes and clinical recommendations are rendered with status indicators.
6. **Visualize**: Seaborn & Matplotlib render high-resolution scatter plots saved to `Static/images/scatter_plot.png`.

---

## 🧬 Biomarkers & Clinical Dataset

The application utilizes the **Pima Indians Diabetes Database** featuring 8 clinical parameters:

| Parameter | Form Name | Unit | Standard Normal Range | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Pregnancies** | `Pregnancies` | count | 0 &ndash; 20 | Number of times pregnant (0 for males/nulliparous) |
| **Glucose** | `Glucose Level` | mg/dL | 70 &ndash; 140 mg/dL | Fasting plasma glucose concentration |
| **Blood Pressure** | `Blood Pressure` | mm Hg | 60 &ndash; 85 mm Hg | Diastolic arterial blood pressure |
| **Skin Thickness** | `Skin Thickness` | mm | 10 &ndash; 50 mm | Triceps skin fold subcutaneous fat thickness |
| **Insulin** | `Insulin Level` | µU/mL | 16 &ndash; 166 µU/mL | 2-Hour post-load serum insulin concentration |
| **BMI** | `BMI` | kg/m² | 18.5 &ndash; 24.9 kg/m² | Body Mass Index (weight in kg / height in m²) |
| **Pedigree** | `Diabetes Pedigree Function` | score | 0.0 &ndash; 2.5 | Genetic scoring based on family diabetes history |
| **Age** | `Age` | years | 5 &ndash; 100 | Chronological biological patient age in years |
| **Outcome** | *(Target)* | binary | `0` or `1` | `0` = No Diabetes Detected, `1` = Diabetes Detected |

---

## 📁 Project Directory Structure

```text
ML app/
├── app.py                      # Core Flask application, DB models, routing & ML pipeline
├── diabetes.csv                # Primary clinical dataset (768 patient records)
├── README.md                   # Comprehensive project documentation
├── models/
│   └── model.pkl               # Serialized trained Random Forest classifier
├── uploads/
│   └── diabetes.csv            # File upload storage repository
├── Static/
│   ├── css/
│   │   └── style.css           # Bio-Sunset Coral & Smoked Obsidian stylesheet
│   └── images/
│       ├── scatter_plot.png    # High-resolution clinical correlation plot
│       └── ...                 # Platform static visual assets
└── templates/
    ├── base.html               # Master layout, responsive navbar & sticky footer
    ├── index.html              # Landing page with hero banner & workflow steps
    ├── dashboard.html          # Clinical command center & KPI metrics
    ├── login.html              # Glassmorphic user sign-in view
    ├── register.html           # New user registration view
    ├── upload.html             # Drag & drop dataset upload & schema table preview
    ├── train.html              # Model retraining confirmation & specifications
    ├── predict.html            # 2-Column risk prediction form with 1-click presets
    ├── result.html             # Patient clinical assessment & diagnostic report
    └── visualize.html          # Exploratory visual analytics & scatter distribution
```

---

## 🛠️ Technology Stack

- **Backend Architecture**: [Python 3](https://www.python.org/) & [Flask](https://flask.palletsprojects.com/)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (`RandomForestClassifier`), [Joblib](https://joblib.readthedocs.io/)
- **Data Engineering**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
- **Data Visualization**: [Matplotlib](https://matplotlib.org/) (`Agg` thread-safe headless backend), [Seaborn](https://seaborn.pydata.org/)
- **Database & Security**: [SQLAlchemy](https://www.sqlalchemy.org/) (`Flask-SQLAlchemy`), [Bcrypt](https://pypi.org/project/Flask-Bcrypt/) (`Flask-Bcrypt`), [PyMySQL](https://pypi.org/project/PyMySQL/)
- **User Interface**: HTML5, Modern CSS3 with Glassmorphism, [Bootstrap 5.3](https://getbootstrap.com/), [Bootstrap Icons 1.11](https://icons.getbootstrap.com/), Google Fonts (*Plus Jakarta Sans* & *JetBrains Mono*)

---

## 🚀 Installation & Quickstart

### 1. Prerequisites
Ensure you have **Python 3.9+** installed on your system.

### 2. Clone or Navigate to the Project Directory
```powershell
cd "c:\Users\ML Intership\ML app"
```

### 3. (Optional) Create and Activate a Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies
Install all required libraries:
```powershell
pip install flask flask-sqlalchemy flask-bcrypt pandas scikit-learn joblib numpy matplotlib seaborn pymysql
```

### 5. Launch the Application
```powershell
python app.py
```

### 6. Access in Browser
Navigate to your local browser:
```text
http://127.0.0.1:5000
```
*(or `http://localhost:5000`)*

---

## 🗄️ Database Configuration

The application is configured with **dual-mode database resilience** in `app.py`:

```python
default_mysql = 'mysql+pymysql://root:root@localhost/ml_app'
configured_db = os.environ.get('DATABASE_URL', default_mysql)
```

1. **MySQL Mode (Default Production / Local Service)**:
   - If MySQL server is running locally on port 3306 with credentials `root:root` and database `ml_app`, the application connects automatically.
2. **Zero-Config SQLite Fallback (Standalone / Sandbox)**:
   - If MySQL is not running or socket permissions are blocked, the application automatically switches to a local SQLite database (`ml_app.db`).
   - No crashes, no manual configuration required.

---

## 🗺️ Application Route Map

| HTTP Method | Route | Description | Template |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Platform landing page with hero CTA & workflow | `index.html` |
| `GET`, `POST` | `/register` | User account creation & password hashing | `register.html` |
| `GET`, `POST` | `/login` | User credential verification & session initiation | `login.html` |
| `POST` | `/logout` | Clears user session and redirects to sign-in | Redirect to `/login` |
| `GET` | `/dashboard` | Executive KPI overview & workflow navigation hub | `dashboard.html` |
| `GET`, `POST` | `/upload` | Drag-and-drop CSV upload & schema preview | `upload.html` |
| `GET`, `POST` | `/train` | Retrains Random Forest model & saves `model.pkl` | `train.html` |
| `GET`, `POST` | `/predict` | Patient biomarker form & inference execution | `predict.html` |
| `POST` (render) | `/result` | Diagnostic report & biomarker evaluation table | `result.html` |
| `GET`, `POST` | `/visualize` | Generates & displays high-res scatter plot | `visualize.html` |

---

## 🎯 Clinical Assessment & Presets

During testing, you can rapidly test both diagnostic pathways using the built-in preset buttons on the `/predict` page:

### 🟢 Low Risk Sample Preset
- **Pregnancies**: `1`
- **Glucose Concentration**: `85 mg/dL`
- **Diastolic Blood Pressure**: `66 mm Hg`
- **Skin Thickness**: `23 mm`
- **Serum Insulin**: `94 µU/mL`
- **BMI**: `22.4 kg/m²`
- **Pedigree Score**: `0.235`
- **Age**: `24 years`
- **Expected Outcome**: `No Diabetes Detected (Low Risk Profile)`

### 🔴 High Risk Sample Preset
- **Pregnancies**: `6`
- **Glucose Concentration**: `168 mg/dL`
- **Diastolic Blood Pressure**: `88 mm Hg`
- **Skin Thickness**: `35 mm`
- **Serum Insulin**: `180 µU/mL`
- **BMI**: `36.5 kg/m²`
- **Pedigree Score**: `0.825`
- **Age**: `54 years`
- **Expected Outcome**: `Diabetes Risk Detected (Elevated Clinical Probability)`

---

## 👨‍💻 Author & Credits

- **Developer**: Akshay
- **Project Role**: Machine Learning Internship Project
- **Application Version**: 2.0 (Modern Clinical Bio-Sunset Edition)
- **Year**: 2026

> [!NOTE]
> **Clinical Disclaimer**: *DiabeTech AI is an educational machine learning and clinical decision-support tool. It is intended to assist medical analysis and screening workflows and does not replace formal clinical diagnostic laboratory confirmation.*

