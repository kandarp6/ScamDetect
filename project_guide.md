# Graphura: Fake Job Detector Project Guide

This guide helps you understand how the project works, how each file contributes to the system, and how to run it step-by-step.

---

## 🚀 How to Run the Project (Step-by-Step)

To run this application on your computer, follow these simple steps:

### Step 1: Open Terminal/PowerShell
Open **PowerShell** or **Command Prompt** and navigate to your project directory:
```powershell
cd "d:\Madhav_Gagneja\INTERNSHIP\GRAPHURA DATA SCIENCE & AI\Projects\Fake_Internship_&_Job_Scam_Detection_System\Graphura\Graphura"
```

### Step 2: Activate the Virtual Environment
Activate the pre-configured Python environment (this ensures the project uses its own private folder of tools instead of messing with your computer's global Python settings):
```powershell
.\venv\Scripts\Activate.ps1
```
*(If you get a permission restriction error, run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell first, then try activating it again).*

### Step 3: Run the Unified Application
Run the master starter file `app.py`:
```powershell
python app.py
```
You will see output logs showing that the server has started successfully.

### Step 4: Open the Website in Your Browser
Open your web browser (Chrome, Edge, or Firefox) and go to:
- **Frontend App**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 📂 File-by-File Breakdown (For a 10th Standard Student)

Think of this project as a **smart fraud detection agency**. It consists of a **Frontend** (what the user sees, like the office lobby) and a **Backend** (the laboratory and detectives working in the background). 

Here is what every single file in the project does:

### 1. Root Files (The Master Switch)
* **app.py**: **The Ignition Key**. This is the file you run. It starts the backend web-server software (called Uvicorn) and mounts the frontend webpage so everything turns on at once.

---

### 2. The Backend (The Brains & Detectives)
* **backend/main.py**: **The Head Receptionist (API)**. This file listens to whatever you type on the website (e.g., pasting a job description or URL), routes that information to our AI models or web-scrapers, gathers the results, and sends them back to the website.
* **backend/requirements.txt**: **The Shopping List**. A simple list of all the Python toolboxes (libraries) that need to be installed for the code to run (like FastAPI, Scikit-Learn, and Playwright).
* **backend/.env**: **The Secret Vault**. A settings file that stores secret keys (like database passwords) and general settings (like how fast to scrape or what search keywords to use) so they are kept separate from the public code.

#### 🧠 Machine Learning (ML) Subdirectory (`backend/ml`)
* **nlp_engine.py**: **The English Teacher (NLP)**. NLP stands for *Natural Language Processing*. This file takes a messy job description, removes grammar punctuation/links/filler words, and looks for suspicious words (like "WhatsApp HR", "Earn Daily", or "Processing Fee").
* **feature_extractor.py**: **The Translator**. Computer algorithms cannot read English words directly; they only understand numbers. This file translates cleaned text and parameters (like salary or location) into mathematical values.
* **predict.py**: **The Jury**. It loads three different pre-trained AI models (XGBoost, Random Forest, and Isolation Forest), takes their individual votes on a job's fraud risk, and calculates a final risk score.
* **train_models.py**: **The School Camp**. This script runs only when we want to teach our AI models. It takes thousands of labeled examples of "Real" and "Fake" jobs and trains the AI models to spot the differences.

#### 🕷️ Scraper & Verification Engines (`backend/scraper/engines`)
* **url_scraper.py**: **The Digital Spy**. If a user inputs a job URL (like Internshala or LinkedIn), this script secretly opens a browser in the background, extracts the title, company name, salary, and description, and brings them back.
* **recruiter_verifier.py**: **The Identity Auditor**. Checks if the recruiter's name is too generic (like "HR Admin"), checks if they use personal emails (like `@gmail.com` instead of `@google.com`), and validates their LinkedIn page.
* **salary_parser.py**: **The Math Auditor**. Analyzes the salary terms. If a job offers "Rs. 50,000 per day for simple clicking", this script marks it as "highly suspicious salary".
* **location_normalizer.py**: **The Geography Expert**. Standardizes location spelling (e.g. turning "Bengaluru Urban, Karnataka" into "Bangalore, India") so we don't have duplicate locations.
* **company_trust.py**: **The Registrar**. Checks if a hiring company is a registered enterprise, checks their website domain age, and computes an initial trust level.
* **skill_extractor.py**: **The Resume Matcher**. Scans what skills are required for the job and flags mismatch anomalies.
* **scoring_engine.py**: **The Report Card Maker**. Combines recruiter trust, company trust, salary anomalies, and keyword alerts to formulate a single scam score report.
* **deduplicator.py**: **The Anti-Copy Guard**. Creates a digital fingerprint of every job posting. If the system has already analyzed this job description, it skips it to save energy.

#### 🗄️ Database Storage (`backend/scraper/storage`)
* **supabase_client.py**: **The Safe Vault Manager**. Responsible for connecting to Supabase (a database in the cloud) to save jobs, recruiters, and user-submitted scam reports.

---

### 3. The Frontend (The Website Lobby)
* **frontend/index.html**: **The Building Exterior**. This contains the HTML structure (buttons, inputs, pages, sidebars) and all the style sheets (colors, animations, fonts, layouts) that make the website look premium and beautiful.
* **frontend/js/config.js**: **The GPS**. Tells the website whether to look for the backend on your own laptop (`localhost`) or online on Render.
* **frontend/js/api.js**: **The Postman**. A helper script that picks up your search request from the browser, runs it to the backend server, waits for the result, and brings it back.
* **frontend/js/app.js**: **The Site Manager**. Listens to button clicks, handles page transitions (e.g., from "Scam Checker" to "Dashboard"), updates stats counters, and draws the color-coded circular risk ring.
