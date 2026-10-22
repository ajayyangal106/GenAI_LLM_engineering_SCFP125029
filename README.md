# GENAI_LLM_Engineering_SCFP125029

**Student:** Ajay Yangal         
**PRN:** SCFP125029

## Project
Generative AI / LLM Engineering lab activities and assignments.

## How to Run

### 1. Open the project
Open this folder in VS Code.

### 2. Create and activate a virtual environment
Windows PowerShell:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
If PowerShell blocks activation, run the Python files with `venv\Scripts\python.exe`, or use Command Prompt:
```cmd
venv\Scripts\activate
```

### 3. Install dependencies
The Python exercises use packages such as `groq`, `python-dotenv`, `langchain-groq`, `langchain-core`, `pydantic`, and `requests`. Install them with:
```bash
pip install groq python-dotenv langchain-groq langchain-core pydantic requests
```
For the RNN/LSTM notebooks, install the notebook/data-science packages as needed:
```bash
pip install jupyter notebook pandas numpy scikit-learn matplotlib torch
```

### 4. Configure the Groq API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
Do not commit the API key to GitHub.

### 5. Run Lab 3 / Lab 5 Python exercises
From the project root, for example:
```bash
python Lab3/exercise1_doctor_summary_generator.py
python Lab3/exercise2_review_classifier.py
python Lab3/exercise3_resume_field_extractor.py
python Lab3/exercise4_customer_support_reply_generator.py
python Lab3/exercise5_multi-language_translator.py
python Lab3/exercise6_iterative_content_refiner.py

python Lab5/exercise1_customer_review.py
python Lab5/exercise2_research_paper_abstract_to_layperson_summary_pipeline.py
python Lab5/exercise3_Interview_transcript_to_candidate_scorecard_pipeline.py
python Lab5/exercise4_outdoor_event_plan_to_weather_adjusted_recommendation_pipeline.py
python Lab5/exercise5_customer_support_chat_log_to_escalation_report_pipeline.py
python Lab5/exercise6_workout_goal_to_weekly_plan_pipeline.py
```

### 6. Run Jupyter notebooks
```bash
jupyter notebook
```
Then open `Activity_1/Assignment_1.ipynb`, `Activity_2/Assignment_2.ipynb`, `groq_tutorial.ipynb`, or `langchain.ipynb` and run the cells.

**Note:** The Groq-based programs require a valid `GROQ_API_KEY` and internet access.

---

## Original README Content

**Student:** Ajay Yangal    
**PRN:** SCFP125029

# GENAI_LLM_Engineering_SCFP125029
