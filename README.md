# Modeling-and-Comparison-of-Computational-Machines

NOTE::: Python must be installed in your computer

STEP 1: CHECK PYTHON
Open terminal and run:
  python --version

If not installed: Download from https://www.python.org/downloads/

STEP 2: OPEN PROJECT IN VS CODE
- Open VS Code
- File → Open Folder → Select your project folder (Path must be inside TOC_Project (in which app.c ,requirements.txt present))
- Open terminal: Ctrl+` (or View → Terminal)

STEP 3: INSTALL DEPENDENCIES
Run in terminal:
  pip install -r requirements.txt

STEP 4: RUN APPLICATION
Run in terminal:
  streamlit run app.py

App will open at http://localhost:8501

TROUBLESHOOTING:
- If streamlit not found: python -m streamlit run app.py
- To stop: Press Ctrl+C in terminal

VALID INPUT SYMBOLS:
- DFA: 0, 1
- NFA: a, b
- PDA: a, b
- TM: a, b, c
