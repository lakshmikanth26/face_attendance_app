@echo off

:: Step 1: Create Virtual Environment
python -m venv venv

:: Step 2: Activate Virtual Environment
call .\venv\Scripts\activate

:: Step 3: Install Dependencies
pip install -r requirements.txt
