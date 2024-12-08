#!/bin/bash

# Step 1: Create Virtual Environment
python3 -m venv venv

# Step 2: Activate Virtual Environment
source venv/bin/activate

# Step 3: Install Dependencies
pip install -r requirements.txt
