# BGU CS Course 'NLP with LLMs' - Spring 2025 - Michael Elhadad - HW1
## April 2025

This repository contains instructions for Assignment 1 in the course.

To setup your environment, the prerequisites are:
* python (> 3.11)
* git
* uv https://docs.astral.sh/uv/getting-started/)
* Visual studio code 

To setup the environment do:

1. Create a folder for the assignment: mkdir hw1; cd hw1
2. Retrieve the dataset we will use and the code from this repo:
    1. git clone https://github.com/UniversalDependencies/UD_English-EWT.git
    2. git clone https://github.com/melhadad/nlp-with-llms-2025-hw1.git
3. Load the required python libraries:
    1. cd nlp-with-llms-2025-hw1; uv sync
4. Define your API keys in either gemini_key.ini or grok_key.ini
    1. Define the environment variables in your shell - for example:
    ```
    # Unix like
    source grok_key.ini
    export GROK_API_KEY=$GROK_API_KEY
    ```
5. Activate the project virtual env: 
   ```
   source .venv\bin\activate
   ```
6. Open ud_pos_tagger_sklearn.ipynb in VS Code and verify you can execute the cells.
