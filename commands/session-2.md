Create Py Env
    python -m venv venv

Activate Env
    venv/Scripts/activate

Create Model
    python .\model-demo\model.py

Run Streamlit App (make sure .pkl is present)
    cd model-demo
    streamlit run .\loan_app.py
