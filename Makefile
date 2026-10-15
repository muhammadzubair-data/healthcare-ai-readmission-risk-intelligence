PY ?= python
export PYTHONPATH := src

.PHONY: fast full test app docs clean-artifacts

fast:
	$(PY) -m readmit.pipeline all --fast

full:
	$(PY) -m readmit.pipeline all

docs:
	$(PY) -m readmit.pipeline docs

test:
	$(PY) -m pytest -q

app:
	streamlit run app/streamlit_app.py
