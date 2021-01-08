FROM python:3.7

COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
RUN pip install .
CMD ["jupyter", "notebook", "--NotebookApp.token='password'", "--ip=0.0.0.0", "--allow-root"]