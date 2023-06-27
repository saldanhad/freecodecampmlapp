FROM python:3.10
RUN python -m pip install --upgrade pip && \
    mkdir /Dashboard
COPY . /Dashboard
WORKDIR /Dashboard
EXPOSE 8501
RUN pip install -r requirements.txt
ENTRYPOINT ["streamlit", "run"]
CMD ["Dashboard.py", "--server.port", "$PORT"]
