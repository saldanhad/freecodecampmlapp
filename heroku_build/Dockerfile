FROM python:3.10
RUN python -m pip install --upgrade pip
RUN mkdir /Dashboard
COPY . /Dashboard
WORKDIR /Dashboard
EXPOSE 8501
RUN pip install -r requirements.txt
CMD streamlit run Dashboard.py --server.port $PORT
