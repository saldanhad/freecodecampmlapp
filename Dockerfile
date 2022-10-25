FROM python:3.8
CMD mkdir /Dashboard
COPY . /Dashboard
WORKDIR /Dashboard
EXPOSE 8501
RUN pip3 install -r requirements.txt
CMD streamlit run Dashboard.py --server.port $PORT