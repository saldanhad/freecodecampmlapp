# freecodecamp
* demo on Heroku: https://freecodecampwebapp.herokuapp.com/


# Steps to run the files locally


* Clone the repository by running the following command:
git clone https://github.com/saldanhad/freecodecampmlapp

* Get your Youtube API key, by creating your project in your GCP environment, refer to video tutorial by Thu Vu, next generate two databases for the sparta and troy systems respectively in Deta and get the API Key for the same from the Deta dashboard, refer to video tutorial by Sven, enter these details in a .env file.

* Install the required Python packages by running the following command: pip install -r requirements.txt

* Run the code by executing the following file in the Python interpreter: streamlit run Dashboard.py

# Database via Elephant SQL
* Create tables to track feedback for each of the models, table names - feedback_trackertroy, feedback_trackersparta
  
![Image Description](https://github.com/saldanhad/freecodecampmlapp/blob/master/elephantsqlss.jpg)

### References & Citations

* Vu, Thu[Thu Vu]. (2022, Jan 22).Youtube API for Python: How to Create a Unique Data Portfolio Project.Youtube.
https://www.youtube.com/watch?v=D56_Cx36oGY

* Sven[Coding is Fun].(2022,Jun 26).Build A Streamlit Web App From Scratch (incl. NoSQL Database + interactive Sankey chart).Youtube.
https://www.youtube.com/watch?v=3egaMfE9388

 * reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "http://arxiv.org/abs/1908.10084",

