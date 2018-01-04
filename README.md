# Week_4_Sous_Do_Chef
Using NLP to help people cook and topic modeling.

Scraped reddit's /r/askculinary and r/cooking for top question-response data and created a specialized bot based off topics analyzed. 25 response topics were created using Non-Negative Matrix Factorization (NMF), which nicely fit into categories such as cast iron, steak, sauces, and chicken, and were used as filters to quickly query the closest response to a question. 
Turned the results into a Flask App which works by determining the closest topic the question pertains to and searching the closest 3 answers within that topic through cosine similarity scores. 

![Flask App](https://raw.githubusercontent.com/leangdamang/Week_4_Sous_Do_Chef/master/bot.png)
