import requests

sentence = 'תגידי לי איזו ביצועים נתמכות על ידך?'
threshold = 0.6
input_data={"data":[sentence, threshold]}



scoring_uri = "http://20.62.153.81:80/api/v1/service/alephbert-deployed/score"
prediction = requests.post(scoring_uri, json = input_data)

print("prediction:", prediction.text.encode().decode('unicode_escape'))