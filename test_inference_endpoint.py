import requests

sentence = 'תגידי לי איזו ביצועים נתמכות על ידך?'
threshold = 0.6
input_data={"data":[sentence, threshold]}

scoring_uri = "http://20.122.242.229:80/api/v1/service/robohon-similarity-algo/score"
prediction = requests.post(scoring_uri, json = input_data)

print("prediction:", prediction.text.encode().decode('unicode_escape'))