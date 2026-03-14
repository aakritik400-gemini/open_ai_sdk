from openai import OpenAI

client = OpenAI(
    api_key="AIzaSyA0hlwMQ39OgL7o4iTJ2y1TJo4bztDcq9E",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

models = client.models.list()

for model in models.data:
    print(model.id)