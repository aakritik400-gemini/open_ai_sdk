from openai import OpenAI

client = OpenAI(
    api_key="",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

models = client.models.list()

for model in models.data:
    print(model.id)