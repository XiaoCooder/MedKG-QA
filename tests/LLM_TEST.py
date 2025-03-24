from openai import OpenAI

api_key = "sk-bd9f56da3a114296b3de026382a4827c"
api_base = "https://api.deepseek.com/v1"
client = OpenAI(api_key=api_key, base_url=api_base)

completion = client.chat.completions.create(
  model="deepseek-chat",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你是谁？"}
  ]
)

print(completion.choices[0].message)