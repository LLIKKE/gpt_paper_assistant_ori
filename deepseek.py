
from openai import OpenAI

OAI_KEY = "sk-7f955bffa0a9492f99e0579756ba321b"
openai_client = OpenAI(api_key=OAI_KEY, base_url="https://api.deepseek.com/v1")


context=openai_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "你是谁"}],
        temperature=0,
        seed=0
    )


print(context)
out_text = context.choices[0].message.content
print(out_text)
# Please install OpenAI SDK first: `pip3 install openai`

from openai import OpenAI

client = OpenAI(api_key=OAI_KEY, base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)



