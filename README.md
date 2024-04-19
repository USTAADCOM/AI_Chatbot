# AI Chatbot


## Introduction

This repository contains the API for AI chatbot. 

## Setup
  ```code
  conda create -n <env_name> python==3.10
  conda activate <env_name>
  git clone https://github.com/USTAADCOM/AI_Chatbot.git
  cd AI_Chatbot
  pip install -r requirements.txt -q
  ```
## create .env
create .env and put your key
```
OPENAI_API_KEY = "your sceret key here"
```
## Run API
```code
python3 app.py 
```
### http://127.0.0.1:5000/query
Payload
```code
{
    "data": "yourquestion or promt"
}
```
Response 
```code
{
    "answer": "LLM Response"
    "chat_history": [
        [
            "Your Question or prompt",
            "LLM resposne"
        ]
    ]
}
```

###  http://127.0.0.1:5000/refresh
Payload
```code
{
    "data": "OK"
}
```
Response 
```code
{
    "message": "success"
}
```
