# Intro

This is a repo for my talk at GDG Melbourne in Feb 2024.

## Demos

### Fine-tune text-bison

_At the time of writing, Gemini model on Vertex AI cannot be fine-tuned._

In this demo, I generated a dataset in jsonl format [file](gemini.jsonl) and follow the Vertex AI document [here](https://cloud.google.com/vertex-ai/generative-ai/docs/models/tune-text-models-supervised) to tune the model.

### Retrieval Augmented Generation with Google Gemini and BigQuery

In this demo, I used [LangChain](https://www.langchain.com/) to build a Retrieval Augmented Generation(RAG) with Google Gemini pro and BigQuery.
You can follow this [Jupyter Notebook](Retrieval-augmented-generation-with-Gemini-and-BigQuery.ipynb) to understand how it works.


To build a web app, I used [streamlit](https://streamlit.io/).
To deploy it to Cloud Run, just clone the repo and follow the below steps:
- `gcloud auth login`
- `gcloud config set project your-project-id`
- `gcloud run deploy`

### Generative AI Chatbot with DialogFlow CX

#### Why?

Limitation of Chatbot on the market:
- Limited ability to understand human language
- Inability to handle complex issues
- Inability to provide a human touch

What generative AI brings:
- Nature Language Understanding (NLU)
- More human-like and intelligent interactions
- Direct users to the predefined flow

#### How to test it

- Follow this Google [document](https://cloud.google.com/dialogflow/cx/docs/quick/build-agent#create-agent) to create a new DialogFlow CX agent.
- Follow this Google [document](https://cloud.google.com/dialogflow/cx/docs/concept/agent#export) to restore the backup from [Diverbooker.zip](Diverbooker.zip) or [Diverbooker-GenAI.zip](Diverbooker-GenAI.zip)
