# Study Notes RAG

A small Chainlit app that accepts JSON study notes, indexes them in an isolated
in-memory Chroma collection for the current chat, and answers questions using
Gemini with Vertex AI embeddings.

## Setup

Requirements: Python 3.13 or newer, `uv`, a Google Gemini API key, a Google
Cloud project with Vertex AI access, and Google Application Default Credentials
(ADC) with permission to use Vertex AI.

From the repository root:

```sh
cd hw2
uv sync
cp .env.example .env
```

Edit `.env` locally and replace the placeholders. The required variables are
`GOOGLE_MODEL`, `GOOGLE_API_KEY`, and `GOOGLE_CLOUD_PROJECT`. The embedding
location defaults to `us-west1`; set `GOOGLE_CLOUD_LOCATION` to override it.
Never commit `.env` or credential files.

For local Vertex AI authentication, set up ADC in a terminal:

```sh
gcloud auth application-default login
```

Alternatively, set `GOOGLE_APPLICATION_CREDENTIALS` to a secured credential
configuration file path in your local environment. Do not place credential
contents in this repository. Ensure the selected identity has access to Vertex
AI and the project has the API enabled.

## Run

From `hw2/`, launch with:

```sh
uv run chainlit run app.py
```

Open the local URL printed by Chainlit, upload a JSON file, then ask a question
about its notes. Uploaded notes and the Chroma collection remain in memory for
the active chat and are released when Chainlit ends that chat.

## JSON format

The top-level value must be a nonempty JSON array. Every item must be an object
with nonempty string `title` and `content` fields. Extra fields are ignored.

```json
[
  {
    "title": "Example topic",
    "content": "A short explanation of the topic."
  }
]
```

`sample_notes.json` is a ready-to-upload example. Each note becomes a LangChain
document. The loader keeps its filename, title, and zero-based note index in
metadata. Retrieval uses that metadata to show note titles, filenames, and
supporting excerpts with each answer.

## Course code adapted

- `02_LangChain/07_RAG/07_rag_loaddb.py`: adapted recursive text splitting,
  document metadata, and Chroma indexing for JSON note documents.
- `02_LangChain/07_RAG/08_rag_docsearch.py`: adapted similarity retrieval and
  source attribution for each question.
- `02_LangChain/07_RAG/10_chainlit_rag_query.py`: adapted the Chainlit handlers,
  local prompt, Gemini chat model, and Vertex AI `gemini-embedding-001` setup.

The app uses `ChatGoogleGenerativeAI` with `GOOGLE_MODEL` and `GOOGLE_API_KEY`,
and `VertexAIEmbeddings` with `GOOGLE_CLOUD_PROJECT`, ADC, and the configured
location. These follow the provider choices in the course examples.

## Known limitations

- Only JSON arrays of notes with `title` and `content` are supported.
- This is a single-file upload per chat; start another chat to index a different
  file. Collections are in memory and are not durable across app restarts.
- The app shows the retrieved passages but does not guarantee factual accuracy;
  review answers against their excerpts.
- Embedding and chat requests require configured Google services and may incur
  usage charges. The local loader/import checks do not call those APIs.
