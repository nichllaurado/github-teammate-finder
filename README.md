Final Project for EN.601.466 Information Retrieval & Web Agents 
by Nicholas Llaurado


How to run:
1.Pip install -r requirements.txt
2.Python -m app.main

Optional OpenAI query generation:
- Set ``OPENAI_API_KEY to let iterative GitHub candidate search generate search phrases with the OpenAI Responses API.
- Set `OPENAI_QUERY_MODEL` to override the default query model.
- Set `OPENAI_QUERY_GENERATION=0` to force the local keyword fallback.
