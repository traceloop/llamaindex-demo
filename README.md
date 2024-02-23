## LLamaIndex + OpenLLMetry observability demo

1. Install:

```python
poetry install
```

2. Set up the environment:

```python
export OPENAI_API_KEY=<your openai api key>
export TRACELOOP_API_KEY=<your traceloop api key>
```

You can connect OpenLLMetry to other platform by following one of the guides [here](https://www.traceloop.com/docs/openllmetry/integrations/introduction).

3. Run any of the demos like this:

```python
poetry run streamlit run llamaindex-demo/<demo name>.py
```

Demos:

1. `llamaindex-demo/chat.py` - An interactive chat allowing you to ask questions about Paul Graham's indexed essays.

   <img width="600" src="https://raw.githubusercontent.com/traceloop/llamaindex-demo/main/img/chat.png">
