import os
import streamlit as st
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    insert,
)

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    SQLDatabase,
    Settings,
)
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.core.node_parser import TokenTextSplitter
from llama_index.core.tools import QueryEngineTool

from llama_index.core.agent import ReActAgent
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow

Traceloop.init()

Settings.chunk_size = 1024
Settings.llm = OpenAI(model="gpt-4")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

text_splitter = TokenTextSplitter(chunk_size=1024)

st.set_page_config(
    page_title="LlamaIndex + OpenLLMetry demo",
    page_icon="🦙",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items=None,
)
st.title("LLamaIndex Agents Demo")
st.info("Ask me something!")

if "messages" not in st.session_state.keys():  # Initialize the chat messages history
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Which topic do you want to read about?",
        }
    ]


# Initialize a dummy SQL Database with some data
@st.cache_resource(show_spinner=False)
@workflow(name="load_sql_database")
def load_sql_database():
    if os.path.exists("database.db"):
        os.remove("database.db")

    engine = create_engine("sqlite:///database.db", future=True)
    metadata_obj = MetaData()

    table_name = "city_stats"
    city_stats_table = Table(
        table_name,
        metadata_obj,
        Column("city_name", String(16), primary_key=True),
        Column("population", Integer),
        Column("country", String(16), nullable=False),
    )
    metadata_obj.create_all(engine)

    rows = [
        {"city_name": "Toronto", "population": 2930000, "country": "Canada"},
        {"city_name": "Tokyo", "population": 13960000, "country": "Japan"},
        {"city_name": "Berlin", "population": 3645000, "country": "Germany"},
    ]
    for row in rows:
        stmt = insert(city_stats_table).values(**row)
        with engine.begin() as connection:
            connection.execute(stmt)

    sql_database = SQLDatabase(engine, include_tables=["city_stats"])

    return NLSQLTableQueryEngine(
        sql_database=sql_database,
        tables=["city_stats"],
    )


@st.cache_resource(show_spinner=False)
@workflow(name="load_wiki_data")
def load_wiki_data():
    with st.spinner(text="Loading and indexing Wikipedia data."):
        storage_context = StorageContext.from_defaults()
        vector_index = VectorStoreIndex([], storage_context=storage_context)

        cities = ["Toronto", "Berlin", "Tokyo"]
        wiki_docs = WikipediaReader().load_data(pages=cities)
        for city, wiki_doc in zip(cities, wiki_docs):
            nodes = text_splitter.get_nodes_from_documents([wiki_doc])
            # add metadata to each node
            for node in nodes:
                node.metadata = {"title": city}
            vector_index.insert_nodes(nodes)

        return vector_index


# Initialize the Agent
if "agent" not in st.session_state.keys():
    query_engine = load_sql_database()
    vector_index = load_wiki_data()

    sql_tool = QueryEngineTool.from_defaults(
        query_engine=query_engine,
        name="sql_tool",
        description=(
            "Useful for translating a natural language query into a SQL query over"
            " a table containing: city_stats, containing the population/country of"
            " each city"
        ),
    )
    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_index.as_query_engine(similarity_top_k=2),
        name="vector_tool",
        description=("Useful for answering semantic questions about different cities"),
    )

    st.session_state.agent = ReActAgent.from_tools(
        [sql_tool, vector_tool], verbose=True
    )


if prompt := st.chat_input("Topic"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.agent.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message)  # Add response to message history
