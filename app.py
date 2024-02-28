import streamlit as st
from langchain.vectorstores.utils import DistanceStrategy
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import BigQueryVectorSearch
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import VertexAI


### Patch BigQueryVectorSearch
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
import json

DEFAULT_TOP_K = 4  # default number of documents returned from similarity search

class FixedBigQueryVectorSearch(BigQueryVectorSearch):
    def _search_with_score_and_embeddings_by_vector(
        self,
        embedding: List[float],
        k: int = DEFAULT_TOP_K,
        filter: Optional[Dict[str, Any]] = None,
        brute_force: bool = False,
        fraction_lists_to_search: Optional[float] = None,
    ) -> List[Tuple[Document, List[float], float]]:
        from google.cloud import bigquery

        # Create an index if no index exists.
        if not self._have_index and not self._creating_index:
            self._initialize_vector_index()
        # Prepare filter
        filter_expr = "TRUE"
        if filter:
            filter_expressions = []
            for i in filter.items():
                if isinstance(i[1], float):
                    expr = (
                        "ABS(CAST(JSON_VALUE("
                        f"base.`{self.metadata_field}`,'$.{i[0]}') "
                        f"AS FLOAT64) - {i[1]}) "
                        f"<= {sys.float_info.epsilon}"
                    )
                else:
                    val = str(i[1]).replace('"', '\\"')
                    expr = (
                        f"JSON_VALUE(base.`{self.metadata_field}`,'$.{i[0]}')"
                        f' = "{val}"'
                    )
                filter_expressions.append(expr)
            filter_expression_str = " AND ".join(filter_expressions)
            filter_expr += f" AND ({filter_expression_str})"
        # Configure and run a query job.
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ArrayQueryParameter("v", "FLOAT64", embedding),
            ],
            use_query_cache=False,
            priority=bigquery.QueryPriority.BATCH,
        )
        if self.distance_strategy == DistanceStrategy.EUCLIDEAN_DISTANCE:
            distance_type = "EUCLIDEAN"
        elif self.distance_strategy == DistanceStrategy.COSINE:
            distance_type = "COSINE"
        # Default to EUCLIDEAN_DISTANCE
        else:
            distance_type = "EUCLIDEAN"
        if brute_force:
            options_string = ",options => '{\"use_brute_force\":true}'"
        elif fraction_lists_to_search:
            if fraction_lists_to_search == 0 or fraction_lists_to_search >= 1.0:
                raise ValueError(
                    "`fraction_lists_to_search` must be between " "0.0 and 1.0"
                )
            options_string = (
                ',options => \'{"fraction_lists_to_search":'
                f"{fraction_lists_to_search}}}'"
            )
        else:
            options_string = ""
        query = f"""
            SELECT
                base.*,
                distance AS _vector_search_distance
            FROM VECTOR_SEARCH(
                TABLE `{self.full_table_id}`,
                "{self.text_embedding_field}",
                (SELECT @v AS {self.text_embedding_field}),
                distance_type => "{distance_type}",
                top_k => {k}
                {options_string}
            )
            WHERE {filter_expr}
            LIMIT {k}
        """
        document_tuples: List[Tuple[Document, List[float], float]] = []
        # TODO(vladkol): Use jobCreationMode=JOB_CREATION_OPTIONAL when available.
        job = self.bq_client.query(
            query, job_config=job_config, api_method=bigquery.enums.QueryApiMethod.QUERY
        )
        # Process job results.
        for row in job:
            metadata = row[self.metadata_field]
            if metadata:
                metadata = json.loads(json.dumps(metadata))
            else:
                metadata = {}
            metadata["__id"] = row[self.doc_id_field]
            metadata["__job_id"] = job.job_id
            doc = Document(page_content=row[self.content_field], metadata=metadata)
            document_tuples.append(
                (doc, row[self.text_embedding_field], row["_vector_search_distance"])
            )
        return document_tuples
### Patch BigQueryVectorSearch




PROJECT_ID = "derrick-doit-sandbox"
REGION = "US"
DATASET = "vector_search"
TABLE = "doc_and_vectors"

llm = VertexAI(model_name="gemini-pro", temperature=0)

embedding = VertexAIEmbeddings(
    model_name="textembedding-gecko@001", project=PROJECT_ID
)

store = FixedBigQueryVectorSearch(
    project_id=PROJECT_ID,
    dataset_name=DATASET,
    table_name=TABLE,
    location=REGION,
    embedding=embedding,
    distance_strategy=DistanceStrategy.EUCLIDEAN_DISTANCE,
)

template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If you don't know the answer, just say that you don't know.

Follow exactly those 4 steps:
1. Read the context below and aggregrate this data
Context : {context}
2. Answer the question using only this context, have detailed explanation
3. Show all the source URL for your answers
4. The answer should be in following format. Keep an eye on the changeline and don't truncate the link:

**Question**: {question}
\n**Answer**:
\n**Source**:
"""

prompt = ChatPromptTemplate.from_template(template)

def generate_response(question):
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    result = qa_chain({"query": question})
    st.info(result["result"])

st.title('🦜🔗 Retrieval Augmented Generation with Google Gemini and BigQuery')

with st.form('my_form'):
    text = st.text_area('Enter text:', 'How to remote debug Flutter?')
    submitted = st.form_submit_button('Submit')
    if submitted:
        generate_response(text)