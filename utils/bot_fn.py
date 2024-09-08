def rewrite_retrieval_read(query):
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_openai import ChatOpenAI

    template = """Answer the users question based only on the following context:

    <context>
    {context}
    </context>

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0)

    search = DuckDuckGoSearchAPIWrapper()

    def retriever(query):
        return search.run(query)

    template = """Provide a better search query for \
    web search engine to answer the given question, end \
    the queries with ’**’. Question: \
    {x} Answer:"""
    rewrite_prompt = ChatPromptTemplate.from_template(template)

    def _parse(text):
        return text.strip('"').strip("**")

    rewriter = rewrite_prompt | ChatOpenAI(temperature=0) | StrOutputParser() | _parse

    rewrite_retrieve_read_chain = (
        {
            "context": {"x": RunnablePassthrough()} | rewriter | retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )

    return rewrite_retrieve_read_chain.invoke(query)

if __name__ == "__main__":
    from langchain.globals import set_debug, set_verbose
    set_debug(True)
    set_verbose(False)
    rewrite_retrieval_read("What is langchain?")