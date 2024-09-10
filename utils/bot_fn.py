def rewrite_retrieval_read(query, retriever=None, model=None, res={}):
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

    if model is None:
        model = ChatOpenAI(temperature=0)

    def _retriever(query):
        search = DuckDuckGoSearchAPIWrapper()
        return search.run(query)
    
    retriever = _retriever if retriever is None else retriever

    template = """Provide a better search query for \
    web search engine to answer the given question, end \
    the queries with ’**’. Question: \
    {x} Answer:"""
    rewrite_prompt = ChatPromptTemplate.from_template(template)

    def _parse(text):
        return text.strip('"').strip("**")
    
    res['input'] = query

    def log_output(key):
        """Log generator for LangChain intermediate results."""
        def inner(context):
            res[key] = context
            return context
        return inner

    rewriter = rewrite_prompt | model | StrOutputParser() | _parse | log_output('rewrite')

    rewrite_retrieve_read_chain = (
        {
            "context": {"x": RunnablePassthrough()} | rewriter | retriever,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
        | StrOutputParser()
    )
    res['output'] = rewrite_retrieve_read_chain.invoke(query)
    return res['output']

if __name__ == "__main__":
    from langchain.globals import set_debug, set_verbose
    set_debug(True)
    set_verbose(False)
    rewrite_retrieval_read("What is langchain?")