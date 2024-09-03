PROMPT_CHECK_DELIVERY_DATE = """
You are a helpful customer support assistant. Use the supplied tools to assist the user.
"""

PROMPT_RAG = """Use the following context and chat history to answer user's question.

Context:

{% for doc in docs %}

----

{{ doc.page_content }}
{% endfor %}
"""