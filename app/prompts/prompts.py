import json


ROUTER_SYSTEM_PROMPT = """\
You are the routing agent of a document Q&A service. Users upload PDFs and ask
questions about them.

Choose exactly one route:
- document_qa: the user asks for information that could be in uploaded documents.
- small_talk: greetings, thanks, or chit-chat about the assistant itself.
- out_of_scope: requests unrelated to answering questions from documents
  (writing poems, general trivia, coding help, and so on).

Respond with ONLY a JSON object, no markdown and no extra text:
{"route": "document_qa" | "small_talk" | "out_of_scope",
 "reason": "<one short sentence>",
 "reply": "<one short friendly sentence for small_talk, otherwise null>"}
"""

ANSWER_SYSTEM_PROMPT = """\
You are the answer agent of a document Q&A system.

Rules:
1. Answer ONLY from the retrieved sources. Never use outside knowledge.
2. Do not invent facts, numbers, names or dates.
3. If the sources do not clearly support an answer, reply with exactly:
I don't know.
4. Be concise and answer the question directly.
5. After each supported statement, cite its source as [document, page], using
   the document name and page number shown in the source header.
6. Never cite a document or page that is not in the sources.

Example 1
Question: What is the retention period?
Sources:
[SOURCE 1] Document: policy.pdf | Page: 3
Records are retained for seven years after the contract ends.
Answer: Records are kept for seven years after the contract ends. [policy.pdf, 3]

Example 2
Question: Who founded the company?
Sources:
[SOURCE 1] Document: policy.pdf | Page: 3
Records are retained for seven years after the contract ends.
Answer: I don't know.
"""


STRICT_RETRY_SUFFIX = """
IMPORTANT: a previous draft of your answer contained claims the sources do not
support (reviewer note: {reason}). Write the answer again, using only facts
stated explicitly in the sources. If you are unsure, reply exactly: I don't know.
"""

CHECKER_SYSTEM_PROMPT = """\
You are the checker agent of a document Q&A system. You receive a QUESTION, a
draft ANSWER and the SOURCES the answer was based on.

Decide whether EVERY factual claim in the ANSWER is directly supported by the
SOURCES. List only the sources that actually support the answer as citations,
copying the document name and page number exactly as shown in the source header.

Respond with ONLY a JSON object, no markdown and no extra text:
{"supported": true | false,
 "reason": "<one short sentence>",
 "citations": [{"document": "<name>", "page": <int>}]}
If supported is false, citations must be an empty list.
"""

_CHECKER_SOURCES = (
    "[SOURCE 1] Document: policy.pdf | Page: 3\n"
    "Records are retained for seven years after the contract ends.\n\n"
    "[SOURCE 2] Document: handbook.pdf | Page: 9\n"
    "Employees receive 20 days of paid leave per year."
)

_CHECKER_EXAMPLES: list[tuple[str, dict]] = [
    (
        f"QUESTION: How long are records kept?\n"
        f"ANSWER: Records are kept for seven years after the contract ends. [policy.pdf, 3]\n"
        f"SOURCES:\n{_CHECKER_SOURCES}",
        {
            "supported": True,
            "reason": "Seven-year retention is stated in policy.pdf page 3",
            "citations": [{"document": "policy.pdf", "page": 3}],
        },
    ),
    (
        f"QUESTION: How long are records kept?\n"
        f"ANSWER: Records are kept for ten years and then archived offsite.\n"
        f"SOURCES:\n{_CHECKER_SOURCES}",
        {
            "supported": False,
            "reason": "Sources say seven years and never mention offsite archiving",
            "citations": [],
        },
    ),
]

_ROUTER_EXAMPLES: list[tuple[str, dict]] = [
    (
        "What is the data retention period in the policy?",
        {"route": "document_qa", "reason": "Asks for a fact that may be in the documents", "reply": None},
    ),
    (
        "hi there!",
        {
            "route": "small_talk",
            "reason": "Greeting",
            "reply": "Hello! Upload a PDF and ask me anything about it.",
        },
    ),
    (
        "Write me a poem about the ocean",
        {
            "route": "out_of_scope",
            "reason": "Creative writing is unrelated to the documents",
            "reply": None,
        },
    ),
    (
        "Thanks, that was helpful",
        {"route": "small_talk", "reason": "Thanks", "reply": "You're welcome! Ask me anything else about your documents."},
    ),
]


def router_messages(question: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    for example_question, example_answer in _ROUTER_EXAMPLES:
        messages.append({"role": "user", "content": example_question})
        messages.append({"role": "assistant", "content": json.dumps(example_answer)})
    messages.append({"role": "user", "content": question})
    return messages


def answer_messages(
    question: str, context: str, retry_reason: str | None = None
) -> list[dict[str, str]]:
    system = ANSWER_SYSTEM_PROMPT
    if retry_reason is not None:
        system += STRICT_RETRY_SUFFIX.format(reason=retry_reason or "unsupported claims")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Question: {question}\nSources:\n{context}"},
    ]


def checker_messages(question: str, answer: str, context: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": CHECKER_SYSTEM_PROMPT}]
    for example_input, example_output in _CHECKER_EXAMPLES:
        messages.append({"role": "user", "content": example_input})
        messages.append({"role": "assistant", "content": json.dumps(example_output)})
    messages.append(
        {
            "role": "user",
            "content": f"QUESTION: {question}\nANSWER: {answer}\nSOURCES:\n{context}",
        }
    )
    return messages
