#Actionableitems , decision , questions 

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import get_llm, invoke_llm



def build_chain(system_prompt : str):
    llm = get_llm(temperature=0.2)
    return (
        RunnablePassthrough() | RunnableLambda(lambda x : {"text" : x}) |ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human","{text}"),
    ]) | llm |StrOutputParser()
    )

def extract_action_items(transcript:str)->str:
    chain = build_chain(
         "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    return invoke_llm(chain, transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )
    return invoke_llm(chain, transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )
    return invoke_llm(chain, transcript)


def extract_meeting_insights(transcript: str) -> dict[str, str]:
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Analyze the meeting transcript and return only one valid JSON object with exactly these string keys: "
            "action_items, key_decisions, open_questions. Use concise numbered lists. "
            "If a category has no findings, say so in that value.",
        ),
        ("human", "{text}"),
    ])
    chain = prompt | get_llm(temperature=0.2) | JsonOutputParser()
    result = invoke_llm(chain, {"text": transcript})
    if not isinstance(result, dict):
        raise ValueError("Ollama returned meeting insights in an unexpected format.")

    return {
        "action_items": str(result.get("action_items", "No action items found.")),
        "key_decisions": str(result.get("key_decisions", "No key decisions found.")),
        "open_questions": str(result.get("open_questions", "No open questions found.")),
    }