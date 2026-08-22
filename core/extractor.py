import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def get_llm():
    """Initializes Google Gemini Free Model instance."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY missing hai! .env file me set karein."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    """Utility to build standard LCEL processing pipeline."""
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    """Extracts action items, owners, and deadlines from transcript."""
    system_prompt = (
        "You are an expert meeting analyst. Extract all action items from the meeting transcript.\n"
        "For each action item, provide:\n"
        "- Task description\n"
        "- Owner (responsible person)\n"
        "- Deadline (if mentioned, else 'Not specified')\n\n"
        "Format as a clean numbered list. If none found, reply 'No action items found.'"
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extracts key decisions made during the meeting."""
    system_prompt = (
        "You are an expert meeting analyst. Extract all key decisions made in the meeting transcript.\n"
        "Format as a clean numbered list. If none found, reply 'No key decisions found.'"
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extracts unresolved questions or topics needing follow-up."""
    system_prompt = (
        "You are an expert meeting analyst. Extract all unresolved questions or open follow-up topics.\n"
        "Format as a clean numbered list. If none found, reply 'No open questions found.'"
    )
    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})