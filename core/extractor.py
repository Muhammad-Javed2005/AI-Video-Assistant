import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

# Ensure environment variables are loaded
load_dotenv()


def get_llm():
    """Initializes and returns the Mistral AI LLM instance."""
    api_key = os.getenv("MISTRAL_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "MISTRAL_API_KEY is missing! Please set MISTRAL_API_KEY in your .env file."
        )

    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=api_key,
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    """Utility to build a standard LangChain processing pipeline."""
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{text}"),
        ]
    )

    # Clean & idiomatic LangChain pipeline
    return prompt | llm | StrOutputParser()


def extract_action_items(transcript: str) -> str:
    """Extracts action items, owners, and deadlines from the transcript."""
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all action items. For each provide:\n"
        "- Task description\n"
        "- Owner (who is responsible)\n"
        "- Deadline (if mentioned, else write 'Not specified')\n\n"
        "Format as a numbered list. If none found say 'No action items found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


def extract_key_decisions(transcript: str) -> str:
    """Extracts key decisions made during the meeting."""
    system_prompt = (
        "You are an expert meeting analyst. From the meeting transcript, "
        "extract all key decisions made. Format as a numbered list. "
        "If none found say 'No key decisions found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})


def extract_questions(transcript: str) -> str:
    """Extracts unresolved questions or topics needing follow-up."""
    system_prompt = (
        "From the meeting transcript, extract all unresolved questions "
        "or topics needing follow-up. Format as a numbered list. "
        "If none found say 'No open questions found.'"
    )

    chain = build_chain(system_prompt)
    return chain.invoke({"text": transcript})