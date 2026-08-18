import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
 
load_dotenv()


def get_llm():
    """Mistral LLM instance initialization."""
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list[str]:
    """Splits long transcript into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=200
    )
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Map-Reduce summarization strategy for long transcripts."""
    llm = get_llm()

    # Step 1: Map Prompt for individual chunks
    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of a video/meeting transcript concisely.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    # Fast parallel processing using batch() instead of sequential loop
    chunks_inputs = [{"text": chunk} for chunk in chunks]
    chunks_summaries = map_chain.batch(chunks_inputs)

    combined_summary_input = "\n\n".join(chunks_summaries)

    # Step 2: Reduce Prompt for combined final output
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert content summarizer. Combine these partial summaries "
                "into one clean, structured, and professional summary with bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    combined_chain = combined_prompt | llm | StrOutputParser()

    return combined_chain.invoke({"text": combined_summary_input})


def generate_title(transcript: str) -> str:
    """Generates a concise title from the transcript."""
    llm = get_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the transcript, generate a short professional title "
                "(max 8 words). Only return the title text, nothing else.",
            ),
            ("human", "{text}"),
        ]
    )

    title_chain = title_prompt | llm | StrOutputParser()

    # Truncate to first 2000 chars for quick title extraction
    return title_chain.invoke({"text": transcript[:2000]}).strip()