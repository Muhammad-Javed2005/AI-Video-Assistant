import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


def get_llm():
    """Initializes Google Gemini Free Model instance."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise ValueError("GOOGLE_API_KEY missing hai!")

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )


def split_transcript(transcript: str) -> list[str]:
    """Splits long transcript into processing chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    """Map-Reduce summarization strategy for transcripts."""
    llm = get_llm()

    # Step 1: Map Prompt
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a transcript concisely, covering all key points."),
            ("human", "{text}"),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)
    chunks_inputs = [{"text": chunk} for chunk in chunks]
    
    # Parallel batch execution
    chunks_summaries = map_chain.batch(chunks_inputs)
    combined_summary_input = "\n\n".join(chunks_summaries)

    # Step 2: Reduce Prompt
    combined_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert content summarizer. Combine these partial summaries into one "
                "well-structured, highly professional executive summary with clear bullet points.",
            ),
            ("human", "{text}"),
        ]
    )
    combined_chain = combined_prompt | llm | StrOutputParser()
    return combined_chain.invoke({"text": combined_summary_input})


def generate_title(transcript: str) -> str:
    """Generates a short concise title from transcript."""
    llm = get_llm()
    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a short, relevant professional title (max 8 words) for this transcript. "
                "Output ONLY the title text.",
            ),
            ("human", "{text}"),
        ]
    )
    title_chain = title_prompt | llm | StrOutputParser()
    return title_chain.invoke({"text": transcript[:2500]}).strip()