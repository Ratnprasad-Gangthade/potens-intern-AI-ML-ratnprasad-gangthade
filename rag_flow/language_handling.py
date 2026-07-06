from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


def detect_language(text: str) -> str:
    """
    Detect the language of the given text.
    """

    prompt = f"""
Detect the language of the following text.

Rules:
- Return only the English name of the language.
- Do not explain.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()


def translate_to_english(text: str) -> str:
    """
    Translate text to English.
    """

    prompt = f"""
Translate the following text to English.

Rules:
- Return only the translated text.
- Do not answer the text.
- Do not explain.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()


def translate_from_english(text: str, target_language: str) -> str:
    """
    Translate English text to the target language.
    """

    prompt = f"""
Translate the following English text to {target_language}.

Rules:
- Return only the translated text.
- Do not add explanations.

Text:
{text}
"""

    result = llm.invoke(prompt)

    return result.content.strip()
