from typing import List, Dict, Hashable
import json
from google import genai
from google.genai import types
from models import BatchFormattedResponse, BatchFormattedClassification, FormattedResponse, UserTaggedAnswer


def clean_user_answer(question: str, answers: List[Dict[Hashable, str]])-> List[FormattedResponse]:
    client = genai.Client()

    prompt = f"""
    You will be given a JSON list of user answers to the question "{question}".
    Your task is to format each one according to these rules:
    1. A valid response to a question can be a paragraph explaining or a directly any kind of list.
    2. Return *only* the JSON object matching the required schema.
    3. For each element of the main list:
        1- Correct spelling and grammar and remove ascents in order to avoid formatting errors.
        2- In the response you can use 'ni' as a placeholder for 'ñ'.
        2- Evaluate if it is a valid response to the question and if not leave the list empty.
        3- The content of each output item cannot contain any special characters and must be plain text dont use ascents.
        4- Each item should be reduce to a compact version still keeping relevant information to the response.
        5- Standarize terminoly.
    4. The valid answers, identify the sentiment (Positive, Negative, Mixed, Non-Response).

    Answers:
    {json.dumps(answers)}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0),
            response_mime_type='application/json',
            response_json_schema=BatchFormattedResponse.model_json_schema()
        ),

    )
    if not response.text:
        raise ValueError("No response from gemini")

    return BatchFormattedResponse.model_validate_json(response.text).formatted_results


def tag_topics(question: str, data_to_classify: List[Dict[Hashable, str]])->List[UserTaggedAnswer]:
    client = genai.Client()

    prompt = f"""
    You are a clustering machine and should tag information in panish.
    You will be given a JSON list of user answers to the question "{question}".
    All tags must be in lowercase snake_case.
    The topic tags should not contain accent nor symbols, it should be just plain text.
    Return *only* the JSON object matching the required schema.
    You follow this indications:
        1. Identify and List the main topics you find and a one-sentence summary of each topic.
        2. Perform topic clustering on these answers.
        3. you should tag each response in order to group related answers with same tag based on the topic.

    Answers:
    {json.dumps(data_to_classify)}
    """
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_budget=0),
            response_mime_type='application/json',
            response_json_schema=BatchFormattedClassification.model_json_schema()
        ),

    )
    if not response.text:
        raise ValueError("No response from gemini")

    return BatchFormattedClassification.model_validate_json(response.text).tagged_answers
