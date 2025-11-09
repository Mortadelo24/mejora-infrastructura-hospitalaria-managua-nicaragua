from typing import List, Dict, Hashable
import json
from google import genai
from google.genai import types
import pandas as pd
from models import BatchFormattedResponse, BatchFormattedClassification

def clean_unstructed_text(question: str, answers: list[str]):
    client = genai.Client()

    prompt = f"""
    You will be given a JSON list of user responses.
    Your task is to format each one according to these rules:
    1. Parse answers for this question "{question}"
    2. A valid response to a question can be a paragraph explaining or a directly any kind of list.
    4. Return *only* the JSON object matching the required schema.
    5. For each element of the main list:
        1. Evaluate if it is a valid response to the question and if not leave the list empty.
        6. The content of each output item can not contain special characters and must be plain text.
        7. Each item should be reduce to a compact version still keep relevant information to the response.

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

    response_df = pd.DataFrame(BatchFormattedResponse.model_validate_json(response.text).model_dump()['formatted_results'])

    return response_df



def tag_topics(question: str, data_to_classify: Dict[Hashable, List[str]]):

    client = genai.Client()

    prompt = f"""
    You will be given a JSON list of user answers to the question "{question}".
    Identify and List the main topics you find and a one-sentence summary of each topic.
    Perform topic clustering on these answers.
    you should tag each response in order to group related answers with same tag based on the topic.
    Return *only* the JSON object matching the required schema.

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
    
    return BatchFormattedClassification.model_validate_json(response.text).model_dump()['tagged_answers']

