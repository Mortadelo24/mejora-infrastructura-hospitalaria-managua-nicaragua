from pydantic import BaseModel, Field
from typing import List, Dict

variable_to_question = {
    # --- Metadata (string, datetime) ---
    'response_id': 'Id',
    'start_response': 'Start time',
    'complete_response': 'Completion time',

    # --- Categorical Variables ---
    'age': 'Edad',
    'sex': 'Sexo',
    'years_studying': 'Año de estudio en el área de la salud.',
    'degree': 'Que carrera esta cursando?',
    'infrastructure_score': '¿Cómo calificaría el estado actual de la infraestructura hospitalaria en Managua?',
    'notice_infrastructure_improvement': '¿Ha notado mejoras en la infraestructura hospitalaria durante el período 2020-2025?',
    'frecuency_digital_systems': '¿Con qué frecuencia ha observado el uso de  sistemas digitales (expedientes electrónicos, telemedicina, imágenes  digitales) durante sus prácticas o rotaciones?',
    'technology_improves_attention': '¿Cree que la incorporación de nuevas tecnologías ha mejorado la atención a los pacientes?',
    'modernization_improves_attention': 'En su experiencia, ¿la modernización hospitalaria ha mejorado la calidad de la atención?',
    'improved_pathologies_treatments': '¿Considera que la red hospitalaria está mejor preparada para atender estas patologías en comparación con hace 5 años?',
    'pathologies': '¿Cuáles son las patologías más frecuentes que ha observado en sus prácticas?',
    
    # --- Numeric Variables (int64) ---
    'modernization_score': 'En general, ¿cómo calificaría la modernización de la red hospitalaria pública en Managua (2020-2025)?',

    # --- Open-Ended (string) Variables ---
    'infrastructure_changes': 'Mencione cambios en la infraestructura hospitalaria que considere relevante que haya llevado a cabo el gobierno de Nicaragua.',
    'most_important_technologies': 'Mencione almenos una tecnología hospitalaria que considere más importante en la actualidad.',
    'improved_aspects': '¿Qué aspecto considera que ha mejorado más?',
    'challenges': '¿Qué desafíos persisten en la atención hospitalaria a pesar de la modernización?',
    'recomendations': '¿Qué recomendaciones daría para mejorar la modernización hospitalaria en Nicaragua?'
}


class FormattedResponse(BaseModel):
    response_id: str = Field(..., description="A unique identifier that links to the original response")
    items: list[str] = Field(..., description="A list of items extracted from the answer. if the answer is not valid, then it should be an empty list")
    sentiment: str = Field(..., description="Sentiment tag from this list (Positive, Negative, Mixed, Non-Response)")

class BatchFormattedResponse(BaseModel):
    formatted_results: List[FormattedResponse] = Field(
        ..., 
        description="A list of the formatted answers, one for each input item, in the same order."
    )


class UserTaggedAnswer(BaseModel):
    response_id: str = Field(..., description="A unique identifier that links to the original response")
    topic: str = Field(..., description="A grouping tag that classfies the response of the user by general topic")

class BatchFormattedClassification(BaseModel):
    tagged_answers: List[UserTaggedAnswer] = Field(
        ..., 
        description="A list of formatted taggedAnswers"
    )