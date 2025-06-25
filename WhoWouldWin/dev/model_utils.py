"""
Utilidades para el modelo WhoWouldWin Argumentative Generator
Funciones auxiliares reutilizables para carga, generación y evaluación
"""

import torch
from torch.utils.data import Dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
import re
import numpy as np
from typing import List, Dict, Tuple, Optional


class WhoWouldWinDataset(Dataset):
    """
    Dataset personalizado para el problema de argumentación.
    Maneja la tokenización y preparación de datos para T5.
    """
    
    def __init__(self, dataframe, tokenizer, max_input_length=256, max_output_length=128):
        """
        Args:
            dataframe: DataFrame con columnas 'input' y 'output'
            tokenizer: Tokenizer de T5
            max_input_length: Longitud máxima del input
            max_output_length: Longitud máxima del output
        """
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Tokenizar input
        input_encoding = self.tokenizer(
            row['input'],
            max_length=self.max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenizar output si existe (para entrenamiento)
        if 'output' in row:
            target_encoding = self.tokenizer(
                row['output'],
                max_length=self.max_output_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # Preparar labels (reemplazar padding con -100)
            labels = target_encoding['input_ids']
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze(),
                'labels': labels.squeeze()
            }
        else:
            # Solo input (para inferencia)
            return {
                'input_ids': input_encoding['input_ids'].squeeze(),
                'attention_mask': input_encoding['attention_mask'].squeeze()
            }


def load_model_and_tokenizer(model_path: str = 'prod/modelo.pth', 
                           model_name: str = 't5-small',
                           device: Optional[torch.device] = None) -> Tuple[object, object]:
    """
    Carga el modelo entrenado y el tokenizer.
    
    Args:
        model_path: Ruta al archivo del modelo
        model_name: Nombre del modelo base
        device: Dispositivo para cargar el modelo
        
    Returns:
        model: Modelo cargado
        tokenizer: Tokenizer cargado
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Cargar tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # Cargar modelo base
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # Cargar pesos entrenados
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Modelo cargado desde {model_path}")
    except FileNotFoundError:
        print(f"Archivo {model_path} no encontrado. Usando modelo base.")
    except Exception as e:
        print(f"Error al cargar el modelo: {e}. Usando modelo base.")
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer


def clean_text(text: str) -> str:
    """
    Limpia y normaliza el texto.
    
    Args:
        text: Texto a limpiar
        
    Returns:
        Texto limpio
    """
    # Eliminar saltos de línea múltiples
    text = re.sub(r'\n+', ' ', text)
    
    # Eliminar espacios múltiples
    text = re.sub(r'\s+', ' ', text)
    
    # Eliminar caracteres de control
    text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
    
    # Normalizar comillas
    text = text.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
    
    # Eliminar espacios al inicio y final
    text = text.strip()
    
    return text


def add_task_prefix(text: str) -> str:
    """
    Agrega el prefijo de tarea para T5.
    
    Args:
        text: Texto del input
        
    Returns:
        Texto con prefijo
    """
    return f"argumenta sobre: {text}"


def generate_response(model, 
                     tokenizer, 
                     input_text: str, 
                     device: torch.device,
                     max_length: int = 128,
                     num_beams: int = 4,
                     temperature: float = 0.7,
                     top_k: int = 50,
                     top_p: float = 0.95,
                     do_sample: bool = True) -> str:
    """
    Genera una respuesta argumentativa para el input dado.
    
    Args:
        model: Modelo T5
        tokenizer: Tokenizer
        input_text: Texto de entrada
        device: Dispositivo de cómputo
        max_length: Longitud máxima de generación
        num_beams: Número de beams para búsqueda
        temperature: Temperatura de generación
        top_k: Top-k sampling
        top_p: Top-p (nucleus) sampling
        do_sample: Si usar sampling o no
        
    Returns:
        Respuesta generada
    """
    # Limpiar y preparar input
    input_text = clean_text(input_text)
    
    # Agregar prefijo si no lo tiene
    if not input_text.startswith("argumenta sobre:"):
        input_text = add_task_prefix(input_text)
    
    # Tokenizar
    inputs = tokenizer(
        input_text,
        return_tensors='pt',
        max_length=256,
        truncation=True,
        padding=True
    ).to(device)
    
    # Generar
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_length=max_length,
            num_beams=num_beams,
            temperature=temperature,
            do_sample=do_sample,
            top_k=top_k,
            top_p=top_p,
            early_stopping=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decodificar
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response


def batch_generate(model,
                  tokenizer,
                  input_texts: List[str],
                  device: torch.device,
                  batch_size: int = 4,
                  **generation_kwargs) -> List[str]:
    """
    Genera respuestas para múltiples inputs en batch.
    
    Args:
        model: Modelo T5
        tokenizer: Tokenizer
        input_texts: Lista de textos de entrada
        device: Dispositivo de cómputo
        batch_size: Tamaño del batch
        **generation_kwargs: Argumentos adicionales para generate()
        
    Returns:
        Lista de respuestas generadas
    """
    responses = []
    
    # Procesar en batches
    for i in range(0, len(input_texts), batch_size):
        batch_texts = input_texts[i:i + batch_size]
        
        # Preparar inputs
        batch_texts = [clean_text(text) for text in batch_texts]
        batch_texts = [add_task_prefix(text) if not text.startswith("argumenta sobre:") 
                      else text for text in batch_texts]
        
        # Tokenizar batch
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            max_length=256,
            truncation=True,
            padding=True
        ).to(device)
        
        # Generar
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                **generation_kwargs
            )
        
        # Decodificar
        for output in outputs:
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(response)
    
    return responses


def calculate_generation_metrics(predictions: List[str], 
                               references: List[str]) -> Dict[str, float]:
    """
    Calcula métricas básicas de generación.
    
    Args:
        predictions: Lista de predicciones
        references: Lista de referencias
        
    Returns:
        Diccionario con métricas
    """
    from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
    
    # Tokenizar
    pred_tokens = [pred.split() for pred in predictions]
    ref_tokens = [[ref.split()] for ref in references]
    
    # Calcular BLEU
    bleu_scores = []
    for pred, ref in zip(pred_tokens, ref_tokens):
        score = sentence_bleu(ref, pred, weights=(0.25, 0.25, 0.25, 0.25))
        bleu_scores.append(score)
    
    # Calcular longitudes promedio
    avg_pred_length = np.mean([len(pred) for pred in pred_tokens])
    avg_ref_length = np.mean([len(ref[0]) for ref in ref_tokens])
    
    return {
        'bleu_average': np.mean(bleu_scores),
        'bleu_corpus': corpus_bleu(ref_tokens, pred_tokens),
        'avg_prediction_length': avg_pred_length,
        'avg_reference_length': avg_ref_length,
        'length_ratio': avg_pred_length / avg_ref_length if avg_ref_length > 0 else 0
    }


class ArgumentativeTextGenerator:
    """
    Clase wrapper para facilitar el uso del modelo en producción.
    """
    
    def __init__(self, model_path: str = 'prod/modelo.pth', 
                 model_name: str = 't5-small',
                 device: Optional[torch.device] = None):
        """
        Inicializa el generador.
        
        Args:
            model_path: Ruta al modelo entrenado
            model_name: Nombre del modelo base
            device: Dispositivo de cómputo
        """
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, model_name, self.device)
        
        # Configuración por defecto
        self.default_config = {
            'max_length': 128,
            'num_beams': 4,
            'temperature': 0.7,
            'top_k': 50,
            'top_p': 0.95,
            'do_sample': True
        }
    
    def generate(self, input_text: str, **kwargs) -> str:
        """
        Genera una respuesta argumentativa.
        
        Args:
            input_text: Texto de entrada
            **kwargs: Parámetros de generación (sobrescriben los default)
            
        Returns:
            Respuesta generada
        """
        # Combinar configuración default con kwargs
        config = {**self.default_config, **kwargs}
        
        return generate_response(
            self.model,
            self.tokenizer,
            input_text,
            self.device,
            **config
        )
    
    def generate_batch(self, input_texts: List[str], batch_size: int = 4, **kwargs) -> List[str]:
        """
        Genera respuestas para múltiples inputs.
        
        Args:
            input_texts: Lista de textos
            batch_size: Tamaño del batch
            **kwargs: Parámetros de generación
            
        Returns:
            Lista de respuestas
        """
        # Combinar configuración default con kwargs
        config = {**self.default_config, **kwargs}
        
        return batch_generate(
            self.model,
            self.tokenizer,
            input_texts,
            self.device,
            batch_size,
            **config
        )
    
    def set_generation_config(self, **kwargs):
        """
        Actualiza la configuración de generación por defecto.
        
        Args:
            **kwargs: Nuevos parámetros
        """
        self.default_config.update(kwargs)


# Función auxiliar para uso rápido
def create_generator(model_path: str = 'prod/modelo.pth') -> ArgumentativeTextGenerator:
    """
    Crea una instancia del generador lista para usar.
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        Instancia de ArgumentativeTextGenerator
    """
    return ArgumentativeTextGenerator(model_path)


if __name__ == "__main__":
    # Ejemplo de uso
    print("Cargando modelo...")
    generator = create_generator()
    
    # Generar respuesta
    question = "Who would win in a fight between a lion and a tiger?"
    response = generator.generate(question)
    
    print(f"\nPregunta: {question}")
    print(f"\nRespuesta: {response}")