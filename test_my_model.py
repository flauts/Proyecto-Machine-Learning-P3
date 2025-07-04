#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simple para probar el modelo entrenado
Uso: python test_my_model.py

Carga el modelo entrenado de proyecto_bert.py y permite probar textos
"""

import os
import torch
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def cargar_modelo():
    """Cargar el modelo entrenado desde saved_cyberbullying_model"""
    model_path = "./saved_cyberbullying_model"
    
    if not os.path.exists(model_path):
        print("❌ No se encontró el modelo entrenado")
        print("💡 Primero ejecuta: python proyecto_bert.py")
        return None, None, None
    
    try:
        print("🔄 Cargando modelo entrenado...")
        
        # Cargar modelo y tokenizer
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # El mapeo de etiquetas está hardcodeado (mismo que en proyecto_bert.py)
        label_mapping = {
            'not_cyberbullying': 0,
            'ethnicity/race': 1, 
            'religion': 2,
            'gender/sexual': 3
        }
        
        # Configurar dispositivo
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        model.eval()
        
        print("✅ Modelo cargado exitosamente!")
        print(f"🖥️ Dispositivo: {device}")
        
        return model, tokenizer, label_mapping
        
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None, None

def predecir_texto(texto, model, tokenizer, label_mapping):
    """Hacer predicción de un texto"""
    device = next(model.parameters()).device
    
    # Mapeo inverso para convertir números a etiquetas
    reverse_mapping = {v: k for k, v in label_mapping.items()}
    
    # Tokenizar el texto
    encoded = tokenizer(
        texto,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}
    
    # Hacer predicción
    with torch.no_grad():
        outputs = model(**encoded)
        probabilities = torch.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    # Obtener la etiqueta predicha y la confianza
    predicted_label = reverse_mapping[prediction]
    confidence = probabilities[0][prediction].item()
    
    return predicted_label, confidence

def main():
    """Función principal"""
    print("🎯 TESTER DE MODELO DE CYBERBULLYING")
    print("="*50)
    
    # Cargar modelo
    model, tokenizer, label_mapping = cargar_modelo()
    
    if not model:
        return
    
    print(f"\n🏷️ Categorías que detecta:")
    print("  🟢 not_cyberbullying - Texto normal, sin cyberbullying")
    print("  🔴 ethnicity/race    - Cyberbullying racial/étnico") 
    print("  🔴 religion          - Cyberbullying religioso")
    print("  🔴 gender/sexual     - Cyberbullying de género/sexual")
    
    print(f"\n💬 Escribe textos para clasificar (o 'quit' para salir)")
    print("="*50)
    
    while True:
        try:
            # Pedir texto al usuario
            texto = input("\n📝 Escribe un texto: ").strip()
            
            # Salir si escribe quit
            if texto.lower() in ['quit', 'exit', 'salir']:
                break
            
            # Verificar que no esté vacío
            if not texto:
                continue
            
            # Hacer predicción
            predicted_label, confidence = predecir_texto(texto, model, tokenizer, label_mapping)
            
            # Mostrar resultado
            print(f"\n📊 Resultado:")
            print(f"   Texto: {texto}")
            print(f"   Predicción: {predicted_label}")
            print(f"   Confianza: {confidence:.3f} ({confidence*100:.1f}%)")
            
            # Añadir emoji según el tipo
            if predicted_label == 'not_cyberbullying':
                emoji = "✅ No es cyberbullying"
            else:
                emoji = "⚠️ Posible cyberbullying detectado"
            
            print(f"   {emoji}")
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta luego!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n🎯 Gracias por usar el tester!")

if __name__ == "__main__":
    main()
