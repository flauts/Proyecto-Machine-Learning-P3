# Clasificación Multiclase de Cyberbullying

Sistema de clasificación automática de mensajes de cyberbullying que identifica diferentes tipos de discriminación (étnica, religiosa, de género, edad u otros) usando técnicas de NLP y machine learning.

## Categorías de Clasificación

- **No Cyberbullying**: Mensajes sin contenido ofensivo
- **Ethnicity/Race**: Discriminación étnica o racial  
- **Religion**: Ataques hacia creencias religiosas
- **Gender/Sexual**: Acoso relacionado con género o orientación sexual
- **Age**: Discriminación por edad
- **Other Cyberbullying**: Otras formas de acoso

## Metodología

### Enfoques de Representación

**TF-IDF** - Características basadas en frecuencia de términos con n-gramas (1,2) y 5,000 dimensiones

**BERT** - Embeddings contextuales de 768 dimensiones usando `bert-base-uncased` con fine-tuning específico

### Modelos Evaluados

**TF-IDF:** Random Forest, Regresión Logística, Naive Bayes Multinomial

**BERT:** Fine-tuning estándar y optimizado con técnicas de regularización avanzadas

## Estructura del Proyecto

```
ultima_bala/
├── datasets/                 # Datos de entrenamiento y prueba
├── experiment/              # Scripts de experimentación BERT y TF-IDF
├── feature_extraction/      # Extracción de características
├── preprocessing/           # Limpieza y preparación de datos
├── plots/                  # Visualizaciones generadas
├── *.slurm                # Configuraciones para cluster SLURM
└── setup_*.sh             # Scripts de configuración automática
```

## Uso

### Configuración del Entorno

```bash
git clone https://github.com/flauts/Proyecto-Machine-Learning-P3
cd ultima_bala
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Ejecución de Experimentos

**En Cluster SLURM:**
```bash
# Extracción de características
bash setup_feature_extraction.sh

# Experimentación completa
bash setup_experiment.sh

# O individualmente
sbatch bert_experiment.slurm
sbatch tfidf_experiment.slurm
```

Los modelos entrenados, resultados y métricas se generan automáticamente al ejecutar los scripts SLURM.

## Resultados Principales

| Modelo | Accuracy | F1-Score Macro |
|--------|----------|----------------|
| TF-IDF (Reg. Logística) | 83.0% | 83.0% |
| BERT Optimizado | **85.9%** | **86.0%** |

**Fortalezas de BERT:**
- Discriminación étnica: F1 = 99%
- Cyberbullying religioso: F1 = 96% 
- Acoso por edad: F1 = 98%

**Principales desafíos:**
- Distinción entre "no cyberbullying" y "otro cyberbullying"
- Casos con contexto ambiguo

## Dataset

**Fuente:** [Cyberbullying Classification Dataset](https://github.com/leoAshu/cyberbullying-classification/tree/master)
- 46,017 mensajes de Twitter
- Distribución balanceada entre 6 categorías
- Preprocesamiento: eliminación de URLs, menciones y emojis

## Aplicación Práctica

Bot de Discord implementado: [BufordBot](https://github.com/moonliit/BufordBot) - Detección de cyberbullying en tiempo real con especificación de tipo y nivel de confianza.

---

*Proyecto desarrollado en UTEC usando el cluster Khipu para experimentación.*
