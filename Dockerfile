# Utiliser une image de base officielle de Python
FROM python:3.12.4-slim-buster

# Mettre à jour les paquets et installer les dépendances nécessaires
RUN apt update -y && apt install awscli -y

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de l'application dans le conteneur
COPY . /app

# Installer les dépendances Python
RUN pip install -r requirements.txt

# Exposer le port que Streamlit utilise par défaut
EXPOSE 8501

# Définir la commande par défaut pour exécuter Streamlit
CMD ["streamlit", "run", "app.py"]
