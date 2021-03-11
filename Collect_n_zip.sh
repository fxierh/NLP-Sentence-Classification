rm -f SentenceClassification.zip                           # Force(-f) delete "SentenceClassification.zip"
zip -r SentenceClassification.zip *.py ./Results ./Logging # Zip all .py files as well as files under directories ./Results and ./Logging
