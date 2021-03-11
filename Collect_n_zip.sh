rm -f SentenceClassification.zip                           # Force(-f) remove(rm) "SentenceClassification.zip"
zip -r SentenceClassification.zip *.py ./Results ./Logging # Zip all(*) .py files as well as those under directories ./Results and ./Logging
