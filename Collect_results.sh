rm -f SentenceClassification_res.zip                                                        # Force(-f) remove(rm) "SentenceClassification.zip"
zip -r SentenceClassification_res.zip *.py ./Results ./Logging ./Model/best_model_state.bin # Zip all(*) .py files as well as those under directories ./Results and ./Logging
