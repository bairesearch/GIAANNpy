# Codex Agent Instructions

- Use tabs, not spaces for indentation.
- Never modify unrelated code.
- Hard failure is universally preferred over soft failure. If you are unsure about any given implementation, record your uncertainty - never insert a fallback strategy (soft failure) that breaks the requirement specification. 
- If a function expects particular arguments to be within a particular range, throw an error if they are not within the expected range (do not soft exit without fulfilling the purpose of the function).
- The current requirement specification is maintained in the ".nlc" (Natural Language Code) file in the main folder. If you are unsure about the purpose of any code, consult the requirement specification.
- Only ever return at the end of functions (do not create multiple return statements throughout functions).
- If new code would be better added to a new file, then create the new file.
- Features are defined in the globalDefs.py file (Boolean global variables: True/False = On/Off. All code within functions pertaining to a particular feature must be encapsulated by an if statement for its specific boolean global variable; e.g. "if(conceptColumnsDelimitByPOS): ...".
