"""
The Spec

Build a QuestionLogger service that records every question asked.
Requirements:

Store questions in a list (in memory, that's fine)
Method: log(question: str, answer: str) → saves both
Method: get_recent(n: int = 10) → returns last N questions
Method: clear() → empties the log
Method: count() → returns total count
Use the singleton pattern (like your other services)
Use classmethods with cls
Add proper docstrings

"""

from typing import List, Dict

class QuestionLogger:
   _logs = []

   @classmethod
   def log(cls, question, answer):
      cls._logs.append({"question": question, "answer": answer})
      

   @classmethod
   def get_recent(cls, n: int = 10):
      return cls._logs[-n:]
   
   @classmethod
   def clear(cls):
      cls._logs.clear()

   @classmethod
   def count(cls):
      return len(cls._logs)
     
   
   




