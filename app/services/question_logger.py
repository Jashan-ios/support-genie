
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
     
   
   




