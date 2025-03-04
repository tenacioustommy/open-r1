"""Reward functions for GRPO training."""

import math
import re
import os
from typing import Dict
from datetime import datetime
from openai import OpenAI
from pydantic import BaseModel
import dotenv
import json
dotenv.load_dotenv()
class Isright(BaseModel):
    isright: bool
prompt="please verify the response is right or wrong and give me true or false in json {'is_correct': } directly\n\nResponse: "+"The equation \(3a^2 = b^2 + 1\) has no integer solutions because, modulo 3, \(b^2\) would have to be congruent to 2, which is impossible since squares modulo 3 can only be 0 or 1."+"\nAnswer: [Nointegersolutionsexist, '\\text{No integer solutions exist}']"
client = OpenAI(api_key="EMPTY",base_url="http://127.0.0.1:8000/v1")
completion= client.chat.completions.create(model="Qwen2.5-7B-Instruct",
                                    messages=[{"role":"user","content":prompt}])
pattern = r"```json\s*(\{.*?\})\s*```"
match = re.search(pattern, completion.choices[0].message.content, re.DOTALL)
if match and json.loads(match.group(1))["is_correct"]:
    reward = 1.0
else:
    reward = -0.5