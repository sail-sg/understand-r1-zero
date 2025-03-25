# Copyright 2025 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

import fire
from openai import OpenAI

"""
Description:
    Detect the self-reflection in the responses.

Example usage:
python sr_detection.py --file_name {fn}_template_{template}_temp{temperature}_topp{top_p}_n{n_samples}.json --n_samples 8
"""


def main(file_name: str = "output_Qwen_Qwen2.5-Math-7B.json", n_samples: int = 1):
    output = json.load(open(file_name))

    instruction = """I will send you a mathematical question along with a detailed response. Your task is to determine whether the response is attempting to answer the question. If the response is off-topic, hallucinated, random talk, or otherwise irrelevant, mark it as **0**. Otherwise, assess whether the response exhibits self-reflection.

### **Categorization Rules**:

1. **Category 0**: The response is **off-topic, nonsensical, incoherent, overly repetitive, or lacks logical reasoning**.
   - Example cases:
     - The response does not relate to the question.
     - It contains meaningless or hallucinated content.
     - It consists of excessive repetition without coherence.
   
2. **Category 1**: The response **attempts to answer the question** but does **not** exhibit self-reflection.
   - Example cases:
     - The response directly solves the problem without revisiting steps.
     - No attempt is made to verify the correctness of the answer or explore alternative solutions.

3. **Category 2**: The response **demonstrates self-reflection** at any level.
   - This may include:
     - **Explicit self-reflection keywords**, such as: *recheck, rethink, reassess, reevaluate, re-evaluate, reevaluation, re-examine, reexamine, reconsider, reanalyze, double-check, check again, think again, verify again, go over the steps*, etc.
     - **Implicit self-reflection behaviors**, such as revisiting the solution, questioning assumptions, or considering alternative approaches **without explicit keywords**.
   - If any form of self-reflection is present, **always categorize it as 2**, regardless of correctness or answer quality.

4. **Category 3**: The response consists **solely of Python code for calculations** without exhibiting self-reflection.
   - Example cases:
     - The response only provides a Python script to compute the solution **without any verification, re-evaluation, or alternative considerations**.

### **Output Format**:

Your response should first provide a **very brief explanation** of your analysis, followed by a **single category number (0, 1, 2, or 3)** at the end. You must include the category number at the end of your response.

**Example outputs:**
- `The response is off-topic and does not attempt to answer the question. 0.`
- `The response provides a direct solution without self-reflection. 1.`
- `The response demonstrates self-reflection. 2.`
- `The response consists solely of Python code without any self-reflection. 3.`

- **Question:** {question}  
- **Response:** {response}  
"""

    # api key, model, and parameters
    os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"
    client = OpenAI(
        api_key=os.environ.get(
            "OPENAI_API_KEY"
        ),  # This is the default and can be omitted
    )

    # choose LLM model and parameters
    llm_model = "gpt-4o-mini-2024-07-18"
    llm_temp = 0.0
    llm_max_tokens = 300

    n_samples = int(n_samples)
    count_signalwords = 0

    print(f"Detecting the self-reflection in {file_name}")
    for idx, o in enumerate(output):
        o["idx"] = idx
        for j in range(n_samples):
            # 1. keyword-based detection
            keywords_pool = {
                "recheck",
                "rethink",
                "reassess",
                "reevaluate",
                "re-evaluate",
                "reevaluation",
                "re-examine",
                "reexamine",
                "reconsider",
                "reanalyze",
                "double-check",
                "check again",
                "think again",
                "verify again",
                "go over the steps",
            }
            matched_keywords = {
                word for word in keywords_pool if word in o[f"output_{j}"].lower()
            }
            if matched_keywords:
                count_signalwords += 1
                keyword_appear = ", ".join(matched_keywords)  # Convert set to a string
            else:
                keyword_appear = ""

            # 2. llm-based detection
            prompt = instruction.format(
                question=o["question"], response=o[f"output_{j}"]
            )
            chat_completion = client.chat.completions.create(
                model=llm_model,
                temperature=llm_temp,
                max_tokens=llm_max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            llm_detection = chat_completion.choices[0].message.content

            # add llm_detection to the output
            o[f"keyword_detection_{j}"] = keyword_appear
            o[f"llm_detection_{j}"] = llm_detection

        # save the file
        file_name = file_name.replace(".json", "_sr.json")
        json.dump(
            output,
            open(f"{file_name}", "w"),
            indent=4,
        )


fire.Fire(main)
