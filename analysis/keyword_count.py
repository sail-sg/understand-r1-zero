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

import fire

"""
Description:
    Count the keywords in the responses, in question- and response- levels.


Example usage:
python keyword_count.py --file_name deepseek_v3_base.json --n_samples 8
"""


def main(file_name: str = "deepseek_v3_base.json", n_samples: int = 8):
    output = json.load(open(file_name))
    print(f"Count the keywords in {file_name}")

    keyword_pool = {
        "recheck": 0,
        "rethink": 0,
        "reassess": 0,
        "reevaluate": 0,
        "re-evaluate": 0,
        "reevaluation": 0,
        "re-examine": 0,
        "reexamine": 0,
        "reconsider": 0,
        "reanalyze": 0,
        "double-check": 0,
        "check again": 0,
        "think again": 0,
        "verify again": 0,
        "go over the steps": 0,
    }

    # sr at question-level
    sr_per_question_keyword_list = []
    sr_per_question_llm_list = []
    sr_per_question_list = []

    # sr at response-level
    sr_per_response_keyword_list = []
    sr_per_response_llm_list = []
    sr_per_response_list = []

    # question-level counts
    sr_per_question_keyword = 0
    sr_per_question_llm = 0
    sr_per_question = 0

    # response-level counts
    sr_per_response_keyword = 0
    sr_per_response_llm = 0
    sr_per_response = 0

    for ques_idx, o in enumerate(output):
        # Track if at least one response for the current question exhibits SR
        question_has_sr_keyword = False
        question_has_sr_llm = False
        question_has_sr = False

        for resp_idx in range(n_samples):
            response = o[f"output_{resp_idx}"].lower()  # Make it case-insensitive

            # Cross-checking by keyword- and LLM-based detection
            keyword_detected = any(
                response.count(keyword) > 0 for keyword in keyword_pool
            )
            llm_detected = "2" in o[f"llm_detection_{resp_idx}"][-3:]

            # Response-level counting
            if keyword_detected:
                sr_per_response_keyword += 1
                question_has_sr_keyword = True  # Mark question-level keyword detection

            if llm_detected:
                sr_per_response_llm += 1
                question_has_sr_llm = True  # Mark question-level LLM detection

            if keyword_detected and llm_detected:
                sr_per_response += 1
                question_has_sr = True  # Mark question as having SR

        # Question-level counting
        if question_has_sr:
            sr_per_question += 1
        if question_has_sr_keyword:
            sr_per_question_keyword += 1
        if question_has_sr_llm:
            sr_per_question_llm += 1

        # Append results to lists
        sr_per_response_keyword_list.append(sr_per_response_keyword)
        sr_per_response_llm_list.append(sr_per_response_llm)
        sr_per_response_list.append(sr_per_response)

        sr_per_question_keyword_list.append(sr_per_question_keyword)
        sr_per_question_llm_list.append(sr_per_question_llm)
        sr_per_question_list.append(sr_per_question)

    print(
        f"response-level: keyword={sr_per_response_keyword_list}; llm={sr_per_response_llm_list}; cross={sr_per_response_list}"
    )
    print(
        f"question-level: keyword={sr_per_question_keyword_list}; llm={sr_per_question_llm_list}; cross={sr_per_question_list}"
    )


fire.Fire(main)
