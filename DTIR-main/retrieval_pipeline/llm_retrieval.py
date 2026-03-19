import ast
import json
import multiprocessing
import os
import re
import concurrent
import time
import argparse
from pathlib import Path
from utils.log import log_info, log_verbose, set_log_level
from utils.utils import create_directory, str_convert_to_int_array
from digital_twins_pipeline.digital_twins_loader import DigitalTwinsLoader
from models.llm_api import LlmAPIWrapper



MAX_SUB_PROCESS_TIMEOUTT = 300
COARSELY_THREAD_POOL_NUM = 3
RANK_THREAD_POOL_NUM = 3
LLM_TEMPERATURE = 0.0
FOCUS_SCORE_WEIGHT = 0.3
SEMANTIC_SCORE_WEIGHT = 0.7
COARSELY_TOP_X_FILTER_NUM = 20
MAX_RANK_NUM = COARSELY_TOP_X_FILTER_NUM
MAX_OBJECT_RETRIEVAL_NUM = 5
MAX_MISSING_ATTRIBUTE_VALUE_STRING_LENGTH = 100
COARSELY_LLM_SYS_PROMPT =\
'''
You are tasked with calculating similarity scores between an image and a textual query, based solely on the textual description of the image.

Input:
- ImageDimensions: The width and height of the image.
- Caption: A concise description of the image, highlighting its most notable feature.
- GlobalSemantics: A paragraph summarizing the overall content of the image.
- ObjectsList: A list of detected objects from the image, each with:
  - id: unique identifier
  - semantics: object description
  - bbox: bounding box (x_min, y_min, width, height)
  - depth: a value from 0-255 (lower means closer)
- Query: A text description of a visual scene or object.
## Note: You will only receive the textual descriptions, not the actual images.

Processing Instructions (Step-by-step):
1. Key Element Extraction
  - Identify key visual and relational elements from the query. Use the following priority order:
    a. Core Subject(s) — the main entity or scene type
    b. Main Objects — prominent items or people in the scene
    c. Visual Attributes — key appearance details (e.g., clothing, size, hairstyle, accessories)
    d. Gender — if specified in the query
    e. Object Count — match to query (e.g., "two dogs" ≠ "one dog")
    f. Relations or Interactions — identify the interactions between subjects/objects or their relationship to the environment
    g. Age Range — approximate, based on textual cues
  - Ignore:
    - Subjective/emotional words (e.g., "beautiful", "happy")
    - Expressions or motion verbs (e.g., "running", "smiling"), unless explicitly required
  - Use the extracted key elements as the basis for matching in the following steps.
2. Focus Alignment Score
  - Understand and infer the main visual focus of the image based on its description.
  - Compare this focus with the core elements extracted from the query.
  - Output a single numeric score between 0 and 1 indicating how well the image's visual focus aligns with the key elements extracted from the query (e.g., [0.7]).
3. Semantic Similarity Score
  - Compare the overall semantic meaning of the query and the image's description.
  - Use the following scoring scale:
    - [0.95-1.0]: Fully matched or near-exact synonyms.
    - [0.8-0.95]: Minor differences or ambiguous synonyms.
    - [0.6-0.8]: One key element missing or altered.
    - [0.4-0.6]: Two key elements missing or altered.
    - [0.2-0.4]: More than two missing elements, partial match, or mismatch in quantity.
    - [0.0-0.2]: Irrelevant or contradictory content.
  - Output a single numeric score between 0 and 1 indicating how well the semantic content of the image matches the query (e.g., [0.91]).

Output Format:
[Explanation]: <Brief reasoning for each step>
[FocusScore]: <A single numeric score between 0 and 1, always enclosed in square brackets, e.g., [0.83]>
[SemanticScore]: <A single numeric score between 0 and 1, always enclosed in square brackets, e.g., [0.57]>

Key Considerations for Accurate Ranking:
- Avoid shallow text-based matching; focus on meaning, intent, and relational structure.
- Prioritize visual accuracy over ambiguous or subjective language.
- Ensure object counts align with the query (e.g., "two dogs" ≠ "one dog").
- Maintain strict attention to physical appearance attributes such as clothing, hair color, accessories.
- Always ignore emotional tone, facial expression, or movement states unless directly tied to appearance.
  
Tolerant Interpretation Guidelines
Real-world textual descriptions may include minor mismatches or ambiguities. Use the following guidance to avoid over-penalizing such cases:
1. Visually Similar but Functionally Equivalent Objects
  - Do not penalize mismatches between highly similar-looking objects unless the query explicitly requires fine-grained distinction. Examples include:
    - Saxophone case vs. Clarinet case
    - Glasses case vs. Sunglasses case
  - These can be treated as semantically equivalent due to visual ambiguity
  - This rule does not apply to semantically distinct objects like apple vs. pear or cat vs. dog
2. Coarse-Grained Color Matching
  - Use broad color categories unless the query calls for exact shades. Variants like light green, dark green, and green should all be interpreted as green. Example mappings:
    - Sky blue, navy, royal blue → blue
    - Beige, tan, light brown → brown
  - Only enforce fine-grained color differences if the query explicitly requires it.

'''
RANK_SYS_PROMPT =\
'''
You are tasked with ranking a list of images by their semantic relevance to a given textual query, based solely on the textual descriptions provided for each image.

Input:
- A query describing a visual scene or object.
- A list of image descriptions in JSON format. Each item contains:
  - ImageId: Unique identifier for the image.
  - ImageDimensions: The width and height of the image.
  - Caption: A concise description of the image, highlighting its most notable feature.
  - GlobalSemantics: A paragraph summarizing the overall content of the image.
  - ObjectsList: A list of detected objects from the image, each with:
    - id: Unique identifier
    - semantics: Object description
    - bbox: Bounding box (x_min, y_min, width, height)
    - depth: A value from 0-255 (lower means closer).
  Note: You will only receive the textual descriptions, not the actual images.

Processing Instructions (Step-by-step):

1. Understand the Core Content of the Images
- Start by analyzing the Caption and GlobalSemantics of all images to form an initial understanding of each image's overall scene. These two fields define the global structure, including key subjects, scene type, and the number of relevant people, animals, or objects.
- If any image lacks sufficient detail for a specific object, refer to its ObjectsList to enrich the description. Use:
  - The semantics field for fine-grained appearance attributes (e.g., clothing, color, accessories)
  - The bbox and depth fields for understanding spatial positioning and foreground/background structure

2. Per-image Relevance Scoring
- Evaluate how well each image's inferred content aligns with the query. Use the following priority order:
  a. Core Subject(s) — the main entity or scene type
  b. Main Objects — prominent items or people in the scene
  c. Visual Attributes — key appearance details (e.g., clothing, size, hairstyle, accessories)
  d. Gender — if specified in the query
  e. Object Count — match to query (e.g., "two dogs" ≠ "one dog")
  f. Relations or Interactions — identify the interactions between subjects/objects or their relationship to the environment
  g. Age Range — approximate, based on textual cues
- Ignore:
  - Subjective/emotional words (e.g., "beautiful", "happy")
  - Expressions or motion verbs (e.g., "running", "smiling"), unless explicitly required

3. Relative Comparison Across Images
- Compare all images together to determine their relative relevance to the query. Rank them based on how closely they match the meaning, intent, and structural constraints of the query.
- Use the same priority order from the Per-image Relevance Scoring step to ensure consistency and fairness across all images.

4. Reasoning Documentation
- Explain your reasoning in a clear, step-by-step format. Ensure that your logic is:
  - Interpretable and grounded in semantic content
  - Consistent across all images
  - Focused on visual meaning, not superficial keyword overlap
  - Inclusive of all images, even those that are minimally relevant. Provide justification for why less relevant images are ranked lower.
- Return a final ranked list containing all image IDs, from most to least relevant. Do not omit any image IDs under any circumstance.

Output Format:
Return exactly two lines:
[Explanation]: <reasoning for each step>
[Answer]: [ <image_id_1>, <image_id_2>, <image_id_3>, ... ]

Important: You must include all image IDs in the [Answer] list, even if some are only loosely related to the query. Omitting image IDs is considered an error.

Key Considerations for Accurate Ranking:
- Avoid shallow text-based matching; focus on meaning, intent, and relational structure.
- Prioritize visual accuracy over ambiguous or subjective language.
- You must fully consider the relationship between all images and the query.
- All image IDs must appear in the output ranked list, in order of relevance. Do not skip or remove any.
- Ensure object counts align with the query (e.g., "two dogs" ≠ "one dog").
- Maintain strict attention to physical appearance attributes such as clothing, hair color, accessories.
- Always ignore emotional tone, facial expression, or movement states unless directly tied to appearance.

Tolerant Interpretation Guidelines
Real-world textual descriptions may include minor mismatches or ambiguities. Use the following guidance to avoid over-penalizing such cases:

1. Visually Similar but Functionally Equivalent Objects
- Do not penalize mismatches between highly similar-looking objects unless the query explicitly requires fine-grained distinction. Examples include:
  - Saxophone case vs. Clarinet case
  - Glasses case vs. Sunglasses case
- These can be treated as semantically equivalent due to visual ambiguity
- This rule does not apply to semantically distinct objects like apple vs. pear or cat vs. dog

2. Coarse-Grained Color Matching
- Use broad color categories unless the query calls for exact shades. Variants like light green, dark green, and green should all be interpreted as green. Example mappings:
  - Sky blue, navy, royal blue → blue
  - Beige, tan, light brown → brown
- Only enforce fine-grained color differences if the query explicitly requires it.

'''
OBJECT_RETRIEVAL_MISSING_ATTRIBUTE_CHECK_PROMPT = \
'''
Task Objective:
You are given structured scene data extracted from an image. Your goal is to determine whether the currently available attributes of detected objects are sufficient to identify which objects are relevant to the given query.

Inputs:
- Query: A description of a target object or scenario.
- ImageDimensions: The width and height of the image.
- Caption: A concise description of the image, highlighting its most notable feature.
- GlobalSemantics: A paragraph summarizing the overall content of the image.
- ObjectsList: A list of detected objects from the image, each with:
  - id: unique identifier
  - semantics: object description
  - bbox: bounding box (x_min, y_min, width, height)
  - depth: a value from 0-255 (lower means closer)

Reasoning Procedure:
1. Single Object Evaluation
- Iterate over each object. Evaluate whether its semantics, bbox (position), and depth indicate potential relevance to the query.
- Look beyond keyword matches — consider implied meaning, intent, and contextual alignment with the global semantics.
2. Multi-object Composition Reasoning
- Consider whether a group of objects together can fulfill the query, especially if the query implies actions, interactions, or events.
- Use spatial relationships, depth ordering, and joint semantics to infer higher-level scene structure and relational meanings.
3. Contextual Alignment
- Always ensure your reasoning is anchored to the Global semantics. Treat it as a narrative frame or reference blueprint — objects must make sense within that scene.
4. Missing Information Check
- If the provided information is not sufficient to determine object-query relevance, and you can identify specific missing attributes (e.g., object color, shape), report them clearly.
- When specifying a missing attribute, use a single PascalCase word as the name (e.g., ObjectColor, Shape). Do not use phrases or sentences.
- Also provide a brief description (1-2 short sentences) of what the attribute represents and why it is relevant to reasoning.

Guidelines:
- Global semantics the high-level scene context and should serve as a blueprint for understanding individual objects. Any reasoning about specific objects must be grounded in this global scene context.
- Avoid shallow string or keyword matching.
- Focus on semantic intent, visual reasoning, and scene-level logic.
- Leverage common-sense understanding to reason about interactions and roles.
- Ensure interpretability: Clearly explain your logic in natural language.

Output Format:
- If the current information is sufficient:
[Explanation]: <Briefly explain how the relevant object(s) match the query and what reasoning steps were followed>
[IsMissingInfo]: false
[MissingName]: null
[MissingDescription]: null
- If the current information is insufficient:
[Explanation]: <Explain why the current inputs are ambiguous or lacking, and what specific information is missing>
[IsMissingInfo]: true
[MissingName]: <Name of the missing attribute>
[MissingDescription]: <Brief description of the missing attribute>
'''
OBJECT_RETRIEVAL_MISSING_ATTRIBUTE_CODE_PROMPT = \
'''
Task Objective:
You are given the name of a missing object-level attribute (in PascalCase format) that is required for downstream query-object reasoning. Your task is to generate a Python function that extracts this specific attribute from an image and a binary object mask.

Inputs:
- Missing attribute: The description of the missing attribute.
- Query: A description of a target object or scenario.
- ImageDimensions: The width and height of the image.
- Caption: A concise description of the image, highlighting its most notable feature.
- GlobalSemantics: A paragraph summarizing the overall content of the image.
- ObjectsList: A list of detected objects from the image, each with:
  - id: unique identifier
  - semantics: object description
  - bbox: bounding box (x_min, y_min, width, height)
  - depth: a value from 0-255 (lower means closer)
  
Code Generation Requirements:
- The function only receives the following parameters:
  - image_path (a string representing the image file path)
  - object_mask (a binary NumPy array of shape (H, W), where True indicates the object's pixels)
- The function must return the missing attribute's value
- Return only the function definition, no examples or test code
- Ensures the function name follows the format: extract_<missing_info_in_snake_case>(). For example, for MissingInfo = "ObjectColor", the function should be named extract_object_color.
- The Python code must be enclosed within a Markdown code block using triple backticks:
```python
# Your code here
```

Output Format:
[Explanation]: <Briefly explain>
[Code]: "```python\n<your function definition>\n```"
'''
OBJECT_RETRIEVAL_PROMPT_PART1 = \
'''
Task Objective:
You are given structured scene data extracted from an image. Your goal is to identify which objects are relevant to the given query, based on currently available object attributes.

Inputs:
- Query: A description of a target object or scenario.
- ImageDimensions: The width and height of the image.
- Caption: A concise description of the image, highlighting its most notable feature.
- GlobalSemantics: A paragraph summarizing the overall content of the image.
- ObjectsList: A list of detected objects from the image, each with:
  - id: unique identifier
  - semantics: object description
  - bbox: bounding box (x_min, y_min, width, height)
  - depth: a value from 0-255 (lower means closer)

'''
OBJECT_RETRIEVAL_PROMPT_PART2 = \
'''
Reasoning Procedure:
1. Single Object Evaluation
- Iterate over each object. Evaluate whether its semantics, bbox (position), and depth indicate potential relevance to the query.
- Look beyond keyword matches — consider implied meaning, intent, and contextual alignment with the global semantics.
2. Multi-object Composition Reasoning
- Consider whether a group of objects together can fulfill the query, especially if the query implies actions, interactions, or events.
- Use spatial relationships, depth ordering, and joint semantics to infer higher-level scene structure and relational meanings.
3. Contextual Alignment
- Always ensure your reasoning is anchored to the Global semantics. Treat it as a narrative frame or reference blueprint — objects must make sense within that scene.
4. Final Determination of Relevant Objects
- Based on the reasoning above, determine which object(s) are relevant to the query. You may return multiple relevant object IDs if appropriate.
- If no object is clearly relevant, explain why and return an empty list.

Guidelines:
- Global semantics the high-level scene context and should serve as a blueprint for understanding individual objects. Any reasoning about specific objects must be grounded in this global scene context.
- Avoid shallow string or keyword matching.
- Focus on semantic intent, visual reasoning, and scene-level logic.
- Leverage common-sense understanding to reason about interactions and roles.
- Ensure interpretability: Clearly explain your logic in natural language.

Output Format:
[Explanation]: <Briefly explain how the relevant object(s) match the query and what reasoning steps were followed>
[RelevantObjectIDs]: <Strict Python list format (e.g., [2,5] or [])>
'''


def fork_and_run_code(shared_list, code_str, digital_twins_info, image_path):
    function_name = None
    pattern = r"def\s+([a-zA-Z_][a-zA-Z_0-9]*)\("
    matches = re.findall(pattern, code_str)
    for match in matches:
        func_name_counts = code_str.count(match+"(")
        if func_name_counts == 1:
            function_name = match

    if function_name is None:
        return

    code =\
f"""
import numpy as np
from numpy import ndarray as NDArray

{code_str}

for obj_info in digital_twins_info:
    obj_id = obj_info['id']
    missing_attribute_disc[obj_id] = {function_name}(image_path, obj_info['mask'])
"""

    missing_attribute_disc = {}
    params = {'digital_twins_info': digital_twins_info, 'image_path': image_path, 'missing_attribute_disc': missing_attribute_disc}

    exec(code, params)
    shared_list.append(params['missing_attribute_disc'])




class LLMRetrieval:
    def __init__(self):
        pass

    def process_coarsely_retrieval_task(self, task, image_digital_twins_info_map, scoring_llm_provider, scoring_llm_model):
        start_time = time.time()
        llm_model = LlmAPIWrapper(provider=scoring_llm_provider, model=scoring_llm_model)

        question = task['query']
        img_idx = task['img_idx']
        image_path = task['img_path']
        image_info = image_digital_twins_info_map[image_path]['image_info']
        digital_twins_info = image_digital_twins_info_map[image_path]['digital_twins_info']
        
        log_str = "="*50 + "\n"
        log_str += f"Coarse retrieval task, query: {question}, image_path: {image_path}\n"

        obj_list = "  [\n"
        for obj in digital_twins_info:
            if obj['description'] == "Nothing":
                continue
            obj_list = obj_list +\
                "    { " + f"'id': {obj['id']}, 'semantics': '{obj['description']}', 'bbox': [{', '.join(map(lambda x: str(int(x)), obj['box']))}], 'depth': {obj['depth']}" + " },\n"
        obj_list = obj_list + "  ]"

        prompt = COARSELY_LLM_SYS_PROMPT + "\n"\
            f"Input:\n"\
            f"Query: {question}\n"\
            f"Caption: {image_info['caption']}\n"\
            f"GlobalSemantics: {image_info['semantic']}\n" +\
            f"ObjectsList:\n{obj_list}"
            
        output_text = llm_model.inference(prompt)
        log_str += f"Prompt send to local LLM for image {img_idx}:\n{prompt}\n"
        log_str += f"Local LLM output text for image {img_idx}:\n{output_text}\n"
        
        # match focus score
        focus_score = 0.0
        match_focus_score = re.search(r'\[FocusScore\]:\s*\[\s*(.*?)\s*]\s*', output_text)
        if match_focus_score:
            focus_score_str = match_focus_score.group(1)
            try:
                focus_score = float(focus_score_str)
            except ValueError:
                log_str += f"[Error] Focus score is not a float: '{focus_score_str}', query: {question}, image_path: {image_path}\n"
        else:
            log_str += f"[Error] No focus score found in the output text, query: {question}, image_path: {image_path}\n"
        
        # match semantic score
        semantic_score = 0.0
        match_semantic_score = re.search(r'\[SemanticScore\]:\s*\[\s*(.*?)\s*]\s*', output_text)
        if match_semantic_score:
            semantic_score_str = match_semantic_score.group(1)
            try:
                semantic_score = float(semantic_score_str)
            except ValueError:
                log_str += f"[Error] Semantic score is not a float: '{semantic_score_str}', query: {question}, image_path: {image_path}\n"
        else:
            log_str += f"[Error] No semantic score found in the output text, query: {question}, image_path: {image_path}\n"

        total_score = round(FOCUS_SCORE_WEIGHT * focus_score + SEMANTIC_SCORE_WEIGHT * semantic_score, 2)
        log_str += f"LLM output for image: {image_path}, query: {question}, focus score: {focus_score}, semantic score: {semantic_score}, total score: {total_score}\n"
        
        llm_model = None
        end_time = time.time()
        log_str += f"Coarse retrieval task cost time: {end_time - start_time} seconds\n"
        log_str += "="*50 + "\n"
        log_verbose(log_str)
        
        return {
            'query': question,
            'image_path': image_path,
            'focus_score': focus_score,
            'semantic_score': semantic_score,
            'total_score': total_score,
            'score_explanation': output_text
        }

    def get_coarsely_answer(self, question_list, image_path_list, image_digital_twins_info_map, scoring_llm_provider, scoring_llm_model):
        coarsely_results = []
        
        for q_idx, question in enumerate(question_list):
            log_info(f"Starting coarse retrieval for query {q_idx+1}/{len(question_list)}: '{question}'")
            
            # Process a single query at a time
            thread_pool_tasks = []
            for img_idx, img_path in enumerate(image_path_list):
                thread_pool_tasks.append({
                    'query': question,
                    'img_idx': img_idx,
                    'img_path': img_path
                })
            
            query_results = []
            completed_count = 0
            total_tasks = len(thread_pool_tasks)
            log_info(f"Processing {total_tasks} images for query: '{question}'")
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=COARSELY_THREAD_POOL_NUM) as executor:
                future_to_task = {
                    executor.submit(self.process_coarsely_retrieval_task, task, image_digital_twins_info_map, scoring_llm_provider, scoring_llm_model): task
                    for task in thread_pool_tasks
                }
                
                for future in concurrent.futures.as_completed(future_to_task):
                    try:
                        result = future.result()
                        completed_count += 1
                        query_results.append(result)
                        
                        if completed_count % 10 == 0:
                            log_info(f"Coarse retrieval progress for query '{question}': {completed_count}/{total_tasks} images processed")
                            
                    except Exception as e:
                        task = future_to_task[future]
                        image_path = task['img_path']
                        log_info(f"[Error] Coarse retrieval task failed for query '{task['query']}', image_path: {image_path}: {str(e)}")
            
            # Sort by total_score and keep only top COARSELY_TOP_X_FILTER_NUM
            query_results = sorted(query_results, key=lambda x: x['total_score'], reverse=True)
            top_results = query_results[:COARSELY_TOP_X_FILTER_NUM]
            
            log_info(f"Coarse retrieval for query '{question}': selecting top {len(top_results)} images from {len(query_results)} processed")
            
            # Add result for this query
            query_answer = {
                'query': question,
                'answers': [
                    {
                        'image_path': result['image_path'],
                        'focus_score': result['focus_score'],
                        'semantic_score': result['semantic_score'],
                        'total_score': result['total_score'],
                        'score_explanation': result['score_explanation']
                    } for result in top_results
                ]
            }
            coarsely_results.append(query_answer)

        return coarsely_results

    def processing_rank_retrieval_answers(self, task, image_digital_twins_info_map, ranking_llm_provider, ranking_llm_model):
        start_time = time.time()

        llm_model = LlmAPIWrapper(provider=ranking_llm_provider, model=ranking_llm_model)
        query = task['query']
        answers_original = task['answers']
        ranked_result = []

        log_str = "="*50 + "\n"
        log_str += f"Ranking task, query: {query}\n"

        if len(answers_original) > 0:
            image_list = "[\n"
            for idx, ans in enumerate(answers_original):
                if idx >= MAX_RANK_NUM:
                    break
                image_path = ans['image_path']
                image_info = image_digital_twins_info_map[image_path]['image_info']
                digital_twins_info = image_digital_twins_info_map[image_path]['digital_twins_info']

                obj_list = "[\n"
                for obj in digital_twins_info:
                    if obj['description'] == "Nothing":
                        continue
                    obj_list = obj_list +\
                        "      { " + f"'id': {obj['id']}, 'semantics': '{obj['description']}', 'bbox': [{', '.join(map(lambda x: str(int(x)), obj['box']))}], 'depth': {obj['depth']}" + " },\n"
                obj_list = obj_list + "    ]"

                image_list = image_list + \
                    "  {\n" + \
                    f"    'ImageId': {idx},\n" + \
                    f"    'ImageDimensions': {{image_width: {image_info['width']}, image_height: {image_info['height']}}},\n" + \
                    f"    'Caption': '{image_info['caption']}',\n" + \
                    f"    'GlobalSemantics': '{image_info['semantic']}',\n" + \
                    f"    'ObjectsList': {obj_list},\n" + \
                    "  },\n"
            image_list = image_list + "]"

            prompt = RANK_SYS_PROMPT +\
                f"Input:\n" +\
                f"Query: {query}\n"\
                f"ImagesList: {image_list}"

            output_text = llm_model.inference(prompt, temperature=LLM_TEMPERATURE)
            log_str += f"Prompt send to LLM:\n{prompt}\n"
            log_str += f"LLM output text:\n{output_text}\n"

            # match answer is list
            match_answer = re.search(r"\[Answer\]:\s*\[\s*(\d+(,\s*\d+)*)?\s*]\s*", output_text)
            answer_ids = []
            if match_answer and match_answer.group(1):
                answer_ids = "[" + match_answer.group(1) + "]"
                try:
                    answer_ids = ast.literal_eval(answer_ids)
                    if not isinstance(answer_ids, list):
                        answer_ids = []
                except (ValueError, SyntaxError) as e:
                    answer_ids = []

            if len(answer_ids) < MAX_RANK_NUM and len(answer_ids) > 0:
                used_indices = set(answer_ids)
                for i, answer in enumerate(answers_original):
                    if i not in used_indices and i < MAX_RANK_NUM:
                        ranked_result.append(answer)

            for idx, answer_id in enumerate(answer_ids):
                if answer_id >= len(answers_original):
                    continue
                ranked_result.append(answers_original[answer_id])

        if len(ranked_result) == 0:
            log_info(f"[Error] No ranked result for query: {query}")
            ranked_result = answers_original

        log_str += f"Ranking task done, query: {query}, ranked result: {ranked_result}\n"
        end_time = time.time()
        log_str += f"Ranking task cost time: {end_time - start_time} seconds\n"
        log_str += "="*50 + "\n"
        log_verbose(log_str)
        return {
            'query': query,
            'answers': ranked_result,
            'rank_explanation': output_text
        }

    def get_rank_answer(self, retrieval_answer, image_digital_twins_info_map, ranking_llm_provider, ranking_llm_model):
        log_info("Starting parallel ranking of answers")
        ranked_answer = []

        thread_pool_tasks = []
        for query_idx, query_item in enumerate(retrieval_answer):
            query = query_item['query']
            answers = query_item['answers']
            thread_pool_tasks.append({
                'query': query,
                'answers': answers
            })
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=RANK_THREAD_POOL_NUM) as executor:
            future_to_task = {
                executor.submit(self.processing_rank_retrieval_answers, task, image_digital_twins_info_map, ranking_llm_provider, ranking_llm_model): task
                for task in thread_pool_tasks
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    ranked_answer.append({
                        'query': result['query'],
                        'rank_explanation': result['rank_explanation'],
                        'answers': result['answers'],
                    })
                except Exception as e:
                    # If ranking fails, use the original unranked answers
                    task = future_to_task[future]
                    query = task['query']
                    answers = task['answers']
                    ranked_answer.append({
                        'query': query,
                        'rank_explanation': "",
                        'answers': answers,
                    })
                    log_info(f"[Error] Final answer processing task failed for query '{task['query']}': {str(e)}")
                    
        log_info("Completed ranking of answers")
        return ranked_answer

    def get_object_missing_attribute(self, llm_model, question, digital_twins_info, image_info):
        start_time = time.time()
        log_str = "="*50 + "\n"
        log_str += f"Missing attribute check task, query: {question}\n"
        
        obj_list = ""
        for obj in digital_twins_info:
            if obj['description'] == "Nothing":
                continue
            obj_list = obj_list +\
                "    {" +\
                f"'id':{obj['id']},'semantics':'{obj['description']}','bbox':[{', '.join(map(lambda x: str(int(x)), obj['box']))}],'depth':{obj['depth']}"\
                + "},\n"
        prompt = OBJECT_RETRIEVAL_MISSING_ATTRIBUTE_CHECK_PROMPT + "\n\n"\
            f"Input:\n"\
            f"- Query: {question}\n"\
            f"- ImageDimensions: {{image_width: {image_info['width']}, image_height: {image_info['height']}}}\n"\
            f"- Caption: {image_info['caption']}\n"\
            f"- GlobalSemantics: {image_info['semantic']}\n"\
            f"- ObjectsList:\n"\
            f"  [\n"\
            f"{obj_list}"\
            f"  ]"
            
        log_str += f"Prompt send to LLM for missing attribute check:\n{prompt}\n"
        output_text = llm_model.inference(prompt)
        log_str += f"LLM output text for missing attribute check:\n{output_text}\n"
        
        is_missing_info = False
        missing_name = None
        missing_description = None
        
        match_is_missing_info = re.search(r'\[IsMissingInfo\]:\s*(.*?)\s*', output_text)
        if match_is_missing_info:
            is_missing_info_str = match_is_missing_info.group(1)
            is_missing_info = is_missing_info_str.lower() == "true"
        
        if is_missing_info:
            match_missing_name = re.search(r'\[MissingName\]:\s*(.*?)\s*', output_text)
            if match_missing_name:
                missing_name = match_missing_name.group(1)
            match_missing_description = re.search(r'\[MissingDescription\]:\s*(.*?)\s*', output_text)
            if match_missing_description:
                missing_description = match_missing_description.group(1)
        
        end_time = time.time()
        log_str += f"Missing attribute check result: is_missing_info={is_missing_info}, missing_name={missing_name}, missing_description={missing_description}\n"
        log_str += f"Missing attribute check task cost time: {end_time - start_time} seconds\n"
        log_str += "="*50 + "\n"
        log_info(log_str)
            
        return is_missing_info, missing_name, missing_description

    def get_object_missing_attribute_code(self, llm_model, question, digital_twins_info, image_info, missing_attribute_name, missing_attribute_description):
        start_time = time.time()
        log_str = "="*50 + "\n"
        log_str += f"Missing attribute code generation task, query: {question}, missing_attribute: {missing_attribute_name}\n"
        
        obj_list = ""
        for obj in digital_twins_info:
            if obj['description'] == "Nothing":
                continue
            obj_list = obj_list +\
                "    {" +\
                f"'id':{obj['id']},'semantics':'{obj['description']}','bbox':[{', '.join(map(lambda x: str(int(x)), obj['box']))}],'depth':{obj['depth']}"\
                + "},\n"
        prompt = OBJECT_RETRIEVAL_MISSING_ATTRIBUTE_CODE_PROMPT + "\n\n"\
            f"Input:\n"\
            f"- Missing attribute: {missing_attribute_name}, {missing_attribute_description}\n"\
            f"- Query: {question}\n"\
            f"- ImageDimensions: {{image_width: {image_info['width']}, image_height: {image_info['height']}}}\n"\
            f"- Caption: {image_info['caption']}\n"\
            f"- GlobalSemantics: {image_info['semantic']}\n"\
            f"- ObjectsList:\n"\
            f"  [\n"\
            f"{obj_list}"\
            f"  ]"
                
        log_str += f"Prompt send to LLM for code generation:\n{prompt}\n"
        output_text = llm_model.inference(prompt)
        log_str += f"LLM output text for code generation:\n{output_text}\n"
        
        python_code = ""
        match_python_code = re.search(r'```python(.*?)```', output_text, re.DOTALL)
        if match_python_code:
            python_code = match_python_code.group(1)
        
        end_time = time.time()
        log_str += f"Generated python code: {python_code} characters\n"
        log_str += f"Missing attribute code generation task cost time: {end_time - start_time} seconds\n"
        log_str += "="*50 + "\n"
        log_info(log_str)
            
        return python_code
            
    def get_object_relevant_ids(self, llm_model, question, digital_twins_info, image_info, is_missing_attribute, missing_attribute_name, missing_attribute_description, missing_attribute_value_disc):
        start_time = time.time()
        log_str = "="*50 + "\n"
        log_str += f"Object relevance identification task, query: {question}, is_missing_attribute: {is_missing_attribute}\n"
        
        obj_list = ""
        for obj in digital_twins_info:
            if obj['description'] == "Nothing":
                continue
            if is_missing_attribute:
                obj_id = obj['id']
                missing_attribute_value = f"{missing_attribute_value_disc[obj_id]}"[:MAX_MISSING_ATTRIBUTE_VALUE_STRING_LENGTH]
                
                obj_list = obj_list +\
                    "    {" +\
                    f"'id':{obj['id']},'semantics':'{obj['description']}','bbox':[{', '.join(map(lambda x: str(int(x)), obj['box']))}],'depth':{obj['depth']}"\
                    + f",'{missing_attribute_name}':{missing_attribute_value}"\
                    + "},\n"
            else:
                obj_list = obj_list +\
                    "    {" +\
                    f"'id':{obj['id']},'semantics':'{obj['description']}','bbox':[{', '.join(map(lambda x: str(int(x)), obj['box']))}],'depth':{obj['depth']}"\
                    + "},\n"
        prompt = OBJECT_RETRIEVAL_PROMPT_PART1
        if is_missing_attribute:
            prompt = prompt + \
                f"  - {missing_attribute_name}: {missing_attribute_description}"
        prompt = prompt + "\n\n" + OBJECT_RETRIEVAL_PROMPT_PART2+ "\n\n"\
            f"Input:\n"\
            f"- Query: {question}\n"\
            f"- ImageDimensions: {{image_width: {image_info['width']}, image_height: {image_info['height']}}}\n"\
            f"- Caption: {image_info['caption']}\n"\
            f"- GlobalSemantics: {image_info['semantic']}\n"\
            f"- ObjectsList:\n"\
            f"  [\n"\
            f"{obj_list}"\
            f"  ]"
        
        log_str += f"Prompt send to LLM for object relevance identification:\n{prompt}\n"
        output_text = llm_model.inference(prompt)
        log_str += f"LLM output text for object relevance identification:\n{output_text}\n"
        
        relevant_object_ids = []
        match_relevant_object_ids = re.search(r'\[RelevantObjectIDs\]:\s*\[\s*(\d+(,\s*\d+)*)?\s*]\s*', output_text)
        if match_relevant_object_ids and match_relevant_object_ids.group(1):
            relevant_object_ids_str = "[" + match_relevant_object_ids.group(1) + "]"
            try:
                relevant_object_ids = ast.literal_eval(relevant_object_ids_str)
                if not isinstance(relevant_object_ids, list):
                    relevant_object_ids = []
            except (ValueError, SyntaxError) as e:
                relevant_object_ids = []
        
        end_time = time.time()
        log_str += f"Relevant object IDs found: {relevant_object_ids}\n"
        log_str += f"Object relevance identification task cost time: {end_time - start_time} seconds\n"
        log_str += "="*50 + "\n"
        log_info(log_str)
            
        return relevant_object_ids, output_text

    def object_retrieval(self, question, image_path, dt_dir, object_retrieval_llm_provider, object_retrieval_llm_model):
        llm_model = LlmAPIWrapper(provider=object_retrieval_llm_provider, model=object_retrieval_llm_model)
        
        log_info(f"Object retrieval, question: {question}, image_path: {image_path}, dt_dir: {dt_dir}, llm_provider: {object_retrieval_llm_provider}, llm_model: {object_retrieval_llm_model}")

        dt_loader = DigitalTwinsLoader()
        image_name = Path(image_path).stem
        dt_path = os.path.join(dt_dir, image_name + ".json")
        dt_mask_path = os.path.join(dt_dir, image_name + "_mask.json")
        if os.path.exists(dt_path) and os.path.exists(dt_mask_path):
            digital_twins_info, image_info = dt_loader.load_digital_twins(dt_path, dt_mask_path)
        else:
            log_info(f"[Error] Digital twins not found for image: {image_path}")
            return
        
        is_missing_attribute, missing_attribute_name, missing_attribute_description = self.get_object_missing_attribute(llm_model, question, digital_twins_info, image_info)
        llm_model.clear_history()
        
        missing_attribute_disc = {}
        if is_missing_attribute:
            missing_attribute_code = self.get_object_missing_attribute_code(llm_model, question, digital_twins_info, image_info, missing_attribute_name, missing_attribute_description)
            llm_model.clear_history()
            
            with multiprocessing.Manager() as manager:
                shared_list = manager.list()
                p = multiprocessing.Process(target=fork_and_run_code, args=(shared_list, missing_attribute_code, digital_twins_info, image_path))
                p.start()

                p.join(timeout=MAX_SUB_PROCESS_TIMEOUTT)
                if p.is_alive():
                    log_info(f"[Error] Subprocess did not finish within {MAX_SUB_PROCESS_TIMEOUTT} seconds. Terminating...")
                    p.terminate()
                    p.join()

                if len(shared_list) >= 1:
                    missing_attribute_disc = shared_list[0]
                    log_info("Exec python code successfully")
                else:
                    log_info(f"[Error]Exec python code failed, query: {question}, image_path: {image_path}")

        relevant_object_ids, explanation = self.get_object_relevant_ids(llm_model, question, digital_twins_info, image_info, is_missing_attribute, missing_attribute_name, missing_attribute_description, missing_attribute_disc)
        llm_model.clear_history()

        return relevant_object_ids, explanation

    def retrieval(
            self,
            question_list,
            image_path_list,
            dt_dir,
            output_dir,
            scoring_llm_provider,
            scoring_llm_model,
            ranking_llm_provider,
            ranking_llm_model,
            object_retrieval_llm_provider,
            object_retrieval_llm_model
        ):
        # Load digital twins for all images
        image_digital_twins_info_map = {}
        image_digital_twins_info_map_coarse = {}
        dt_loader = DigitalTwinsLoader()
        for image_path in image_path_list:
            image_name = Path(image_path).stem
            dt_path = os.path.join(dt_dir, image_name + ".json")
            dt_path_coarse = os.path.join(dt_dir, image_name + "_coarse_grained.json")
            dt_mask_path = os.path.join(dt_dir, image_name + "_mask.json")

            if os.path.exists(dt_path) and os.path.exists(dt_mask_path):
                digital_twins_info, image_info = dt_loader.load_digital_twins(dt_path, dt_mask_path, False)
                image_digital_twins_info_map[image_path] = {
                    'digital_twins_info': digital_twins_info,
                    'image_info': image_info
                }
            if os.path.exists(dt_path_coarse) and os.path.exists(dt_mask_path):
                digital_twins_info_coarse, image_info_coarse = dt_loader.load_digital_twins(dt_path_coarse, dt_mask_path, False)
                image_digital_twins_info_map_coarse[image_path] = {
                    'digital_twins_info': digital_twins_info_coarse,
                    'image_info': image_info_coarse
                }
        if len(image_path_list) == len(image_digital_twins_info_map):
            log_info(f"Load digital twins for all images, total images: {len(image_path_list)}, images list: {image_path_list}")
        else:
            missing_image_path_list = [img_path for img_path in image_path_list if img_path not in image_digital_twins_info_map]
            log_info(f"[Error] Load digital twins for all images failed, total images: {len(image_path_list)}, missing images: {len(missing_image_path_list)}, missing images: {missing_image_path_list}")
            error_message = f"FATAL ERROR: Failed to load digital twins for {len(missing_image_path_list)} images out of {len(image_path_list)}."
            error_message += f"\nMissing digital twins for images: {missing_image_path_list}"
            raise RuntimeError(error_message)

        # Coarsely retrieval
        log_info("Start coarsely retrieval")
        coarsely_retrieval_answer = self.get_coarsely_answer(
            question_list,
            image_path_list,
            image_digital_twins_info_map_coarse,
            scoring_llm_provider,
            scoring_llm_model
        )
        with open(os.path.join(output_dir, "scoring_answers.json"), 'w', encoding='utf-8') as f:
            json.dump(coarsely_retrieval_answer, f, ensure_ascii=False, indent=2)
        log_info("Coarsely retrieval done")

        # Get ranked answers
        log_info("Start ranking answers")
        ranked_answers = self.get_rank_answer(coarsely_retrieval_answer, image_digital_twins_info_map, ranking_llm_provider, ranking_llm_model)
        with open(os.path.join(output_dir, "ranking_answers.json"), 'w', encoding='utf-8') as f:
            json.dump(ranked_answers, f, ensure_ascii=False, indent=2)
        log_info("Ranking answers done")

        # Get object retrieval answers
        log_info("Start object retrieval")
        object_retrieval_answers = []
        for item in ranked_answers:
            question = item['query']
            answers = item['answers']
            query_obj_answers = []
            for ans in answers[:MAX_OBJECT_RETRIEVAL_NUM]:
                image_path = ans['image_path']
                object_retrieval_answer, explanation = self.object_retrieval(question, image_path, dt_dir, object_retrieval_llm_provider, object_retrieval_llm_model)
                ans['object_retrieval_answer'] = object_retrieval_answer
                ans['object_retrieval_explanation'] = explanation
                query_obj_answers.append(ans)
            object_retrieval_answers.append({
                'query': question,
                'answers': query_obj_answers
            })
        with open(os.path.join(output_dir, "object_retrieval_answers.json"), 'w', encoding='utf-8') as f:
            json.dump(object_retrieval_answers, f, ensure_ascii=False, indent=2)
        log_info("Object retrieval done")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--query_info', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--dt_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--scoring_llm_provider', type=str)
    parser.add_argument('--scoring_llm_model', type=str)
    parser.add_argument('--ranking_llm_provider', type=str)
    parser.add_argument('--ranking_llm_model', type=str)
    parser.add_argument('--object_retrieval_llm_provider', type=str)
    parser.add_argument('--object_retrieval_llm_model', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_log_level(True)

    # Extract image paths from query_json_path
    image_path_list = []
    question_list = []
    with open(args.query_info, 'r') as f:
        query_data = json.load(f)
    for item in query_data:
        image_path_list.append(os.path.join(args.image_dir, item['image']))
        for question in item['caption']:
            question_list.append(question)

    log_info(f"Question list len: {len(question_list)}, list: {question_list}")
    log_info(f"Image list len: {len(image_path_list)}, list: {image_path_list}")

    llm_retrieval = LLMRetrieval()
    llm_retrieval.retrieval(
        question_list,
        image_path_list,
        args.dt_dir,
        args.output_dir,
        args.scoring_llm_provider,
        args.scoring_llm_model,
        args.ranking_llm_provider,
        args.ranking_llm_model,
        args.object_retrieval_llm_provider,
        args.object_retrieval_llm_model
    )