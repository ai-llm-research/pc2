from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from languages import language_dict, all_languages
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain_openai import ChatOpenAI,OpenAI
import re
import json
import argparse
import textwrap
import sys

def divide_list(lst, num):
    # Calculate chunk size
    k, m = divmod(len(lst), num)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(num)]

class Translator:
    def __init__(self, model=None):
        self.model = model

        format_instructions = textwrap.dedent(
            """
            in the following format, including the leading and trailing "```" and "```" 
            The result within [TRANSLATED TEXT] must be valid JSON object, and it MUST contain exactly {languages_count} keys, one for each language in the languages_list. 
            Do not omit or add any keys. The keys must appear exactly as listed.

            ```
            [TRANSLATED TEXT START]
            {{
                "<LanguageName1>": string, // translation of the text into <LanguageName1> with placeholders preserved.
                                           // If not possible, output "None".
                "<LanguageName2>": string, // translation of the text into <LanguageName2> with placeholders preserved.
                                           // If not possible, output "None".
                ...
                "<LanguageNameN>": string  // translation of the text into <LanguageNameN> with placeholders preserved.
                                           // If not possible, output "None".
            }}
            [TRANSLATED TEXT END]
            ```
            """
        )

        # Prompt with placeholders
        prompt_with_placeholder = ChatPromptTemplate(
            [
                (
                    "system",
                    textwrap.dedent(
                        """
                        You are a translation assistant. Your task is to translate ONLY the text inside [TARGET TEXT START] and [TARGET TEXT END] 
                        into ALL of the following languages: {languages_list}.
                        
                        CRITICAL INSTRUCTIONS:
                        - The target text ALWAYS contains placeholders like <PLACE_HOLDER_DIRECT_x> or <PLACE_HOLDER_INDIRECT_x>. 
                          These placeholders MUST remain exactly as they are: unchanged, untranslated, unmoved.
                        - NEVER translate the example text inside [EXAMPLE WITHOUT PLACEHOLDER]. It is provided for context ONLY. 
                          Ignore it in the final translations.
                        - Always base your output ONLY on the placeholder-containing text inside [TARGET TEXT]. 
                          The version without placeholders is never to be output.
                        - If no reliable translation exists in one of the requested languages, output "None" for that language key.
                        - Do not omit any language key from the output. The total number of keys MUST be exactly {languages_count}.

                        Provide the final result {format_instructions}.
                        """
                    ),
                ),
                (
                    "user",
                    textwrap.dedent(
                        """
                        [EXAMPLE WITHOUT PLACEHOLDER START]
                        {target_example}
                        [EXAMPLE WITHOUT PLACEHOLDER END]

                        [TARGET TEXT START]
                        {target}
                        [TARGET TEXT END]
                        """
                    ),
                ),
            ],
            partial_variables={
                "format_instructions": format_instructions
            },
        )

        self.chain_with_placeholder = prompt_with_placeholder | model | StrOutputParser()

        # Prompt without placeholders
        prompt_without_placeholder = ChatPromptTemplate(
            [
                (
                    "system",
                    textwrap.dedent(
                        """
                        You are a translation assistant. Translate the user-provided text into ALL of the following languages: {languages_list},
                        preserving its original meaning, tone, and context.

                        CRITICAL INSTRUCTIONS:
                        - If translation into one of the requested languages does not exist, output "None" instead.
                        - Do not omit any language key from the output. The total number of keys MUST be exactly {languages_count}.

                        Provide the final result {format_instructions}.
                        """
                    ),
                ),
                (
                    "user",
                    textwrap.dedent(
                        """
                        [TARGET TEXT START]
                        {target}
                        [TARGET TEXT END]
                        """
                    ),
                ),
            ],
            partial_variables={
                "format_instructions": format_instructions
            },
        )

        self.chain_without_placeholder = prompt_without_placeholder | model | StrOutputParser()
        


    def parse(self, output, languages_list):
        # Now parse using JSON (or regex)
        json_output = None
        try:
            # Remove code block markers (``` or ```json)
            parsed_result = output[output.find('[TRANSLATED TEXT START]') + len('[TRANSLATED TEXT START]'): output.find('[TRANSLATED TEXT END]')]
            # Remove inline // comments
            parsed_result = re.sub(r'//.*', '', parsed_result)
            json_output_text = parsed_result.strip()
        
            # json_output_text = re.findall(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)[-1].strip()
            json_output = json.loads(json_output_text)
            for lang in languages_list:
                if lang not in json_output:
                    print(lang,"is missing")
                    return None
        except Exception:
            print("LLM output not directly JSON. Need manual parsing.")
            print(output)
            return None

        return json_output

    def invoke(self, input, is_for_placeholder):
        if is_for_placeholder:
            return self.chain_with_placeholder.invoke(input)
        else:
            return self.chain_without_placeholder.invoke(input)

    def invoke_and_parse(self, input, is_for_placeholder, languages_list):
        # if parse output is None repeat invoke and parse until parse is not None or max retries
        max_retries =5
        retries = 0
        parsed_output = None
        while parsed_output is None and retries < max_retries:
            retries += 1
            output = self.invoke(input, is_for_placeholder)
            parsed_output = self.parse(output, languages_list)
        
        if retries == max_retries:
            print("Max retries reached. Returning None.")
            return None

        return parsed_output

    def translate_many_parallel(self, text: str, text_example:str, languages_list: List[str], is_for_placeholder, max_workers: int = 5) -> Dict[str, str]:

        divided_languages_list = divide_list(languages_list, max_workers)

        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_lang = {
                executor.submit(self.invoke_and_parse, {"target": text, "target_example":text_example, "languages_list": ", ".join(lang_list), "languages_count":len(lang_list),}, is_for_placeholder, lang_list): lang_list
                for lang_list in divided_languages_list
            }
            for future in as_completed(future_to_lang):
                try:
                    parsed = future.result()
                    results.update(parsed)
                except Exception as e:
                    print("Something Wrong in translate_many_parallel parsed output")
                    sys.exit(-1)
        return results
