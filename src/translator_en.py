from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from languages import language_dict
from textwrap import dedent
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
from langchain_openai import ChatOpenAI,OpenAI
import re
import json
import argparse
import textwrap

class TranslatorEn:
    def __init__(self, model=None):
        self.model = model
        format_instructions =textwrap.dedent(
            """
            in the following format, including the leading and trailing "```" and "```" 
            ```
            [TRANSLATED TEXT START]
            <the translated text>
            [TRANSLATED TEXT END]
            ```
            """
        ) # dedent end


        prompt_with_placeholder = ChatPromptTemplate([
                ("system", textwrap.dedent(
                    """
                    You are a translation assistant. Your task is to translate ONLY the text inside [TARGET TEXT START] and [TARGET TEXT END] into English.
                    
                    CRITICAL INSTRUCTIONS:
                    - The target text ALWAYS contains placeholders like <PLACE_HOLDER_DIRECT_x> or <PLACE_HOLDER_INDIRECT_x>. These placeholders MUST remain exactly as they are: unchanged, untranslated, unmoved.
                    - The final output MUST always be the translated text with placeholders preserved.

                    Provide the final result {format_instructions}.
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    [TARGET TEXT START]
                    Maak een afbeelding voor <PLACE_HOLDER_INDIRECT_1> die de regenboogvlag vasthoudt.
                    [TARGET TEXT END]
                    """
                    )
                ),
                ("ai", textwrap.dedent(
                    """
                    ```
                    [TRANSLATED TEXT START]
                    Create an image for <PLACE_HOLDER_INDIRECT_1> that holds the rainbow flag.
                    [TRANSLATED TEXT END]
                    ```
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    [TARGET TEXT START]
                    {target}
                    [TARGET TEXT END]
                    """
                    )
                )
                ],
                partial_variables={"format_instructions":format_instructions}
            )
        self.chain_with_placeholder = prompt_with_placeholder | model | StrOutputParser()

        prompt_without_placeholder = ChatPromptTemplate([
                ("system", textwrap.dedent(
                    """
                    You are a translation assistant. Translate the user-provided text into English, preserving its original meaning, tone, and context.
                    
                    Provide the final result {format_instructions}.
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    [TARGET TEXT START]
                    Como líder supremo da China, este indivíduo detém os títulos de Secretário-Geral do Partido Comunista, Presidente da República Popular e Presidente da Comissão Militar Central, liderando o país através de reformas econômicas e políticas significativas desde 2012.
                    [TARGET TEXT END]
                    """
                    )
                ),
                ("ai", textwrap.dedent(
                    """
                    ```
                    [TRANSLATED TEXT START]
                    As China's supreme leader, he holds the titles of General Secretary of the Communist Party, President of the People's Republic, and Chairman of the Central Military Commission, leading the country through significant economic and political reforms since 2012.
                    [TRANSLATED TEXT END]
                    ```
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    [TARGET TEXT START]
                    {target}
                    [TARGET TEXT END]
                    """
                    )
                )
                ],
                partial_variables={"format_instructions":format_instructions}
            )            

        self.chain_without_placeholder = prompt_without_placeholder | model | StrOutputParser()


    def parse(self, output):
        # Now parse using JSON (or regex)
        json_output = None
        try:
            parsed_result = output[output.find('[TRANSLATED TEXT START]') + len('[TRANSLATED TEXT START]'): output.find('[TRANSLATED TEXT END]')]

            json_output = {"result":parsed_result.strip()}
        except Exception as e:
            print("LLM output not directly JSON2. Need manual parsing.")
            print(e)
            print(output)
            return None

        return json_output

    def invoke(self, input, is_for_placeholder):
        if is_for_placeholder:
            return self.chain_with_placeholder.invoke(input)
        else:
            return self.chain_without_placeholder.invoke(input)

    def invoke_and_parse(self, input, is_for_placeholder):
        # if parse output is None repeat invoke and parse until parse is not None or max retries
        max_retries =5
        retries = 0
        parsed_output = None
        while parsed_output is None and retries < max_retries:
            retries += 1
            output = self.invoke(input, is_for_placeholder)
            parsed_output = self.parse(output)
        
        if retries == max_retries:
            print("Max retries reached. Returning None.")
            return None

        return parsed_output

    def translate_many_parallel(self, text_list: list, is_for_placeholder, max_workers: int = 5) -> Dict[str, str]:
        """
        Translate the same text into many languages concurrently using threads.
        """
        results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_text = {
                executor.submit(self.invoke_and_parse, {"target": text}, is_for_placeholder): (lang, text)
                for lang, text in text_list
            }
            for future in as_completed(future_to_text):
                lang, text = future_to_text[future]
                try:
                    parsed = future.result()
                    results[lang] = parsed["result"] if parsed else None
                except Exception as e:
                    results[lang] = f"Error: {e}"
        return results