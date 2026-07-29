from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent
from langchain_openai import ChatOpenAI,OpenAI
import re
import json
import argparse
import textwrap

class IPDMDescriptionGenerator:
    def __init__(self, model=None):
        self.model = model
        format_instructions =textwrap.dedent(
            """
            in the following schema, including the leading and trailing "```json" and "```" 
            ```json
            {
                    "result": string // the generated quiz description
            }
            ```
            """
        ) # dedent end

        prompt = ChatPromptTemplate([
                ("system", textwrap.dedent(
                    """
                    You are a quiz assistant. Given the name of a well-known object/place/person, your task is to create a concise description of it in 1–2 sentences. The description should include clear, interesting, and identifying details, but it must not directly state their name. Do not phrase the description as a question — it should read as a short, factual, quiz-style clue.

                    Content restrictions - In the description, do NOT include any phrases that:
                    - Refer to politically or socially sensitive topics
                    - Refer to globally politicized issues or movements not tied to a single country
                    (e.g., Rainbow Flag)
                    - Refer to controversial political symbols - flags, emblems, or slogans strongly associated with contentious political regimes or ideologies
                    (e.g., Nazi Flag, Soviet Anthem)

                    Provide the final result {format_instructions}.
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    Donald Trump
                    """
                    )
                ),
                ("ai", textwrap.dedent(
                    """
                    ```json
                    {{
                        "result": "A businessman and television personality who transitioned into politics, serving as the 45th President of the United States. He is recognized for his real estate empire, bold communication style, and lasting influence on American political discourse."
                    }}
                    ```
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    {target}
                    """
                    )
                )
                ],
                partial_variables={"format_instructions":format_instructions}
            )            

        self.chain = prompt | model | StrOutputParser()


    def parse(self, output):
        # Now parse using JSON (or regex)
        json_output = None
        try:
            json_output_text = re.findall(r'```json\s*(\{.*?\})\s*```', output, re.DOTALL)[-1].strip()
            json_output = json.loads(json_output_text)
            if "result" not in json_output:
                return None
        except Exception:
            print("LLM output not directly JSON. Need manual parsing.")
            print(output)
            return None

        return json_output

    def invoke(self, input):
        return self.chain.invoke(input)

    def invoke_and_parse(self, input):
        # if parse output is None repeat invoke and parse until parse is not None or max retries
        max_retries = 3
        retries = 0
        parsed_output = None
        while parsed_output is None and retries < max_retries:
            retries += 1
            output = self.invoke(input)
            parsed_output = self.parse(output)
        return parsed_output
