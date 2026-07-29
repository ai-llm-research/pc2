from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent
from langchain_openai import ChatOpenAI,OpenAI
import re
import json
import argparse
import textwrap

class PersonCountryClassifier:
    def __init__(self, model=None):
        self.model = model
        format_instructions =textwrap.dedent(
            """
            in the following schema, including the leading and trailing "```json" and "```" 
            ```json
            {
                    "result": string, // the resulting country name
                    "explanation": string // brief explanation of the reasoning behind the decision
            }
            ```
            """
        ) # dedent end

        prompt = ChatPromptTemplate([
                ("system", textwrap.dedent(
                    """
                    Your task is to identify the most related country name from the name of a public figure, such as a politician or celebrity. Instead of directly using the name or nationality of the person, you should infer the country based on their well-known associations, roles, or significant actions.

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
                    {{"result": "United States"}}
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