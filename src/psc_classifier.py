from langchain_openai import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from textwrap import dedent
from langchain_openai import ChatOpenAI,OpenAI
import re
import json
import argparse
import textwrap

class PSCClassifier:
    def __init__(self, model=None):
        self.model = model
        format_instructions =textwrap.dedent(
            """
            in the following schema, including the leading and trailing "```json" and "```" 
            ```json
            {
                "result": boolean, // true if the text is relevant to one or more countries, false otherwise
                "countries": list, // list of country names relevant to the text, empty if none
                "most_relevant_country": string, // the single most relevant country to the text, empty if none
                "explanation": string // brief explanation of the reasoning behind the decision
            }
            ```
            """
        ) # dedent end

        prompt = ChatPromptTemplate([
                ("system", textwrap.dedent(
                    """
                    You are an information extraction assistant. Your task is to analyze user input and determine whether it is related to:
                    - A specific country (or countries), and/or
                    - A politically or socially sensitive topic.
                    Also determine the relevant countries and also the most relevant country.

                    Rules:
                    - If the text explicitly mentions a country → mark it relevant (e.g., "France", "Japan").
                    - If the text refers to a historical regime, symbol, or entity uniquely associated with a regime,
                    map it to its closest modern equivalent:
                        - Nazi Germany → Germany
                        - Soviet Union → Russia
                        - Ottoman Empire → Turkey
                        - Persia → Iran
                    - Symbols or terms uniquely tied to these regimes (e.g., "Nazi flag", "Soviet anthem", "Ottoman sultan") 
                    should also be considered relevant and mapped.
                    - If the text involves global-wide issues without specific country ties (e.g., "rainbow flag"), map to United States.
                    - If the text is completely generic (e.g., "flag", "government", "currency") with no unique tie, mark as not relevant.


                    Provide the final result {format_instructions}.
                    """
                    )
                ),
                ("user", textwrap.dedent(
                    """
                    Crimea annexation
                    """
                    )
                ),
                ("ai", textwrap.dedent(
                    """
                    ```json
                    {{
                        "result": true,
                        "countries": ["Russia", "Ukraine"],
                        "most_relevant_country": "Ukraine",
                        "explanation": "The annexation of Crimea by Russia in 2014 created a territorial conflict with Ukraine, leading to ongoing tensions."
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