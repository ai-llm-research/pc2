import sys
import os
sys.path.append('src/')

import fasttext
import numpy as np
import re
from psc_classifier import PSCClassifier
from langchain_openai import ChatOpenAI,OpenAI
from translator import Translator
from languages import language_dict, language_dict_google
from deep_translator import GoogleTranslator
import argparse
import json

class RelLangFilter:
    def __init__(self, psc_classifier_model, lang_detection_model_path="lid.176.bin"):
        self.lang_detection_model = fasttext.load_model(lang_detection_model_path)
        self.psc_classifier = PSCClassifier(psc_classifier_model)


    def compartment_text(self, text):
        ###########
        # Compartment text based on language.
        # Currently, we split text into sentences based on punctuation and newlines, then group sentences by predicted language.
        # Future improvement: use a more sophisticated method to compartmentalize text.
        ###########
        pattern = r'(?<=[\.。｡․꓿።!?！？])\s*|\n+'
        sentences = re.split(pattern, text)
        sentences = [s.strip() for s in sentences if s.strip()]

        compartments = []
        cur_lang = None
        cur_compartment = ""
        for sentence in sentences:
            if len(sentence) < 3 and cur_lang is not None:
                lang_pred = cur_lang
            else:
                pred = self.lang_detection_model.predict(sentence)
                lang_pred = pred[0][0]

            if cur_lang is None:
                cur_lang = lang_pred
                cur_sentences = [sentence]
            elif lang_pred == cur_lang:
                cur_sentences.append(sentence)
            else:
                compartments.append(" ".join(cur_sentences))
                cur_lang = lang_pred
                cur_sentences = [sentence]

        if cur_sentences:
            compartments.append(" ".join(cur_sentences))

        return compartments

    def get_most_relevant_country(self, text):
        input = {
        "target": text,
        }
        result = self.psc_classifier.invoke_and_parse(input)
        if not result:
            return "United States"
        else:
            return result["most_relevant_country"]

    def translate(self, text, language):
        translated_text = GoogleTranslator(source='auto', target=language).translate(text)
        return translated_text

    def filter(self, target_prompt):
        compartments = self.compartment_text(target_prompt)

        new_prompt = ""
        for compartment in compartments:
            if new_prompt.strip():
                new_prompt += "\n\n"

            print(f"Analyzing compartment: {compartment}")
            country = self.get_most_relevant_country(compartment)
            print(f"Compartment: {compartment}\nMost Relevant Country: {country}\n")
            if not country or country == "None":
                language = "en"
            else:
                language = language_dict_google.get(country, "en")

            print(f"Translating compartment to {language}...")
            translated_text = self.translate(compartment, language)
            print(f"Translated Text: {translated_text}\n")
            new_prompt += translated_text

        return new_prompt
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input", action="store")
    parser.add_argument("-o", "--output", dest="output", action="store")
    args = parser.parse_args()
    
    with open(args.input, "r") as f:
        input_data = json.load(f)

    target_prompt = input_data["final_prompt"]
    
    # Set your key in the environment before running:  export OPENAI_API_KEY=sk-...
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=<your key>")
    model = ChatOpenAI(model="gpt-4o-2024-05-13", temperature=0.0, api_key=api_key)

    filter = RelLangFilter(model, lang_detection_model_path="lid.176.bin")
    filtered_result = filter.filter(target_prompt)

    # FOR TESTING
    # print(filtered_result)

    # print(f"Writing output to {args.output}...")
    with open(args.output, "w") as f:
        output_data = {"final_prompt": filtered_result}
        f.write(json.dumps(output_data, ensure_ascii=False, indent=4))
