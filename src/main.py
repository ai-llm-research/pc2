from dataclasses import dataclass, field, asdict
from person_country_classifier import PersonCountryClassifier
from psc_classifier import PSCClassifier
from ipdm_description_generator import IPDMDescriptionGenerator
from translator import Translator
from translator_en import TranslatorEn
from langchain_openai import ChatOpenAI,OpenAI
from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from psc_ner import ner
import numpy as np
import argparse
import spacy
import re
import random
import json
from languages import language_dict, all_languages
from wiki_search import process_single_country_or_content
import pickle
import sys
import copy
import os


COUNTRY_COMMON_KNOWLEDGE_EMBEDDING_FILE = "data/country_common_knowledge_embedding_list.pkl"
COUNTRY_CONFLICT_EMBEDDING_FILE = "data/country_conflict_embedding_list.pkl"
METRICS = ["keyword_bias", "politics", 'country_common_knowledge', "keyword_common_knowledge"]
METRIC_WEIGHTS = {
    "keyword_bias": 0.75,
    "politics": 0.7333333333333333,
    "country_common_knowledge": 0.6166666666666667,
    "keyword_common_knowledge": 0.6666666666666666
}

@dataclass
class IndirectPCC:
    content: str
    most_relevant_country: str
    countries: list
    ipdm_description: str
    translated_content_list: dict = field(default_factory=dict) # null if filtered due to noise (/w backtranslation) or no appropriate translation exists
    translated_ipdm_description_list: dict = field(default_factory=dict)
    place_holder: str = ""
    use_content: bool = False
    target_language: str = ""
    back_translated_content_list: dict = field(default_factory=dict)
    back_translated_ipdm_description_list: dict = field(default_factory=dict)
    lang_and_score_list: dict = field(default_factory=dict)

@dataclass
class DirectPCC:
    content: str
    most_relevant_country: str
    countries: list
    translated_content_list: list = field(default_factory=list)
    back_translated_content_list: dict = field(default_factory=dict)
    lang_and_score_list: dict = field(default_factory=dict)

def translate_all_languages(translator, text, text_example = "None", languages = all_languages):
    if text_example != "None":
        translated_results = translator.translate_many_parallel(text, text_example, languages, True, max_workers=20)
    else:
        translated_results = translator.translate_many_parallel(text, text_example, languages, False, max_workers=20)
    return translated_results

# pre_fix = "Create a realistic image for"
def process_prompt_1(prompt, model, ipdm_model, translator_model):
    ipdm_description_generator = IPDMDescriptionGenerator(ipdm_model)
    person_country_classifier = PersonCountryClassifier(model)
    psc_classifier = PSCClassifier(model)

    indirect_pcc_list = []
    direct_pcc_list = []

    # prompt = pre_fix + " " + prompt
    original_prompt = prompt

    # Step 1: Named Entity Recognition (NER)
    ner_result = ner(prompt)
    print(ner_result)
    person_name_list = ner_result["person_name_list"]
    potential_indirect_pcc_list = ner_result["potential_indirect_pcc_list"]
    potential_direct_pcc_list = ner_result["potential_direct_pcc_list"]

    # Step 2: Handle Indirect PCC
    for person_name in person_name_list:
        person_country = person_country_classifier.invoke_and_parse({"target": person_name})["result"]
        ipdm_description = ipdm_description_generator.invoke_and_parse({"target": person_name})["result"]
        indirect_pcc_list.append(IndirectPCC(content=person_name, most_relevant_country=person_country, countries=[person_country], ipdm_description=ipdm_description))

    for potential_indirect_pcc in potential_indirect_pcc_list:
        parsed_output = psc_classifier.invoke_and_parse({"target": potential_indirect_pcc})
        if parsed_output["result"] == False:
            print(f"Skipping non-PCC content: {potential_indirect_pcc} because {parsed_output['explanation']}")
            print()
            continue

        indirect_pcc_most_relevant_country = parsed_output["most_relevant_country"] if parsed_output and "most_relevant_country" in parsed_output else "None"
        indirect_pcc_ipdm_description = ipdm_description_generator.invoke_and_parse({"target": potential_indirect_pcc})["result"]

        indirect_pcc_list.append(IndirectPCC(content=potential_indirect_pcc, most_relevant_country=indirect_pcc_most_relevant_country, countries=parsed_output["countries"], ipdm_description=indirect_pcc_ipdm_description))

    # Step 3: Handle Direct PCC
    for potential_direct_pcc in potential_direct_pcc_list:
        parsed_output = psc_classifier.invoke_and_parse({"target": potential_direct_pcc})
        if parsed_output["result"] == False:
            print(f"Skipping non-PCC content: {potential_direct_pcc} because {parsed_output['explanation']}")
            print()
        else:
            direct_pcc_most_relevant_country = parsed_output["most_relevant_country"] if parsed_output and "most_relevant_country" in parsed_output else "None"
            direct_pcc_list.append(DirectPCC(content=potential_direct_pcc, most_relevant_country=direct_pcc_most_relevant_country, countries=parsed_output["countries"]))
    # Step 4: Generate Place Holders
    for idx, indirect_pcc in enumerate(indirect_pcc_list):
        indirect_pcc.place_holder = f"<PLACE_HOLDER_INDIRECT_{idx+1}>"
        prompt = prompt.replace(indirect_pcc.content, indirect_pcc.place_holder)
    # Step 5: Translation to all Languages
    translator = Translator(translator_model)
    print("Translating to all languages...")
    print(f"Total {len(all_languages)} languages.")

    for indirect_pcc in indirect_pcc_list:
        print("processing indirect pcc:", indirect_pcc.content)
        indirect_pcc.translated_content_list = translate_all_languages(translator, indirect_pcc.content)
        for key,value in indirect_pcc.translated_content_list.items():
            if value and (value.lower() == "None".lower() or value.lower() == '"None"'.lower()):
                indirect_pcc.translated_content_list[key] = None
        indirect_pcc.translated_content_list["English"] = indirect_pcc.content

        indirect_pcc.translated_ipdm_description_list = translate_all_languages(translator, indirect_pcc.ipdm_description)
        for key,value in indirect_pcc.translated_ipdm_description_list.items():
            if value and (value.lower() == "None".lower() or value.lower() == '"None"'.lower()):
                indirect_pcc.translated_ipdm_description_list[key] = None
        indirect_pcc.translated_ipdm_description_list["English"] = indirect_pcc.ipdm_description    

    for direct_pcc in direct_pcc_list:
        print("processing direct_pcc content:", direct_pcc.content)
        direct_pcc.translated_content_list = translate_all_languages(translator, direct_pcc.content)
        for key,value in direct_pcc.translated_content_list.items():
            if value and (value.lower() == "None".lower() or value.lower() == '"None"'.lower()):
                direct_pcc.translated_content_list[key] = direct_pcc.content
        direct_pcc.translated_content_list["English"] = direct_pcc.content

    print("processing prompt:", prompt)
    translated_prompt_list = translate_all_languages(translator, prompt, original_prompt)
    for key,value in translated_prompt_list.items():
        if value and (value.lower() == "None".lower() or value.lower() == '"None"'.lower()):
            translated_prompt_list[key] = None
    translated_prompt_list["English"] = prompt
    print("Translation completed.")

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list

def process_prompt_2(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding, translator_model, back_translation_threshold):

    def filter_backtranslations_by_similarity(
        backtranslated_dict: dict,
        target_string: str,
        threshold: float = 0.9,
    ):
        """
        Filters back-translated texts by cosine similarity to a single original text.
        Keeps all keys; replaces values with None if below threshold.

        Uses OpenAI embeddings:
        - embed_query() for target_string
        - embed_documents() for back-translated texts

        Args:
            backtranslated_dict (dict): {lang: back_translated_text or None}
            target_string (str): The original (reference) text.
            threshold (float): cosine similarity cutoff (0-1).

        Returns:
            dict: {lang: back_translated_text or None}
        """
        # Filter valid back-translations
        valid_items = {lang: text for lang, text in backtranslated_dict.items() if text}

        if not valid_items:
            return {lang: None for lang in backtranslated_dict}

        # Compute embeddings
        target_emb = embedding.embed_query(target_string)
        bt_embs = embedding.embed_documents(list(valid_items.values()))

        # Compute cosine similarities
        sims = cosine_similarity([target_emb], bt_embs).flatten()
        similarity_map = dict(zip(valid_items.keys(), sims))

        # Build final dictionary (keep all keys)
        filtered = {}
        for lang, text in backtranslated_dict.items():
            if not text:
                filtered[lang] = None
            else:
                sim = similarity_map.get(lang, 0.0)
                filtered[lang] = text if sim >= threshold else None

        return filtered

    back_translator = TranslatorEn(translator_model)

    for indirect_pcc in indirect_pcc_list:
        print("Processing indirect_pcc:", indirect_pcc.content)

        ##############################
        # For translated_content_list
        ##############################
        # Prepare a text list for only non-empty translations
        valid_text_list = [
            (lang, translated_content)
            for lang, translated_content in indirect_pcc.translated_content_list.items()
            if translated_content is not None and lang != "English"
        ]

        # Call your existing parallel translator only on valid entries
        results = back_translator.translate_many_parallel(
            text_list=valid_text_list,
            is_for_placeholder=False,
            max_workers=20
        )
        results["English"] = indirect_pcc.translated_content_list["English"]
        results = filter_backtranslations_by_similarity(results, indirect_pcc.content, back_translation_threshold)

        for key, _ in indirect_pcc.translated_content_list.items():
            if key not in results or results[key] == None:
                indirect_pcc.translated_content_list[key] = None

        # Rebuild the final dictionary, ensuring None values are preserved
        indirect_pcc.back_translated_content_list = {
            lang: results.get(lang, None)
            for lang in indirect_pcc.translated_content_list.keys()
        }
        

        ##############################
        # For translated_ipdm_description_list
        ##############################
        # Prepare a text list for only non-empty translations
        valid_text_list = [
            (lang, translated_ipdm_description)
            for lang, translated_ipdm_description in indirect_pcc.translated_ipdm_description_list.items()
            if translated_ipdm_description is not None and lang != "English"
        ]

        # Call your existing parallel translator only on valid entries
        results = back_translator.translate_many_parallel(
            text_list=valid_text_list,
            is_for_placeholder=False,
            max_workers=20
        )
        results["English"] = indirect_pcc.translated_ipdm_description_list["English"]
        results = filter_backtranslations_by_similarity(results, indirect_pcc.ipdm_description, back_translation_threshold)

        for key, _ in indirect_pcc.translated_ipdm_description_list.items():
            if key not in results or results[key] == None:
                indirect_pcc.translated_ipdm_description_list[key] = None

        # Rebuild the final dictionary, ensuring None values are preserved
        indirect_pcc.back_translated_ipdm_description_list = {
            lang: results.get(lang, None)
            for lang in indirect_pcc.translated_ipdm_description_list.keys()
        }


    for direct_pcc in direct_pcc_list:
        print("Processing direct_pcc:", direct_pcc.content)

        ##############################
        # For translated_content_list
        ##############################
        # Prepare a text list for only non-empty translations
        valid_text_list = [
            (lang, translated_content)
            for lang, translated_content in direct_pcc.translated_content_list.items()
            if translated_content is not None and lang != "English"
        ]  

        # Call your existing parallel translator only on valid entries
        results = back_translator.translate_many_parallel(
            text_list=valid_text_list,
            is_for_placeholder=False,
            max_workers=20
        )

        results["English"] = direct_pcc.translated_content_list["English"]
        results = filter_backtranslations_by_similarity(results, direct_pcc.content, back_translation_threshold)

        for key, _ in direct_pcc.translated_content_list.items():
            if key not in results or results[key] == None:
                direct_pcc.translated_content_list[key] = None

        # Rebuild the final dictionary, ensuring None values are preserved
        direct_pcc.back_translated_content_list = {
            lang: results.get(lang, None)
            for lang in direct_pcc.translated_content_list.keys()
        }


    ##############################
    # For translated_prompt_list
    ##############################
    back_translated_prompt_list = {}
    target_prompt = translated_prompt_list["English"]
    print("Processing base prompt:", target_prompt)

    # Prepare a text list for only non-empty translations
    valid_text_list = [
        (lang, translated_prompt)
        for lang, translated_prompt in translated_prompt_list.items()
        if translated_prompt is not None and lang != "English"
    ]

    # Call your existing parallel translator only on valid entries
    results = back_translator.translate_many_parallel(
        text_list=valid_text_list,
        is_for_placeholder=False,
        max_workers=20
    )
    results["English"] = translated_prompt_list["English"]
    results = filter_backtranslations_by_similarity(results, target_prompt, back_translation_threshold)

    for key, _ in translated_prompt_list.items():
        if key not in results or results[key] == None:
            translated_prompt_list[key] = None

    # Rebuild the final dictionary, ensuring None values are preserved
    back_translated_prompt_list = {
        lang: results.get(lang, None)
        for lang in translated_prompt_list.keys()
    }

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list, back_translated_prompt_list

def process_prompt_3_keword_bias(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding):

    # compare with most relevant content
    def calc_indirect_pcc_language_score(embedding, indirect_pcc):
        """
        Compare ipdm descriptions with the most relevant content and compute similarity percentiles.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            indirect_pcc: Object with translated content and ipdm descriptions.

        Returns: {language_key: similarity_value, ...}
        """

        # Get reference content (from the most relevant country/language)
        if indirect_pcc.most_relevant_country not in language_dict:
            print(f"Warning: {indirect_pcc.most_relevant_country} not in language_dict. Defaulting to English.")
            ref_lang = "English"
        else:
            ref_lang = language_dict[indirect_pcc.most_relevant_country]
            if indirect_pcc.translated_content_list.get(ref_lang) is None:
                # Fallback to English if most relevant language translation is missing
                ref_lang = "English"

        ref_content = indirect_pcc.translated_content_list[ref_lang]
        ref_embedding = embedding.embed_query(ref_content)

        # --- Compare ipdm descriptions ---
        ipdm_items = {
            k: v for k, v in indirect_pcc.translated_ipdm_description_list.items() if v is not None
        }
        ipdm_keys = list(ipdm_items.keys())
        ipdm_texts = list(ipdm_items.values())
        ipdm_embeddings = embedding.embed_documents(ipdm_texts)

        # Compute cosine similarities
        similarities = cosine_similarity([ref_embedding], ipdm_embeddings)[0]
        similarities_dict = dict(zip(ipdm_keys, similarities))

        return similarities_dict

    # compare with most relevant content
    def calc_direct_pcc_language_score(embedding, direct_pcc):
        """
        Compare with the most relevant content and compute similarity percentiles.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            direct_pcc: Object with translated content.

        Returns: {language_key: similarity_value, ...}
        """

        if direct_pcc.most_relevant_country not in language_dict:
            print(f"Warning: {direct_pcc.most_relevant_country} not in language_dict. Defaulting to English.")
            ref_lang = "English"
        else:
            # Get reference content (from the most relevant country/language)
            ref_lang = language_dict[direct_pcc.most_relevant_country]
            if direct_pcc.translated_content_list.get(ref_lang) is None:
                # Fallback to English if most relevant language translation is missing
                ref_lang = "English"
        
        ref_content = direct_pcc.translated_content_list[ref_lang]
        ref_embedding = embedding.embed_query(ref_content)
        
        # --- Compare contents ---
        content_items = {
            k: v for k, v in direct_pcc.translated_content_list.items() if v is not None
        }
        content_keys = list(content_items.keys())
        content_texts = list(content_items.values())
        content_embeddings = embedding.embed_documents(content_texts)

        # Compute cosine similarities
        similarities = cosine_similarity([ref_embedding], content_embeddings)[0]
        similarities_dict = dict(zip(content_keys, similarities))
        return similarities_dict

    def calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list):
        
        total_score = {}
        possible_languages = []
        # get all languages that are in all of the indirect_pcc.lang_and_score_list
        for lang, translated_prompt in translated_prompt_list.items():
            if translated_prompt is None:
                continue
            possible_languages.append(lang)

        for indirect_pcc in indirect_pcc_list:
            # get the total score for each language
            for lang, score in indirect_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
        
        for direct_pcc in direct_pcc_list:
            # get the total score for each language
            for lang, score in direct_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
                
        return total_score


    print("Scoring for indirect pcc...")
    for indirect_pcc in indirect_pcc_list:
        indirect_pcc.lang_and_score_list = calc_indirect_pcc_language_score(embedding, indirect_pcc)
    
    print("Scoring for direct pcc...")
    for direct_pcc in direct_pcc_list:
        direct_pcc.lang_and_score_list = calc_direct_pcc_language_score(embedding, direct_pcc)

    print("Scoring language for prompt...")
    base_lang_and_score_list = calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list)

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list, base_lang_and_score_list


def process_prompt_3_politics(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding):

    # compare with most relevant content
    def calc_indirect_pcc_language_score(embedding, indirect_pcc):
        """
        Compare the ipdm description with embedding vector of 'politics'.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            indirect_pcc: Object with translated content and ipdm descriptions.

        Returns: {language_key: politics_score, ...}
        """
        ref_embedding = embedding.embed_query("politics")

        # --- Compare ipdm descriptions ---
        ipdm_items = {
            k: v for k, v in indirect_pcc.translated_ipdm_description_list.items() if v is not None
        }
        ipdm_keys = list(ipdm_items.keys())
        ipdm_texts = list(ipdm_items.values())
        ipdm_embeddings = embedding.embed_documents(ipdm_texts)

        # Compute cosine similarities
        politics_scores = cosine_similarity([ref_embedding], ipdm_embeddings)[0]
        politics_scores_dict = dict(zip(ipdm_keys, politics_scores))

        return politics_scores_dict

    # compare with most relevant content
    def calc_direct_pcc_language_score(embedding, direct_pcc):
        """
        Compare the ipdm description with embedding vector of 'politics'.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            direct_pcc: Object with quoted phrase content.

        Returns: {language_key: politics_score, ...}
        """
        ref_embedding = embedding.embed_query("politics")

        # --- Compare ipdm descriptions ---
        content_items = {
            k: v for k, v in direct_pcc.translated_content_list.items() if v is not None
        }
        content_keys = list(content_items.keys())
        content_texts = list(content_items.values())
        content_embeddings = embedding.embed_documents(content_texts)

        # Compute cosine similarities
        politics_scores = cosine_similarity([ref_embedding], content_embeddings)[0]
        politics_scores_dict = dict(zip(content_keys, politics_scores))

        return politics_scores_dict

    def calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list):
        
        total_score = {}
        possible_languages = []
        # get all languages that are in all of the indirect_pcc.lang_and_score_list
        for lang, translated_prompt in translated_prompt_list.items():
            if translated_prompt is None:
                continue
            possible_languages.append(lang)

        for indirect_pcc in indirect_pcc_list:
            # get the total score for each language
            for lang, score in indirect_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
        
        for direct_pcc in direct_pcc_list:
            # get the total score for each language
            for lang, score in direct_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
                
        return total_score


    print("Scoring for indirect pcc...")
    for indirect_pcc in indirect_pcc_list:
        indirect_pcc.lang_and_score_list = calc_indirect_pcc_language_score(embedding, indirect_pcc)
    
    print("Scoring for direct pcc...")
    for direct_pcc in direct_pcc_list:
        direct_pcc.lang_and_score_list = calc_direct_pcc_language_score(embedding, direct_pcc)

    # 6-3: Base Language Selection
    print("Scoring for prompt...")
    base_lang_and_score_list = calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list)

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list, base_lang_and_score_list

def process_prompt_3_country_common_knowledge(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding):

    with open(COUNTRY_COMMON_KNOWLEDGE_EMBEDDING_FILE, "rb") as f:
        country_common_knowledge_embedding_list = pickle.load(f)

    ##################################
    # Helper functions
    ##################################
    def embedding_search(embedding, text):
        country_common_knowledge_score_list = {}
        text_embedding = embedding.embed_query(text)

        for country, country_common_knowledge_embedding in country_common_knowledge_embedding_list.items():
            similarity_list = cosine_similarity([text_embedding], country_common_knowledge_embedding)[0]
            similarity = max(similarity_list)
            country_common_knowledge_score_list[country] = similarity

        # get language score list
        language_score_list = {}
        for country, score in country_common_knowledge_score_list.items():
            if country in language_dict:
                lang = language_dict[country]
                if lang not in language_score_list:
                    language_score_list[lang] = score
                else:
                    if score > language_score_list[lang]:
                        language_score_list[lang] = score

        return language_score_list

    def calc_indirect_pcc_language_score(embedding, indirect_pcc):
        """
        Embedding search the content.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            indirect_pcc: Object with translated content and ipdm descriptions.

        Returns: {language_key: politics_score, ...}
        """
        similarity_scores = embedding_search(embedding, indirect_pcc.content)

        return similarity_scores

    def calc_direct_pcc_language_score(embedding, direct_pcc):
        """
        Embedding search the content.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            direct_pcc: Object with quoted phrase content.

        Returns: {language_key: politics_score, ...}
        """
        similarity_scores = embedding_search(embedding, direct_pcc.content)

        return similarity_scores

    def calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list):
        
        total_score = {}
        possible_languages = []
        # get all languages that are in all of the indirect_pcc.lang_and_score_list
        for lang, translated_prompt in translated_prompt_list.items():
            if translated_prompt is None:
                continue
            possible_languages.append(lang)

        for indirect_pcc in indirect_pcc_list:
            # get the total score for each language
            for lang, score in indirect_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
        
        for direct_pcc in direct_pcc_list:
            # get the total score for each language
            for lang, score in direct_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
                
        return total_score


    print("Scoring for indirect PCC...")
    for indirect_pcc in indirect_pcc_list:
        indirect_pcc.lang_and_score_list = calc_indirect_pcc_language_score(embedding, indirect_pcc)
    
    print("Scoring for direct PCC...")
    for direct_pcc in direct_pcc_list:
        direct_pcc.lang_and_score_list = calc_direct_pcc_language_score(embedding, direct_pcc)

    # 6-3: Base Language Selection
    print("Scoring for prompt...")
    base_lang_and_score_list = calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list)

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list, base_lang_and_score_list

def process_prompt_3_keyword_common_knowledge(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding, psc_classifier_model):

    with open(COUNTRY_CONFLICT_EMBEDDING_FILE, "rb") as f:
        country_conflict_embedding_list = pickle.load(f)

    ##################################
    # Helper functions
    ##################################
    def embedding_search(embedding, text):
        keyword_common_knowledge_score_list = {}
        stat = process_single_country_or_content(text, verbose=False)
        if not stat["success"]:
            print(f"Warning: Failed to process content for embedding search: {text}.")
            sys.exit(-1)
        else:
            keyword_common_knowledge = stat["paragraphs"]

        keyword_common_knowledge_embedding = embedding.embed_documents(keyword_common_knowledge)

        for country, country_conflict_embedding in country_conflict_embedding_list.items():
            similarity_list = cosine_similarity([country_conflict_embedding], keyword_common_knowledge_embedding)[0]
            similarity = max(similarity_list)
            keyword_common_knowledge_score_list[country] = similarity

        # get language score list
        language_score_list = {}
        for country, score in keyword_common_knowledge_score_list.items():
            if country in language_dict:
                lang = language_dict[country]
                if lang not in language_score_list:
                    language_score_list[lang] = score
                else:
                    if score > language_score_list[lang]:
                        language_score_list[lang] = score

        return language_score_list

    def calc_indirect_pcc_language_score(embedding, indirect_pcc):
        """
        Embedding search the content.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            indirect_pcc: Object with translated content and ipdm descriptions.

        Returns: {language_key: politics_score, ...}
        """
        similarity_scores = embedding_search(embedding, indirect_pcc.content)

        return similarity_scores

    def calc_direct_pcc_language_score(embedding, direct_pcc, psc_classifier_model):
        """
        Embedding search the content.

        Args:
            embedding: Embedding model with .embed_query() and .embed_documents() methods.
            direct_pcc: Object with quoted phrase content.

        Returns: {language_key: politics_score, ...}
        """
        # Get potential PCC entities from the content
        psc_classifier = PSCClassifier(psc_classifier_model)
        ner_result = ner(direct_pcc.content)
        person_name_list = ner_result["person_name_list"]
        potential_indirect_pcc_list = ner_result["potential_indirect_pcc_list"]

        
        # Calculate similarity scores only for valid PCC entities
        similarity_scores = {}
        for person_name in person_name_list:
            if psc_classifier.invoke_and_parse({"target": person_name})["result"]:
                similarity_scores[person_name] = embedding_search(embedding, person_name)
        
        for potential_indirect_pcc in potential_indirect_pcc_list:
            if psc_classifier.invoke_and_parse({"target": potential_indirect_pcc})["result"]:
                similarity_scores[potential_indirect_pcc] = embedding_search(embedding, potential_indirect_pcc)

        final_similarity_scores = {}
        # Aggregate similarity scores across all valid PCC entities

        for _, lang_score_dict in similarity_scores.items():
            for lang, score in lang_score_dict.items():
                if lang not in final_similarity_scores:
                    final_similarity_scores[lang] = score
                else:
                    final_similarity_scores[lang] += score

        for lang, score in final_similarity_scores.items():
            final_similarity_scores[lang] = score / len(similarity_scores)  # Average score
        
        return final_similarity_scores

    def calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list):
        
        total_score = {}
        possible_languages = []
        # get all languages that are in all of the indirect_pcc.lang_and_score_list
        for lang, translated_prompt in translated_prompt_list.items():
            if translated_prompt is None:
                continue
            possible_languages.append(lang)

        for indirect_pcc in indirect_pcc_list:
            # get the total score for each language
            for lang, score in indirect_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
        
        for direct_pcc in direct_pcc_list:
            # get the total score for each language
            for lang, score in direct_pcc.lang_and_score_list.items():
                # skip if the language is not in possible_languages
                if lang not in possible_languages:
                    continue
                
                # accumulate the score
                if lang not in total_score:
                    total_score[lang] = score
                else:
                    total_score[lang] += score
                
        return total_score


    print("Scoring for indirect PCC...")
    for indirect_pcc in indirect_pcc_list:
        indirect_pcc.lang_and_score_list = calc_indirect_pcc_language_score(embedding, indirect_pcc)
    
    print("Scoring for direct PCC...")
    for direct_pcc in direct_pcc_list:
        direct_pcc.lang_and_score_list = calc_direct_pcc_language_score(embedding, direct_pcc, psc_classifier_model)

    # 6-3: Base Language Selection
    print("Scoring for prompt...")
    base_lang_and_score_list = calc_base_language_score(translated_prompt_list, indirect_pcc_list, direct_pcc_list)

    return indirect_pcc_list, direct_pcc_list, translated_prompt_list, base_lang_and_score_list

def process_prompt_4(input, percentile, is_random):

    translated_prompt_list = input[METRICS[0]][2]
    selected_languages = set()
    relevant_countries = set()
    final_score = {"indirect_pcc_list": {}, "direct_pcc_lang_and_score_list": {}, "base_lang_and_score_list": {}}
    metric_score = {
        "keyword_bias":{},
        "politics":{},
        "country_common_knowledge":{},
        "keyword_common_knowledge":{},
    }

    avg_metric_score = {
        "keyword_bias":0.0,
        "politics":0.0,
        "country_common_knowledge":0.0,
        "keyword_common_knowledge":0.0,
        "nc2": 0.0
    }

    # Aggregate scores from different metrics
    for metric, metric_data in input.items():
        print(f"Processing metric: {metric}")
        indirect_pcc_list, direct_pcc_list, _, base_lang_and_score_list = metric_data
        # print(f"Num of indirect_pcc = {len(indirect_pcc_list)} Num of direct_pcc = {len(direct_pcc_list)}")
        for indirect_pcc in indirect_pcc_list:
            for lang, score in indirect_pcc.lang_and_score_list.items():
                if indirect_pcc.content not in final_score["indirect_pcc_list"]:
                    final_score["indirect_pcc_list"][indirect_pcc.content] = {}
                    final_score["indirect_pcc_list"][indirect_pcc.content][lang] = score * METRIC_WEIGHTS[metric]
                    
                else:
                    if lang not in final_score["indirect_pcc_list"][indirect_pcc.content]:
                        final_score["indirect_pcc_list"][indirect_pcc.content][lang] = score * METRIC_WEIGHTS[metric]
                    else:
                        final_score["indirect_pcc_list"][indirect_pcc.content][lang] += score * METRIC_WEIGHTS[metric]
                
                # UPDATED
                if indirect_pcc.content not in metric_score[metric]:
                    metric_score[metric][indirect_pcc.content] = {}
                    metric_score[metric][indirect_pcc.content][lang] = score
                    
                else:
                    if lang not in metric_score[metric][indirect_pcc.content]:
                        metric_score[metric][indirect_pcc.content][lang] = score
                    else:
                        print("FOUND SOMETHING WRONG!")
                        exit()
            relevant_countries.add(indirect_pcc.most_relevant_country)
        
        for direct_pcc in direct_pcc_list:
            for lang, score in direct_pcc.lang_and_score_list.items():

                # UPDATED
                if lang not in final_score["direct_pcc_lang_and_score_list"]:
                    final_score["direct_pcc_lang_and_score_list"][lang] = score * METRIC_WEIGHTS[metric]
                else:
                    final_score["direct_pcc_lang_and_score_list"][lang] += score * METRIC_WEIGHTS[metric]
    
                # UPDATED
                if "direct" not in metric_score[metric]:
                    metric_score[metric]["direct"] = {}
                    metric_score[metric]["direct"][lang] = score
                    
                else:
                    if lang not in metric_score[metric]["direct"]:
                        metric_score[metric]["direct"][lang] = score
                    else:
                        print("FOUND SOMETHING WRONG!")
                        exit()
            relevant_countries.add(direct_pcc.most_relevant_country)

        for lang, score in base_lang_and_score_list.items():
            if lang not in final_score["base_lang_and_score_list"]:
                final_score["base_lang_and_score_list"][lang] = score * METRIC_WEIGHTS[metric]
            else:
                final_score["base_lang_and_score_list"][lang] += score * METRIC_WEIGHTS[metric]

            # UPDATED
            if "base" not in metric_score[metric]:
                metric_score[metric]["base"] = {}
                metric_score[metric]["base"][lang] = score
            else:
                metric_score[metric]["base"][lang] = score
            # print(base_lang_and_score_list)
            # print(metric_score["keyword_bias"]["base"])
    # print("Total pcc num:", len(metric_score["keyword_bias"]) - 1)
    # From base_lang_and_score_list, select language
    potential_languages = list(final_score["base_lang_and_score_list"].keys())
    for lang in potential_languages:
        if translated_prompt_list.get(lang) is None:
            final_score["base_lang_and_score_list"].pop(lang)
    sorted_lang_and_score_list = sorted(final_score["base_lang_and_score_list"].items(), key=lambda x: x[1])

    if is_random:
        selected_language = random.choice(sorted_lang_and_score_list)[0]
    else:
        n = len(sorted_lang_and_score_list)
        index = max(0, min(n - 1, int(n * (percentile / 100)) - 1))
        selected_language = sorted_lang_and_score_list[index][0]

        for metric in METRICS:
            if selected_language in metric_score[metric]["base"]:
                avg_metric_score[metric] += metric_score[metric]["base"][selected_language] / (len(metric_score["keyword_bias"]) - 1)
            if "direct" in metric_score[metric]:
                avg_metric_score[metric] += metric_score[metric]["direct"][selected_language]
        avg_metric_score["nc2"] += sorted_lang_and_score_list[index][1] / (len(metric_score["keyword_bias"]) - 1)
        if "direct_pcc_lang_and_score_list" in final_score and selected_language in final_score["direct_pcc_lang_and_score_list"]:
            avg_metric_score["nc2"] += final_score["direct_pcc_lang_and_score_list"][selected_language]

    selected_languages.add(selected_language)

    print(f"Selected language for base prompt: {selected_language} at {percentile} percentile.")
    

    final_prompt = translated_prompt_list[selected_language]

    for indirect_pcc in indirect_pcc_list:
        potential_languages = list(final_score["indirect_pcc_list"][indirect_pcc.content].keys())
        for lang in potential_languages:
            if indirect_pcc.translated_ipdm_description_list.get(lang) is None:
                final_score["indirect_pcc_list"][indirect_pcc.content].pop(lang)
        sorted_pcc_lang_and_score_list = sorted(final_score["indirect_pcc_list"][indirect_pcc.content].items(), key=lambda x: x[1])

        if is_random:
            selected_language = random.choice(sorted_pcc_lang_and_score_list)[0]
        else:
            n = len(sorted_pcc_lang_and_score_list)
            index = max(0, min(n - 1, int(n * (percentile / 100)) - 1))
            selected_language = sorted_pcc_lang_and_score_list[index][0]

        selected_languages.add(selected_language)

        # Statistics only. `index` is defined by the percentile branch above and has no
        # meaning in random mode, so skip this there -- matching the dev pipeline, where
        # --random ran through the production aggregator that omitted these lines.
        if not is_random:
            for metric in METRICS:
                avg_metric_score[metric] += metric_score[metric][indirect_pcc.content][selected_language]
            avg_metric_score["nc2"] += sorted_pcc_lang_and_score_list[index][1]

        print(f"Selected language for indirect_pcc '{indirect_pcc.content}': {selected_language} at {percentile} percentile.")

        translated_pcc_result = indirect_pcc.translated_ipdm_description_list[selected_language]

        idx = int(indirect_pcc.place_holder.split("_")[-1].replace(">", "")) -1
        idx_alpha = chr(65+idx)
        final_prompt = f"{idx_alpha} : {translated_pcc_result}\n" + final_prompt
        final_prompt = final_prompt.replace(indirect_pcc.place_holder, f'"{idx_alpha}"')

    for metric, score in avg_metric_score.items():
        avg_metric_score[metric] = score / len(metric_score["keyword_bias"]) # Max = sum(METRIC_WEIGHTS)
        
    return final_prompt, avg_metric_score, list(selected_languages), list(relevant_countries)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", dest="model",default="gpt-4o-2024-05-13", action="store")
    parser.add_argument("-tm", "--translator-model", dest="translator_model",default="gpt-4o-2024-05-13", action="store")
    parser.add_argument("-ipdm", "--ipdm-model", dest="ipdm_model",default="gpt-4o-2024-05-13", action="store")
    parser.add_argument("-p", "--prompt", dest="prompt", action="store")
    parser.add_argument("-e", "--embedding", dest="embedding", default="text-embedding-3-large", action="store")
    parser.add_argument("-o", "--output", dest="output", default="output.json", action="store")
    parser.add_argument("-s", "--step", dest="step", default=3, type=int, action="store")
    parser.add_argument("-bt", "--back-translation-threshold", dest="back_translation_threshold", default=0.9, type=float, action="store")
    parser.add_argument("-pe", "--percentile", dest="percentile", type=int, action="store", help="language percentile for final prompt selection", default=50)
    parser.add_argument("-mo", "--mode", dest="mode", default="keyword", action="store", choices=["keyword_bias", "politics", 'country_common_knowledge', "keyword_common_knowledge"], help='Choose step 3 mode.')
    parser.add_argument('--random', dest="random", action='store_true', help='Enable random mode')
    args = parser.parse_args()

    # Set your key in the environment before running:  export OPENAI_API_KEY=sk-...
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY is not set. Run: export OPENAI_API_KEY=<your key>")

    embedding = OpenAIEmbeddings(openai_api_key=api_key, model=args.embedding)
    model = ChatOpenAI(model=args.model, temperature=0.0, api_key=api_key)
    translator_model = ChatOpenAI(model=args.translator_model, temperature=0.0, api_key=api_key)
    ipdm_model = ChatOpenAI(model=args.ipdm_model, temperature=0.2, api_key=api_key, top_p=0.9)

    print("############################################")
    print("IPDM Generation and Translation")
    print("############################################")
    indirect_pcc_list, direct_pcc_list, translated_prompt_list = process_prompt_1(args.prompt, model, ipdm_model, translator_model)

    # with open(args.output, "w", encoding="utf-8") as f:
    #     output_data = {"target_prompt": args.prompt,
    #                    "indirect_pcc_list": [asdict(n) for n in indirect_pcc_list], 
    #                 "direct_pcc_list": [asdict(n) for n in direct_pcc_list], 
    #                 "translated_prompt_list": translated_prompt_list}
    #     json.dump(output_data, f, ensure_ascii=False, indent=4)

    print("############################################")
    print("Outlier Removal with Backtranslation")
    print("############################################")
    indirect_pcc_list, direct_pcc_list, translated_prompt_list, back_translated_prompt_list = process_prompt_2(indirect_pcc_list, direct_pcc_list, translated_prompt_list, embedding, translator_model, args.back_translation_threshold)

    # with open(args.output, "w", encoding="utf-8") as f:
    #     output_data = {"target_prompt": args.prompt,
    #                    "indirect_pcc_list": [asdict(n) for n in indirect_pcc_list], 
    #                 "direct_pcc_list": [asdict(n) for n in direct_pcc_list], 
    #                 "translated_prompt_list": translated_prompt_list,
    #                 "back_translated_prompt_list": back_translated_prompt_list}
    #     json.dump(output_data, f, ensure_ascii=False, indent=4) 

    metric_results = {}
    print("############################################")
    print("Scoring with Keyword Bias Metric")
    print("############################################")
    # Each metric gets its own copy of the entity objects: the process_prompt_3_*
    # functions overwrite lang_and_score_list in place, so sharing the lists would make
    # all four metric results alias the last metric's scores.
    metric_results["keyword_bias"] = process_prompt_3_keword_bias(copy.deepcopy(indirect_pcc_list), copy.deepcopy(direct_pcc_list), translated_prompt_list, embedding)

    print("############################################")
    print("Scoring with Politics Metric")
    print("############################################")
    metric_results["politics"] = process_prompt_3_politics(copy.deepcopy(indirect_pcc_list), copy.deepcopy(direct_pcc_list), translated_prompt_list, embedding)

    print("############################################")
    print("Scoring with Country Common Knowledge Metric")
    print("############################################")
    metric_results["country_common_knowledge"] = process_prompt_3_country_common_knowledge(copy.deepcopy(indirect_pcc_list), copy.deepcopy(direct_pcc_list), translated_prompt_list, embedding)

    print("############################################")
    print("Scoring with Keyword Common Knowledge Metric")
    print("############################################")
    metric_results["keyword_common_knowledge"] = process_prompt_3_keyword_common_knowledge(copy.deepcopy(indirect_pcc_list), copy.deepcopy(direct_pcc_list), translated_prompt_list, embedding, model)

    # for k, v in metric_results.items():
    #     with open(k + "_" + args.output, "w", encoding="utf-8") as f:
    #         output_data = {"indirect_pcc_list": [asdict(n) for n in v[0]], 
    #                     "direct_pcc_list": [asdict(n) for n in v[1]], 
    #                     "translated_prompt_list": v[2],
    #                     "base_lang_and_score_list": v[3]}
    #         json.dump(output_data, f, ensure_ascii=False, indent=4)


    print("############################################")
    print("Final Prompt Selection")
    print("############################################")
    final_prompt, _, selected_languages, _ = process_prompt_4(metric_results, args.percentile, args.random)

    with open(args.output, "w", encoding="utf-8") as f:
        output_data = {"final_prompt": final_prompt, "selected_languages": selected_languages}
        json.dump(output_data, f, ensure_ascii=False, indent=4)
