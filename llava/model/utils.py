import torch
import re
import difflib

from transformers import AutoConfig


def auto_upgrade(config):
    cfg = AutoConfig.from_pretrained(config)
    if "llava" in config and "llava" not in cfg.model_type:
        assert cfg.model_type == "llama"
        print("You are using newer LLaVA code base, while the checkpoint of v0 is from older code base.")
        print("You must upgrade the checkpoint to the new code base (this can be done automatically).")
        confirm = input("Please confirm that you want to upgrade the checkpoint. [Y/N]")
        if confirm.lower() in ["y", "yes"]:
            print("Upgrading checkpoint...")
            assert len(cfg.architectures) == 1
            setattr(cfg.__class__, "model_type", "llava")
            cfg.architectures[0] = "LlavaLlamaForCausalLM"
            cfg.save_pretrained(config)
            print("Checkpoint upgraded.")
        else:
            print("Checkpoint upgrade aborted.")
            exit(1)


import spacy
nlp = spacy.load("en_core_web_trf")

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(list(stopwords.words('english')) + list(spacy.lang.en.stop_words.STOP_WORDS))

def parse_object_nouns_and_action_verbs(input_paragraph):
    """
    Parse the input paragraph to get the action verbs and noun objects.
    """
    object_nouns, action_verbs = [], []
    doc = nlp(input_paragraph)
    
    def is_action_verb(token):
        if token.pos_ == "VERB":
            has_direct_object = any(child.dep_ == "dobj" for child in token.children)  # 檢查是否有直接賓語
            has_prep_and_object = any(child.dep_ == "prep" and any(grandchild.dep_ == "pobj" for grandchild in child.children) for child in token.children)  # 檢查是否有介詞及其受詞結構
            has_subject = any(child.dep_ in ["nsubj", "nsubjpass"] for child in token.children)  # 檢查是否有主語（包括主動和被動）
            
            if has_direct_object or has_prep_and_object or has_subject:
                return True
        return False
    
    def is_stopword(token):
        for stop_word in stop_words:
            if token.text.lower() == stop_word:
                return True
        return False

    for token in doc:
        if is_action_verb(token) and not is_stopword(token):
            action_verbs.append(token.text)
        
        # 判斷token是否為直接受詞、介詞受詞、主語或其他物件類型，且詞性必須是名詞、專有名詞或代詞
        if token.dep_ in ["dobj", "pobj", "nsubj", "obj"] and token.pos_ in ["NOUN", "PROPN", "PRON"] and not is_stopword(token):
            object_nouns.append(token.text)
    
    # find the most similar substring in the input_paragraph for each object_noun and action_verb
    object_nouns = [find_most_similar_substring(input_paragraph, noun) for noun in object_nouns]
    action_verbs = [find_most_similar_substring(input_paragraph, verb) for verb in action_verbs]
    return object_nouns, action_verbs


def get_phrase_indices(tokenizer, words, caption_ids, caption_prefix_length):
    """
    Get (1) the words having corresponding indices in the caption and (2) the indices of the words in the caption.

    Args:
        tokenizer (transformers.PreTrainedTokenizer): The tokenizer.
        words (List[str]): The words to get the indices.
        caption_ids (torch.Tensor): The tokenized caption.
        caption_prefix_length (int): The length of the caption prefix.

    Returns:
        _type_: _description_
    """
    all_phrase_indices = []
    returned_words = []
    for word in words:
        if tokenizer.__class__.__name__ == "Qwen2TokenizerFast":
            word = f" {word}"
        else:
            raise NotImplementedError(f"Unsupported tokenizer: {tokenizer.__class__.__name__}")
        token_ids = torch.tensor(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(word))).to(caption_ids.device)
        
        token_len = len(token_ids)
        single_phrase_indices = []
        for i in range(len(caption_ids) - token_len + 1):
            if torch.equal(caption_ids[i:i+token_len], token_ids):
                matched_indices = list(range(i, i+token_len))
                matched_indices = [idx + caption_prefix_length for idx in matched_indices]
                single_phrase_indices += matched_indices
        if single_phrase_indices != []:
            all_phrase_indices.append(single_phrase_indices)
            returned_words.append(word)
        else:
            print(f"WARNING: {word} has token_ids: {token_ids} but no indices")
    return returned_words, all_phrase_indices


def find_most_similar_substring(input_paragraph, given_str):
    # Adjust the regex to include hyphens and apostrophes in words
    pattern = r"\b\w+(?:['\-]\w+)*\b"
    words = re.findall(pattern, input_paragraph)
    given_words = re.findall(pattern, given_str)
    n = len(given_words)

    # Generate substrings (n-grams) of lengths n-1, n, and n+1
    substrings = [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]
    if n > 1:
        substrings += [' '.join(words[i:i+n-1]) for i in range(len(words)-n+2)]
    substrings += [' '.join(words[i:i+n+1]) for i in range(len(words)-n)]

    # Find the substring with the highest similarity ratio
    max_ratio = 0
    best_match = given_str
    for substring in substrings:
        ratio = difflib.SequenceMatcher(None, given_str, substring).ratio()
        if ratio > max_ratio:
            max_ratio = ratio
            best_match = substring
    return best_match