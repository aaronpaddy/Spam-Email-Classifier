# preprocess.py

import pandas as pd
import re

def preprocess_email(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove unwanted characters (e.g., punctuation, numbers)
    text = re.sub(r'[^\w\s]', '', text)
    
    # Feature extraction based on word frequencies
    features = {
        "word_freq_make": text.count('make'),
        "word_freq_address": text.count('address'),
        "word_freq_all": text.count('all'),
        "word_freq_3d": text.count('3d'),
        "word_freq_our": text.count('our'),
        "word_freq_over": text.count('over'),
        "word_freq_remove": text.count('remove'),
        "word_freq_internet": text.count('internet'),
        "word_freq_order": text.count('order'),
        "word_freq_mail": text.count('mail'),
        "word_freq_receive": text.count('receive'),
        "word_freq_will": text.count('will'),
        "word_freq_people": text.count('people'),
        "word_freq_report": text.count('report'),
        "word_freq_addresses": text.count('addresses'),
        "word_freq_free": text.count('free'),
        "word_freq_business": text.count('business'),
        "word_freq_email": text.count('email'),
        "word_freq_you": text.count('you'),
        "word_freq_credit": text.count('credit'),
        "word_freq_your": text.count('your'),
        "word_freq_font": text.count('font'),
        "word_freq_000": text.count('000'),
        "word_freq_money": text.count('money'),
        "word_freq_hp": text.count('hp'),
        "word_freq_hpl": text.count('hpl'),
        "word_freq_george": text.count('george'),
        "word_freq_650": text.count('650'),
        "word_freq_lab": text.count('lab'),
        "word_freq_labs": text.count('labs'),
        "word_freq_telnet": text.count('telnet'),
        "word_freq_857": text.count('857'),
        "word_freq_data": text.count('data'),
        "word_freq_415": text.count('415'),
        "word_freq_85": text.count('85'), 
        "word_freq_technology": text.count('technology'),
        "word_freq_1999": text.count('1999'),
        "word_freq_parts": text.count('parts'),
        "word_freq_pm": text.count('pm'),
        "word_freq_direct": text.count('direct'),
        "word_freq_cs": text.count('cs'),
        "word_freq_meeting": text.count('meeting'),
        "word_freq_original": text.count('original'),
        "word_freq_project": text.count('project'),
        "word_freq_re": text.count('re'),
        "word_freq_edu": text.count('edu'),
        "word_freq_table": text.count('table'),
        "word_freq_conference": text.count('conference'),
        "char_freq_;": text.count(';'),
        "char_freq_(" : text.count('('),
        "char_freq_[": text.count('['),
        "char_freq_!": text.count('!'),
        "char_freq_$": text.count('$'),
        "char_freq_#": text.count('#'),
        "capital_run_length_average": sum(len(word) for word in text.split() if word.isupper()) / max(1, len(text.split())),
        "capital_run_length_longest": max(len(word) for word in text.split() if word.isupper()) if any(word.isupper() for word in text.split()) else 0,
        "capital_run_length_total": sum(len(word) for word in text.split() if word.isupper())
    }

    # Return the features as a DataFrame (1 row, 57 columns)
    return pd.DataFrame([features])
