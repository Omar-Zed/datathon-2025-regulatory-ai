# reg_analysis.py - Updated for better multilingual support (basic keyword expansion)
import re
import pandas as pd
import xml.etree.ElementTree as ET

def clean_text(reg_text, file_extension):
    reg_text = re.sub(r'\s+', ' ', reg_text)  # Normalize whitespace
    
    if file_extension.lower() in ['html', 'htm']:
        reg_text = re.sub(r'<[^>]+>', ' ', reg_text)  # Remove HTML tags
        reg_text = re.sub(r'&[^;]+;', ' ', reg_text)  # Remove entities
    
    elif file_extension.lower() == 'xml':
        try:
            root = ET.fromstring(reg_text)
            reg_text = ' '.join([elem.text.strip() for elem in root.iter() if elem.text])
        except ET.ParseError:
            reg_text = re.sub(r'<[^>]+>', ' ', reg_text)
    
    return reg_text

def extract_reg_info(reg_text, file_extension='txt'):
    reg_text = clean_text(reg_text, file_extension)
    
    # Expanded entities (add multilingual keywords if possible; regex for Latin/English terms in docs)
    entities_pattern = r'\b(US|USA|United States|China|EU|European Union|Europe|Japan|pharma|pharmaceutical|tech|technology|energy|corporations|sectors|countries|AI|artificial intelligence|consumer|protection|inflation|reduction|act|directive|regulation|law|promotion|advancement|research|development|exploitation|carbon|neutrality|drought|relief|environmental|impact|assessment|renewable|sources|人工知能|能源法|directive|regulation|parlement|conseil|penalties|risks|classification|transparency|cybersecurity)\b'
    entities = set(re.findall(entities_pattern, reg_text, re.I))
    
    # Dates
    dates_pattern = r'\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}|after\s+\w+\s+\d{1,2},\s+\d{4}|\d{4})\b'
    dates = list(set(re.findall(dates_pattern, reg_text, re.I)))
    
    # Measures
    measures_pattern = r'(tax|rebates|negotiations|tariffs|measures|directive|regulation|act|law|promotion|advancement|research|development|exploitation|energy|AI|artificial intelligence|consumer rights|penalties|prohibitions|safety|risks|classification|transparency|cybersecurity|environmental|impact|assessment|carbon|emissions|renewable|sources|subsidies|fines|sanctions|compliance|reporting|obligations|drought|relief|mitigation|financial assistance)'
    measures = list(set(re.findall(measures_pattern, reg_text, re.I)))
    
    # Type inference
    lower_text = reg_text.lower()
    if 'tax' in lower_text or 'inflation' in lower_text or 'reduction act' in lower_text:
        type_reg = 'Taxation / Economic'
    elif 'price' in lower_text or 'consumer' in lower_text or 'directive' in lower_text:
        type_reg = 'Consumer Protection'
    elif 'energy' in lower_text or '碳' in lower_text:
        type_reg = 'Energy / Environmental'
    elif 'ai' in lower_text or 'artificial intelligence' in lower_text or '人工知能' in lower_text:
        type_reg = 'AI Promotion / Regulation'
    else:
        type_reg = 'Other'
    
    return {'entities': entities, 'dates': dates, 'measures': measures, 'type_reg': type_reg}

# analyze_reg_impact remains the same
def analyze_reg_impact(portfolio_df, reg_text, file_extension='txt'):
    # Same as before...
    pass  # (keep the original function)