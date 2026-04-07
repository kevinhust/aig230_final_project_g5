"""
Generate Knowledge Base from Amazon Reviews Multi Dataset

Downloads mteb/amazon_reviews_multi from HuggingFace, filters for
customer-service relevant reviews (1-2 star, product issues), and converts
them into FAQ-format markdown documents for the RAG knowledge base.

Languages: en, zh, fr, es
Target: ~200-500 reviews per language → organized into category-based FAQ files
"""

import os
import re
import random
from datasets import load_dataset
from collections import defaultdict

# Configuration
LANGUAGES = ['en', 'zh', 'fr', 'es']
LANG_NAMES = {'en': 'English', 'zh': '中文', 'fr': 'Français', 'es': 'Español'}
MAX_REVIEWS_PER_LANG = 500
MIN_TEXT_LENGTH = 50
KB_DIR = "kb"

# Product category keywords for classification
CATEGORY_KEYWORDS = {
    'Appliances': {
        'en': ['fridge', 'refrigerator', 'washer', 'dryer', 'dishwasher', 'microwave',
               'oven', 'stove', 'freezer', 'blender', 'toaster', 'coffee maker'],
        'zh': ['冰箱', '洗衣机', '微波炉', '烤箱', '洗碗机', '搅拌机', '电器'],
        'fr': ['frigo', 'réfrigérateur', 'lave-linge', 'four', 'micro-ondes', 'lave-vaisselle'],
        'es': ['frigorífico', 'nevera', 'lavadora', 'horno', 'microondas', 'lavavajillas']
    },
    'Electronics': {
        'en': ['phone', 'tablet', 'laptop', 'headphones', 'speaker', 'camera',
               'monitor', 'keyboard', 'mouse', 'charger', 'battery', 'screen'],
        'zh': ['手机', '平板', '电脑', '耳机', '音箱', '相机', '屏幕', '电池'],
        'fr': ['téléphone', 'tablette', 'ordinateur', 'écouteurs', 'enceinte', 'batterie'],
        'es': ['teléfono', 'tableta', 'portátil', 'auriculares', 'altavoz', 'batería']
    },
    'Furniture': {
        'en': ['chair', 'table', 'desk', 'sofa', 'couch', 'bed', 'shelf', 'cabinet',
               'drawer', 'dresser', 'bookcase'],
        'zh': ['椅子', '桌子', '沙发', '床', '柜子', '书架'],
        'fr': ['chaise', 'table', 'bureau', 'canapé', 'lit', 'étagère'],
        'es': ['silla', 'mesa', 'escritorio', 'sofá', 'cama', 'estantería']
    },
    'Clothing': {
        'en': ['shirt', 'pants', 'dress', 'jacket', 'shoes', 'socks', 'coat',
               'sweater', 'jeans', 'size'],
        'zh': ['衬衫', '裤子', '裙子', '外套', '鞋子', '尺码', '衣服'],
        'fr': ['chemise', 'pantalon', 'robe', 'veste', 'chaussures', 'taille'],
        'es': ['camisa', 'pantalón', 'vestido', 'chaqueta', 'zapatos', 'talla']
    },
    'Kitchen': {
        'en': ['knife', 'pan', 'pot', 'cutting board', 'spatula', 'utensil',
               'kitchen', 'cookware', 'dishes', 'plate', 'bowl', 'cup'],
        'zh': ['刀', '锅', '厨具', '碗', '盘子', '厨房'],
        'fr': ['couteau', 'poêle', 'casserole', 'cuisine', 'ustensile'],
        'es': ['cuchillo', 'sartén', 'cacerola', 'cocina', 'utensilio']
    },
    'SmartHome': {
        'en': ['smart', 'wifi', 'bluetooth', 'alexa', 'google home', 'ring',
               'thermostat', 'sensor', 'hub', 'connect', 'app'],
        'zh': ['智能', '蓝牙', '路由', '传感器', '设备'],
        'fr': ['intelligent', 'connecté', 'domotique', 'capteur'],
        'es': ['inteligente', 'conectado', 'domótica', 'sensor']
    },
    'Headphones': {
        'en': ['headphone', 'earbuds', 'earphone', 'airpods', 'anc', 'noise cancel',
               'bluetooth earbuds', 'wireless'],
        'zh': ['耳机', '耳塞', '蓝牙耳机', '降噪'],
        'fr': ['écouteur', 'casque', 'sans fil'],
        'es': ['auricular', 'casco', 'inalámbrico']
    }
}

# Customer service issue keywords
ISSUE_KEYWORDS = {
    'broken': {
        'en': ['broken', 'broke', 'cracked', 'defective', 'damaged', 'snapped', 'shattered'],
        'zh': ['坏', '碎', '裂', '破损', '缺陷', '故障'],
        'fr': ['cassé', 'brisé', 'défectueux', 'endommagé', 'fissuré'],
        'es': ['roto', 'defectuoso', 'dañado', 'agrietado']
    },
    'quality': {
        'en': ['poor quality', 'cheap', 'flimsy', 'waste', 'junk', 'garbage', 'terrible'],
        'zh': ['质量差', '劣质', '浪费', '垃圾', '太差'],
        'fr': ['mauvaise qualité', 'cheap', 'poubelle', 'nul'],
        'es': ['mala calidad', 'barato', 'basura', 'terrible']
    },
    'not_working': {
        'en': ["doesn't work", 'not working', 'stopped', 'stopped working', 'failed',
               'malfunction', 'won\'t turn on', 'won\'t charge'],
        'zh': ['不工作', '不能用', '停止工作', '无法', '失灵'],
        'fr': ['ne fonctionne pas', 'ne marche pas', 'arrêté', 'panne'],
        'es': ['no funciona', 'no funciona', 'parado', 'averiado']
    },
    'wrong_item': {
        'en': ['wrong', 'different', 'not as described', 'not what', 'misleading',
               'incorrect', 'missing parts'],
        'zh': ['错', '不对', '不符', '描述不符', '缺件'],
        'fr': ['mauvais', 'différent', 'pas comme décrit', 'pièces manquantes'],
        'es': ['equivocado', 'diferente', 'no como se describe', 'piezas faltantes']
    },
    'shipping': {
        'en': ['shipping', 'delivery', 'arrived', 'late', 'delayed', 'never received',
               'tracking', 'package', 'lost'],
        'zh': ['发货', '配送', '延迟', '没收到', '物流', '快递'],
        'fr': ['livraison', 'expédition', 'retard', 'jamais reçu', 'colis'],
        'es': ['envío', 'entrega', 'retraso', 'nunca recibí', 'paquete']
    }
}


def classify_category(text, lang):
    """Classify a review into a product category."""
    text_lower = text.lower()
    scores = {}
    for category, lang_keywords in CATEGORY_KEYWORDS.items():
        score = 0
        keywords = lang_keywords.get(lang, lang_keywords.get('en', []))
        for kw in keywords:
            if kw.lower() in text_lower:
                score += 1
        scores[category] = score

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'General'


def detect_issue_type(text, lang):
    """Detect the type of customer service issue."""
    text_lower = text.lower()
    for issue_type, lang_keywords in ISSUE_KEYWORDS.items():
        keywords = lang_keywords.get(lang, lang_keywords.get('en', []))
        for kw in keywords:
            if kw.lower() in text_lower:
                return issue_type
    return 'general'


def review_to_faq(review, lang, index):
    """Convert a review into a FAQ-style entry."""
    text = review['text']
    stars = review['label']
    review_id = review['id']

    # Clean up text
    text = text.strip()
    lines = text.split('\n')
    title = lines[0].strip() if lines else ''
    body = '\n'.join(lines[1:]).strip() if len(lines) > 1 else ''

    category = classify_category(text, lang)
    issue_type = detect_issue_type(text, lang)

    # Create FAQ entry
    faq = {
        'category': category,
        'issue_type': issue_type,
        'stars': stars,
        'lang': lang,
        'review_id': review_id,
        'title': title,
        'body': body,
        'full_text': text
    }
    return faq


def generate_faq_document(faqs, category, lang):
    """Generate a markdown FAQ document from reviews."""
    lang_name = LANG_NAMES.get(lang, lang)
    lines = [
        f"# FAQ - {category} ({lang_name})",
        f"",
        f"Category: {category}",
        f"Language: {lang}",
        f"Source: Amazon Reviews Multi (mteb/amazon_reviews_multi)",
        f"",
        f"---",
        f""
    ]

    # Group by issue type
    by_issue = defaultdict(list)
    for faq in faqs:
        by_issue[faq['issue_type']].append(faq)

    for issue_type, issue_faqs in by_issue.items():
        lines.append(f"## {issue_type.replace('_', ' ').title()}")
        lines.append("")

        for i, faq in enumerate(issue_faqs[:15]):  # Max 15 per issue type
            title = faq['title'][:100] if faq['title'] else 'Customer Issue'
            body = faq['body'][:300] if faq['body'] else ''

            # Create Q&A from review
            q_text = title if title else body[:100]
            lines.append(f"**Q: {q_text}**")
            lines.append(f"A: {body[:200]}")
            lines.append(f"*Source: Amazon Review ({faq['stars']-0}★) [{faq['review_id']}]*")
            lines.append("")

        lines.append("---")
        lines.append("")

    return '\n'.join(lines)


def main():
    print("=" * 60)
    print("Amazon Reviews Multi → Knowledge Base Generator")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    ds = load_dataset('mteb/amazon_reviews_multi', split='train',
                       revision='refs/convert/parquet')
    print(f"Total reviews: {len(ds):,}")

    os.makedirs(KB_DIR, exist_ok=True)

    total_generated = 0

    for lang in LANGUAGES:
        lang_name = LANG_NAMES[lang]
        print(f"\n{'='*40}")
        print(f"Processing {lang_name} ({lang})")

        # Filter for this language, low stars, meaningful length
        lang_reviews = ds.filter(
            lambda x: x['id'].startswith(f'{lang}_') and x['label'] <= 2 and len(x['text']) >= MIN_TEXT_LENGTH,
            num_proc=4
        )
        print(f"  Relevant reviews: {len(lang_reviews):,}")

        # Sample
        n_samples = min(MAX_REVIEWS_PER_LANG, len(lang_reviews))
        indices = random.sample(range(len(lang_reviews)), n_samples)
        sampled = lang_reviews.select(indices)

        # Convert to FAQs and group by category
        category_faqs = defaultdict(list)
        for review in sampled:
            faq = review_to_faq(review, lang, 0)
            category_faqs[faq['category']].append(faq)

        print(f"  Categories: {dict((k, len(v)) for k, v in category_faqs.items())}")

        # Generate FAQ documents
        for category, faqs in category_faqs.items():
            if len(faqs) < 3:
                continue  # Skip categories with too few entries

            filename = f"FAQ-{category}-{lang.upper()}.md"
            filepath = os.path.join(KB_DIR, filename)

            content = generate_faq_document(faqs, category, lang)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            total_generated += 1
            print(f"  Generated: {filename} ({len(faqs)} entries)")

    print(f"\n{'='*60}")
    print(f"Done! Generated {total_generated} FAQ documents in {KB_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(42)  # Reproducibility
    main()
