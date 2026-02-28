"""
Sentiment Analysis Module for E-Commerce Customer Support

This module provides sentiment analysis for customer queries,
focusing on detecting angry/frustrated customers for escalation.
"""

import re
from typing import Tuple, Dict, Optional

# Configuration
SENTIMENT_ANGRY_THRESHOLD = -0.3
SENTIMENT_ESCALATION_THRESHOLD = -0.6

# Multilingual angry keywords
ANGRY_KEYWORDS = {
    'en': [
        'unacceptable', 'terrible', 'worst', 'scam', 'lawsuit', 'manager',
        'ridiculous', 'awful', 'hate', 'disgusting', 'furious', 'angry',
        'horrible', 'disaster', 'never again', 'waste', 'disappointed',
        'frustrated', 'annoyed', 'fed up', 'enough', 'joke'
    ],
    'zh': [
        '离谱', '骗子', '投诉', '报警', '差评', '垃圾', '不可接受',
        '愤怒', '失望', '糟糕', '恶心', '骗人', '退货', '退款',
        '再也不', '浪费', '无语', '受不了', '太过分', '欺人太甚'
    ],
    'fr': [
        'inacceptable', 'terrible', 'pire', 'arnaque', 'colère',
        'horrible', 'déçu', 'nul', 'honteux', 'scandaleux',
        'révoltant', 'insupportable', 'jamais plus'
    ],
    'es': [
        'inaceptable', 'terrible', 'peor', 'estafa', 'denuncia',
        'horrible', 'decepcionado', 'basura', ' vergonzoso',
        'nunca más', 'fraude', ' indignado'
    ]
}

# Escalation trigger keywords
ESCALATION_KEYWORDS = {
    'en': [
        'speak to manager', 'talk to manager', 'see manager',
        'lawsuit', 'sue', 'lawyer', 'attorney',
        'bbb', 'better business bureau',
        'social media', 'twitter', 'facebook', 'instagram',
        'review', 'negative review', 'bad review',
        'report', 'complain', 'formal complaint'
    ],
    'zh': [
        '经理', '主管', '领导',
        '报警', '律师', '法院',
        '曝光', '媒体', '微博', '朋友圈',
        '投诉', '举报', '差评'
    ],
    'fr': [
        'parler au responsable', 'voir le responsable', 'manager',
        'avocat', 'procès', 'tribunal',
        'réseaux sociaux', 'twitter', 'facebook',
        'plainte officielle', 'signaler'
    ],
    'es': [
        'hablar con el gerente', 'ver al gerente', 'manager',
        'abogado', 'demanda', 'juicio',
        'redes sociales', 'twitter', 'facebook',
        'queja oficial', 'denunciar'
    ]
}

# Apology templates by language
APOLOGY_TEMPLATES = {
    'English': "I sincerely apologize for this inconvenience. ",
    '中文': "非常抱歉给您带来不便。 ",
    'Français': "Je vous présente mes sincères excuses pour ce désagrément. ",
    'Español': "Le pido sinceras disculpas por este inconveniente. ",
    'Deutsch': "Ich entschuldige mich aufrichtig für die Unannehmlichkeiten. ",
    'Italiano': "Le porgo le mie sincere scuse per l'inconveniente. ",
    'Português': "Peço sinceras desculpas pelo inconveniente. ",
    'default': "I sincerely apologize for this inconvenience. "
}

# Escalation responses by language
ESCALATION_RESPONSES = {
    'English': (
        "I understand your frustration and want to make this right immediately. "
        "I'm connecting you to our senior support team who can better assist with this issue. "
        "Please hold briefly."
    ),
    '中文': (
        "我完全理解您的不满，我们会立即处理这个问题。"
        "我将为您转接高级客服团队，他们能更好地帮助您。请稍等。"
    ),
    'Français': (
        "Je comprends votre frustration et je veux régler cela immédiatement. "
        "Je vous connecte à notre équipe senior qui pourra mieux vous aider. Veuillez patienter."
    ),
    'Español': (
        "Entiendo su frustración y quiero solucionar esto inmediatamente. "
        "Lo conectaré con nuestro equipo senior que podrá ayudarlo mejor. Por favor espere."
    ),
    'default': (
        "I understand your frustration. I'm connecting you to our senior support team. "
        "Please hold briefly."
    )
}


def analyze_sentiment(query: str) -> Dict:
    """
    Analyze the sentiment of a customer query.

    Args:
        query: The customer's message

    Returns:
        dict with keys:
            - score: float from -1 (very angry) to 1 (very happy)
            - is_angry: bool, True if score < threshold
            - escalation_needed: bool, True if escalation triggers detected
            - angry_signals: list of detected angry keywords
            - escalation_signals: list of detected escalation keywords
    """
    query_lower = query.lower()

    # Detect angry keywords
    angry_signals = []
    for lang, keywords in ANGRY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                angry_signals.append(keyword)

    # Detect escalation keywords
    escalation_signals = []
    for lang, keywords in ESCALATION_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                escalation_signals.append(keyword)

    # Calculate sentiment score
    sentiment_score = 0.0

    # Penalty for angry keywords
    sentiment_score -= len(angry_signals) * 0.15

    # Penalty for excessive punctuation
    exclamation_count = query.count('!')
    question_marks = query.count('?')
    sentiment_score -= min(exclamation_count * 0.05, 0.2)
    sentiment_score -= min(question_marks * 0.02, 0.1)

    # Penalty for ALL CAPS ratio
    if len(query) > 0:
        caps_ratio = sum(1 for c in query if c.isupper()) / len(query)
        if caps_ratio > 0.3:  # More than 30% caps is angry signal
            sentiment_score -= min(caps_ratio * 0.5, 0.3)

    # Penalty for repeated characters (e.g., "!!!", "???")
    repeated_punct = len(re.findall(r'[!?]{2,}', query))
    sentiment_score -= repeated_punct * 0.1

    # Determine anger level
    is_angry = sentiment_score < SENTIMENT_ANGRY_THRESHOLD
    escalation_needed = (
        len(escalation_signals) > 0 or
        sentiment_score < SENTIMENT_ESCALATION_THRESHOLD
    )

    return {
        "score": round(sentiment_score, 2),
        "is_angry": is_angry,
        "escalation_needed": escalation_needed,
        "angry_signals": angry_signals[:5],  # Limit to top 5
        "escalation_signals": escalation_signals[:3]
    }


def get_apology_prefix(language: str) -> str:
    """Get appropriate apology prefix based on language."""
    return APOLOGY_TEMPLATES.get(language, APOLOGY_TEMPLATES['default'])


def get_escalation_response(language: str) -> str:
    """Get appropriate escalation response based on language."""
    return ESCALATION_RESPONSES.get(language, ESCALATION_RESPONSES['default'])


def format_sentiment_badge(sentiment_result: Dict) -> str:
    """
    Format sentiment result as a visual badge for UI.

    Returns emoji-based sentiment indicator.
    """
    score = sentiment_result['score']
    is_angry = sentiment_result['is_angry']
    escalation = sentiment_result['escalation_needed']

    if escalation:
        return "🔴 Escalation"
    elif is_angry:
        return "🟠 Frustrated"
    elif score < 0:
        return "🟡 Concerned"
    else:
        return "🟢 Neutral"


class SentimentAnalyzer:
    """
    Class-based sentiment analyzer for more advanced usage.
    """

    def __init__(self, angry_threshold: float = SENTIMENT_ANGRY_THRESHOLD,
                 escalation_threshold: float = SENTIMENT_ESCALATION_THRESHOLD):
        self.angry_threshold = angry_threshold
        self.escalation_threshold = escalation_threshold

    def analyze(self, query: str) -> Dict:
        """Analyze sentiment of a query."""
        return analyze_sentiment(query)

    def should_escalate(self, query: str) -> bool:
        """Quick check if query needs escalation."""
        result = self.analyze(query)
        return result['escalation_needed']

    def is_angry(self, query: str) -> bool:
        """Quick check if query shows anger."""
        result = self.analyze(query)
        return result['is_angry']


# Quick test
if __name__ == "__main__":
    test_queries = [
        "How do I return my order?",
        "This is UNACCEPTABLE! I want a refund NOW!!!",
        "我的订单两周没到，太离谱了！！！",
        "C'est terrible! Je veux parler au manager!",
        "Mi pedido no ha llegado, estoy muy decepcionado",
        "Thanks for your help!",
    ]

    print("=" * 60)
    print("Sentiment Analysis Test")
    print("=" * 60)

    for query in test_queries:
        result = analyze_sentiment(query)
        badge = format_sentiment_badge(result)
        print(f"\nQuery: {query}")
        print(f"Badge: {badge}")
        print(f"Score: {result['score']}")
        print(f"Angry: {result['is_angry']}, Escalate: {result['escalation_needed']}")
        if result['angry_signals']:
            print(f"Angry signals: {result['angry_signals']}")
        if result['escalation_signals']:
            print(f"Escalation signals: {result['escalation_signals']}")
