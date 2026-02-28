import os
import json

def generate_kb():
    # Synthetic FAQ documents for different categories in multiple languages
    faq_data = [
        {
            "category": "Appliances",
            "title": "Refrigerator Warranty & Returns (冰箱保修与退货)",
            "content": """
            Q: 我的 fridge 噪音大，保修怎么走？ (My fridge is noisy, how does warranty work?)
            A: 我们的所有电器提供一年有限保修。如果您的冰箱产生异常噪音，请先检查放置是否平稳。如需保修服务，请提供订单号联系客服。
            (All our appliances come with a 1-year limited warranty. If your fridge is making unusual noise, please check if it is level. For warranty service, contact support with your order ID.)
            
            Q: How to request a return for a broken washer?
            A: Returns are accepted within 30 days of delivery. The item must be in original packaging. 
            For broken units, we provide a free pre-paid shipping label.
            """,
            "source": "FAQ-Appliances.md",
            "lang": "zh-en"
        },
        {
            "category": "Electronics",
            "title": "Smartphone Troubleshooting (手机故障排除)",
            "content": """
            Q: Mon téléphone ne s'allume pas, que faire ? (My phone won't turn on, what to do?)
            A: Essayez de le charger pendant au moins 30 minutes avec le chargeur d'origine. Si le problème persiste, contactez le support technique.
            (Try charging it for at least 30 minutes with the original charger. If the problem persists, contact technical support.)
            
            Q: ¿Cómo solicito una devolución de mi smartphone?
            A: Las devoluciones se pueden solicitar a través de su portal de cliente dentro de los 14 días. 
            El dispositivo debe estar en estado 'como nuevo'.
            """,
            "source": "FAQ-Electronics.md",
            "lang": "fr-es"
        },
        {
            "category": "Shipping",
            "title": "Global Shipping Policy",
            "content": """
            Q: Do you ship to Canada?
            A: Yes, we ship to all provinces in Canada. Standard shipping takes 3-5 business days.
            
            Q: 你们支持发货到多伦多吗？
            A: 是的，我们支持发货到多伦多及全加拿大地区。
            """,
            "source": "Shipping-Policy.md",
            "lang": "en-zh"
        }
    ]

    os.makedirs("kb", exist_ok=True)
    
    for doc in faq_data:
        file_path = f"kb/{doc['source']}"
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"# {doc['title']}\n\nCategory: {doc['category']}\nLanguage: {doc['lang']}\n\n{doc['content']}")
        print(f"Generated {file_path}")

if __name__ == "__main__":
    generate_kb()
