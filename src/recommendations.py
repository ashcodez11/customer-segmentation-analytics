import pandas as pd
import numpy as np

class AIRecommendationEngine:
    """
    Generates actionable marketing playbooks & strategic recommendations
    for each customer persona and RFM cluster.
    """
    @staticmethod
    def get_persona_strategies():
        return {
            "VIP Cosmetics Enthusiasts ✨": {
                "business_value": "High Revenue & High Frequency",
                "recommended_channels": "SMS VIP Line, Direct Concierge, Email",
                "strategies": [
                    "🚀 Launch an exclusive VIP Loyalty Tier with early access to luxury product drops.",
                    "🎁 Send surprise full-sized anniversary gift boxes (skincare/fragrance).",
                    "💬 Provide dedicated beauty concierge consultation and personalized skincare regimens."
                ]
            },
            "Frequent Budget Buyers 🛍️": {
                "business_value": "High Volume, Price Sensitive",
                "recommended_channels": "Push Notifications, Email, Instagram Retargeting",
                "strategies": [
                    "📦 Introduce cross-category 'Build Your Own Bundle' savings to increase Average Order Value.",
                    "🎟️ Deploy threshold discounts (e.g., 'Spend $60, get $15 off').",
                    "⚡ Flash sales on trending lip & eye cosmetics during weekend windows."
                ]
            },
            "At-Risk / Inactive Shoppers ⚠️": {
                "business_value": "Dormant High Value",
                "recommended_channels": "Automated Win-back Email, Targeted Paid Social",
                "strategies": [
                    "📩 Trigger a automated 3-step 'We Miss You' re-engagement campaign with 20% discount.",
                    "🔄 Highlight product replacements for items purchased > 180 days ago (replenishment reminder).",
                    "📋 Invite customer to take a 1-minute survey in exchange for store credit."
                ]
            },
            "Occasional / New Shoppers 🌱": {
                "business_value": "Unlocking Long-term CLV Potential",
                "recommended_channels": "Email Welcome Series, TikTok / IG Ads",
                "strategies": [
                    "🎓 Send an educational 4-part 'Beauty & Skincare Routine' onboarding email series.",
                    "🎁 Offer a free travel-size sample on their second purchase.",
                    "⭐ Incentivize first product review with 100 loyalty points."
                ]
            }
        }

    @staticmethod
    def get_rfm_strategies():
        return {
            "Champions": "Reward loyalty with VIP events, early access, and brand ambassador invitations.",
            "Loyal Customers": "Upsell premium product lines and encourage subscription auto-replenishment.",
            "Potential Loyalists": "Offer bundle discounts and loyalty program enrollment triggers.",
            "New / Promising": "Deliver welcome onboard sequence and highlight best-seller bundles.",
            "Need Attention": "Send limited-time personalized discount codes to reactivate engagement.",
            "At Risk": "Send personalized re-engagement offers and product recommendations based on past purchases.",
            "Can't Lose Them": "Provide phone/concierge outreach, heavy renewal incentives, and direct support.",
            "Lost Customers": "Include in low-cost seasonal email campaigns; test aggressive discount win-backs."
        }

if __name__ == "__main__":
    strategies = AIRecommendationEngine.get_persona_strategies()
    for persona, details in strategies.items():
        print(f"\n[{persona}]")
        print(f"Value: {details['business_value']}")
        for s in details['strategies']:
            print(f"  - {s}")
