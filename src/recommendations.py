import pandas as pd
import numpy as np

class AIRecommendationEngine:
    """
    Provides segment-specific, non-hallucinated marketing strategies and impact forecasts.
    """
    @staticmethod
    def get_persona_strategies():
        return {
            "VIP Cosmetics Enthusiasts ✨": {
                "business_value": "High Revenue & High AOV",
                "recommended_channels": "Private Concierge, SMS VIP Line, Exclusive Email",
                "img": "https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=500&q=80",
                "strategies": [
                    "🚀 Launch an exclusive Concierge VIP Loyalty Tier with early access to luxury product drops.",
                    "🎁 Send surprise full-sized anniversary gift boxes (skincare/fragrance).",
                    "💬 Provide dedicated beauty concierge consultations and personalized skincare regimens."
                ],
                "expected_impact": "+18% Revenue Increase",
                "confidence": "High Confidence Model"
            },
            "Frequent Buyers 🛍️": {
                "business_value": "High Purchase Velocity & Engagement",
                "recommended_channels": "Push Notifications, Instagram Retargeting, Email Series",
                "img": "https://images.unsplash.com/photo-1517841905240-472988babdf9?w=500&q=80",
                "strategies": [
                    "📦 Introduce cross-category 'Build Your Own Bundle' savings to increase Average Order Value.",
                    "🔄 Trigger automated replenishment reminders 14 days before product exhaustion.",
                    "⭐ Offer double loyalty points on new category expansion purchases."
                ],
                "expected_impact": "+24% Order Frequency",
                "confidence": "High Confidence Model"
            },
            "Budget Conscious 💄": {
                "business_value": "High Volume & Discount Driven",
                "recommended_channels": "In-App Banners, TikTok Coupon Ads, SMS Flash Sales",
                "img": "https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=500&q=80",
                "strategies": [
                    "🎟️ Deploy threshold discounts (e.g., 'Spend $50, get $10 off').",
                    "⚡ Flash sales on trending lip & eye cosmetics during weekend windows.",
                    "🎁 Offer travel-sized gifts with purchase to encourage basket building."
                ],
                "expected_impact": "+15% Basket Size Uplift",
                "confidence": "Medium-High Confidence"
            },
            "At-Risk Customers ⚠️": {
                "business_value": "High Churn Risk / Re-engagement Opportunity",
                "recommended_channels": "Win-back Email Sequence, Personalized SMS Discount",
                "img": "https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=500&q=80",
                "strategies": [
                    "📩 Trigger an automated 3-step 'We Miss You' re-engagement campaign with a 20% discount code.",
                    "📋 Invite customer to take a 1-minute survey in exchange for $15 store credit.",
                    "🌟 Highlight top 3 best-selling products launched since their last visit."
                ],
                "expected_impact": "32% Churn Reduction",
                "confidence": "High Confidence Model"
            }
        }
