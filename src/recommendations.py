class AIRecommendationEngine:
    @staticmethod
    def get_persona_strategies():
        return {
            'VIP Cosmetics Enthusiasts': {
                'business_value': 'Highest Lifetime Value & Brand Advocacy',
                'recommended_channels': 'Private Client Concierge, VIP App Invites, SMS',
                'strategies': [
                    'Offer exclusive early access to limited-edition formulations',
                    'Provide complimentary 1-on-1 consultations with Master Estheticians',
                    'Enrol automatically in double-tier loyalty rewards'
                ],
                'expected_impact': '+18% Spend Growth',
                'confidence': '95% Confidence (Data-backed)',
                'img': 'https://images.unsplash.com/photo-1534528741775-53994a69daeb?w=400&q=80'
            },
            'Frequent Buyers': {
                'business_value': 'High Order Volume & Consistent Cash Flow',
                'recommended_channels': 'Email Newsletters, Push Notifications',
                'strategies': [
                    'Introduce replenishment subscription discounts (Save 15%)',
                    'Cross-sell complimentary routine categories (e.g. Serum + Eye Cream)',
                    'Gamify purchase milestones with gift-with-purchase tiers'
                ],
                'expected_impact': '+24% Purchase Frequency',
                'confidence': '92% Confidence (Data-backed)',
                'img': 'https://images.unsplash.com/photo-1517841905240-472988babdf9?w=400&q=80'
            },
            'Budget Conscious': {
                'business_value': 'Price-Sensitive Volume Base',
                'recommended_channels': 'Social Retargeting, Flash Sale Emails',
                'strategies': [
                    'Promote value-bundle sets and travel mini-sizes',
                    'Deploy timed cashback discount vouchers for abandoned carts',
                    'Highlight seasonal promotional campaigns and clearance events'
                ],
                'expected_impact': '+15% Conversion Rate',
                'confidence': '88% Confidence (Data-backed)',
                'img': 'https://images.unsplash.com/photo-1524504388940-b1c1722653e1?w=400&q=80'
            },
            'At-Risk Customers': {
                'business_value': 'Dormant High-Potential Revenue',
                'recommended_channels': 'Win-back Email Series, Direct Mailer Vouchers',
                'strategies': [
                    'Trigger personalized "We Miss You" 20% discount incentives',
                    'Send feedback surveys to identify dissatisfaction causes',
                    'Offer personalized product samples based on past category preference'
                ],
                'expected_impact': '32% Churn Reduction',
                'confidence': '90% Confidence (Data-backed)',
                'img': 'https://images.unsplash.com/photo-1494790108377-be9c29b29330?w=400&q=80'
            }
        }
